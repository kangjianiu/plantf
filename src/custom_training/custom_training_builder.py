import logging
import os
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import cast

import pytorch_lightning as pl
from hydra.utils import instantiate
from nuplan.planning.script.builders.data_augmentation_builder import (
    build_agent_augmentor,
)
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_metrics_builder import (
    build_training_metrics,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from .custom_datamodule import CustomDataModule

logger = logging.getLogger(__name__)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.cache_path is None:
        logger.warning("Parameter cache_path is not set, caching is disabled")
    else:
        if not str(cfg.cache.cache_path).startswith("s3://"):
            if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
                rmtree(cfg.cache.cache_path)

            Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)

    if cfg.lightning.trainer.overfitting.enable:
        cfg.data_loader.params.num_workers = 0

    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(
            f"{prefix}_Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config..."
        )
        logger.info("\n" + OmegaConf.to_yaml(cfg))


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""

    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"{prefix}_<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = (
        build_agent_augmentor(cfg.data_augmentation)
        if "data_augmentation" in cfg
        else None
    )

    # Build dataset scenarios
    scenarios = build_scenarios(cfg, worker, model)

    # Create datamodule
    datamodule: pl.LightningDataModule = CustomDataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(
    cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper
) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Create the complete Module
    if "custom_trainer" in cfg:   
        
        """
        cfg.custom_trainer:  _target_: src.models.planTF.lightning_trainer.LightningTrainerclass 类:(pl.LightningModule)
        调用路径:
        python run_training.py py_func=train +training=train_planTF  ->  - override /custom_trainer: planTF 
                ->_target_: src.models.planTF.lightning_trainer.LightningTrainer(pl.LightningModule)

        如果包含 custom_trainer 配置项，使用 instantiate 函数根据配置实例化自定义训练器。
        instantiate 函数会根据 cfg.custom_trainer 中的 _target_ 字段指定的类来创建实例，
        并传递其他参数（如 model、lr、weight_decay、epochs 和 warmup_epochs）。
        """
        model = instantiate(
            cfg.custom_trainer,
            model=torch_module_wrapper,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
        )
    else:
        objectives = build_objectives(cfg)
        metrics = build_training_metrics(cfg)
        model = LightningModuleWrapper(
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
            if "warm_up_lr_scheduler" in cfg
            else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )

    return cast(pl.LightningModule, model)


def build_custom_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    # callbacks = build_callbacks(cfg)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            filename="{epoch}-{val_minFDE:.3f}",
            monitor=cfg.lightning.trainer.checkpoint.monitor,
            mode=cfg.lightning.trainer.checkpoint.mode,
            save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        PrintEpochEndResults(),
    ]

    if cfg.wandb.mode == "disable":
        training_logger = TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        )
    else:
        if cfg.wandb.artifact is not None:
            os.system(f"{prefix}_wandb artifact get {cfg.wandb.artifact}")
            _, _, artifact = cfg.wandb.artifact.split("/")
            checkpoint = os.path.join(os.getcwd(), f"{prefix}_artifacts/{artifact}/model.ckpt")
            run_id = artifact.split(":")[0][-8:]
            cfg.checkpoint = checkpoint
            cfg.wandb.run_id = run_id

        training_logger = WandbLogger(
            save_dir=cfg.group,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            resume=cfg.checkpoint is not None,
            id=cfg.wandb.run_id,
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=training_logger,
        **params,
    )

    return trainer


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info("Building training engine...")

    # 不用操心
    trainer = build_custom_trainer(cfg)

    # Create model  实例化自己实现的   PlanningModel    (TorchModuleWrapper)
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)
    #cfg.model :planTF\config\model\planTF.yaml:_target_: src.models.planTF.planning_model.PlanningModel(TorchModuleWrapper)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    # Build lightning module
    # 根据cfg.custom_trainer 和PlanningModel实例化自己实现的   LightningTrainer   类:(pl.LightningModule)
    model = build_lightning_module(cfg, torch_module_wrapper)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine

from pytorch_lightning.callbacks import Callback

import logging
from pytorch_lightning.callbacks import Callback

class PrintEpochEndResults(Callback):
    def __init__(self):
        super().__init__()
        

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_epoch_results(trainer, pl_module, prefix="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_epoch_results(trainer, pl_module, prefix="val")

    def log_epoch_results(self, trainer, pl_module, prefix: str):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        logger = logging.getLogger()

        # 构建日志消息，保留五位小数
        # 加上reg_loss，cls_loss，prediction_loss，diffusion_loss，输出

        log_message = (
            f"\n[Epoch {epoch}] "
            f"{prefix}_loss: {metrics.get(f'loss/{prefix}_loss', 0.0):.3f}, "
            f"{prefix}_reg_loss: {metrics.get(f'objectives/{prefix}_reg_loss', 0.0):.3f}, "
            f"{prefix}_cls_loss: {metrics.get(f'objectives/{prefix}_cls_loss', 0.0):.3f}, "
            f"{prefix}_pred_losss: {metrics.get(f'objectives/{prefix}_pred_loss', 0.0):.3f}, "
            f"{prefix}_anchor_reg_loss: {metrics.get(f'objectives/{prefix}_anchor_reg_loss', 0.0):.3f}, "
            f"{prefix}_anchor_cls_loss: {metrics.get(f'objectives/{prefix}_anchor_cls_loss', 0.0):.3f}, "
            f"{prefix}_MR: {metrics.get(f'{prefix}/MR', 0.0):.3f}, "
            f"{prefix}_minADE1: {metrics.get(f'{prefix}/minADE1', 0.0):.3f}, "
            f"{prefix}_minADE6: {metrics.get(f'{prefix}/minADE6', 0.0):.3f}, "
            f"{prefix}_minFDE1: {metrics.get(f'{prefix}/minFDE1', 0.0):.3f}, "
            f"{prefix}_minFDE6: {metrics.get(f'{prefix}/minFDE6', 0.0):.3f}"

        )

        # 使用简化的日志器输出
        logger.info(log_message)