import logging
import os
import sys
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR

logger = logging.getLogger(__name__)

"""
调用路径:
    python run_training.py py_func=train +training=train_planTF  ->  - override /custom_trainer: planTF 
            ->_target_: src.models.planTF.lightning_trainer.LightningTrainer

    class LightningTrainer(pl.LightningModule):(对 PlanningModel(TorchModuleWrapper) 进行包装)
    def __init__(
        self,
        model: TorchModuleWrapper,

作用:安装pl要求再封装一层,包装planning_model.py中的PlanningModel(TorchModuleWrapper)模型

在 LightningTrainer 类中，模仿学习的具体实现可以通过以下函数找到：

前向传播：在 _step 函数中，通过调用 self.forward(features["feature"].data) 实现。
计算损失：在 _step 函数中，通过调用 self._compute_objectives(res, features["feature"].data) 实现。
计算度量指标：在 _step 函数中，通过调用 self._compute_metrics(res, features["feature"].data, prefix) 实现。
记录损失和度量指标：在 _step 函数中，通过调用 self._log_step(losses["loss"], losses, metrics, prefix) 实现。
总结
在 LightningTrainer 类中，模仿学习的具体实现主要体现在 training_step、_step 和 on_fit_start 函数中。
通过前向传播、计算损失、计算度量指标和记录损失与度量指标，这些函数共同实现了模仿学习的训练过程。
"""
class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, ScenarioList = batch
        
        # Features[feature] __dict__['data'].keys(): ['agent', 'map', 'current_state', 'origin', 'angle']
        # Features[feature] __dict__['data']['agent']: ['position', 'heading', 'velocity', 'shape', 'category', 'valid_mask', 'target']

        res = self.forward(features["feature"].data)
        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, probability, prediction, diffusion_losses = (
            res["trajectory"],
            res["probability"],
            res["prediction"],
            res["diffusion_losses"],#[bs,num_modes]
        )
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:] # [32, 32, 80, 3],    [32, 32, 80]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())
  
        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )
        if isinstance(diffusion_losses, list):
            diffusion_loss = torch.stack(diffusion_losses).float().mean()
        else:
            diffusion_loss = diffusion_losses.mean()
        # print(f"ego_reg_loss: {ego_reg_loss}, ego_cls_loss: {ego_cls_loss}, agent_reg_loss: {agent_reg_loss}, diffusion_losses: {diffusion_loss}")
        # sys.exit(1)
        # ego_reg_loss: 14.728102684020996, ego_cls_loss: 1.7920138835906982, agent_reg_loss: 4.655642986297607, diffusion_losses: 14.878222465515137

        loss = ego_reg_loss + 0.2*ego_cls_loss + agent_reg_loss + diffusion_loss#加loss

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss,
            "cls_loss": ego_cls_loss,
            "pred_loss": agent_reg_loss,
            "diff_loss": diffusion_loss,
        }

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        # 我加的
        features, _, _ = batch
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )

        for module_name, module in self.named_modules():

            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                    if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                        print(f"===0=== {full_param_name} is in bias, ===no_decay")                    
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules) and full_param_name not in no_decay:
                        decay.add(full_param_name)
                        if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                            print(f"===1.1=== {full_param_name} is in white, ===decay")

                    elif isinstance(module, blacklist_weight_modules) and full_param_name not in decay:
                        no_decay.add(full_param_name)
                        if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                            print(f"===2=== {full_param_name} is in black, ===no_decay")                        
                       
                    elif "norm" in full_param_name or "emb" in full_param_name and full_param_name not in decay:
                        no_decay.add(full_param_name)
                        if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                            print(f"===3=== {full_param_name} is in norm or emb, ===no_decay")
                    
                    else:
                        if full_param_name not in no_decay:
                            decay.add(full_param_name)
                            if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                                print(f"===4=== {full_param_name} is in none of any class, ===decay")

                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
                    if full_param_name == "model.trajectory_decoder_diffu.probability_decoder.1.weight":
                        print(f"===5=== {full_param_name} is no weight or bias, ===no_decay")                    
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # 打印decay 和 no_decay，inter_params，union_params的长度
        print(f"=============\ndecay: {len(decay)}, no_decay: {len(no_decay)}, inter_params: {len(inter_params)}, union_params: {len(union_params)}")
        # 打印下面一行的两个参数数量是否等于所有参数数量
        print(f"param_dict.keys(): {len(param_dict.keys())}\n=====")
        #打印inter_params
        print(f"重复参数inter_params: {inter_params}")
        # 打印在param_dict里面但是不在union_params里面的参数名
        print(f"=============\n未分类参数param_dict.keys() - union_params: {param_dict.keys() - union_params}")
        assert len(inter_params) == 0

        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
