import time
from typing import List, Optional, Type

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.common.utils import rotate_round_z_axis

from .planner_utils import global_trajectory_to_states, load_checkpoint

"""
似乎只在模拟中使用: plantf_single_scenarios.sh 脚本中,planner=planTF 参数用于指定配置文件中的 planner 配置项为 planTF。
这意味着在运行 run_simulation.py 脚本时,
Hydra 会将 planner 配置项设置为 planTF,并根据 planTF 的配置来初始化和使用规划器。planTF\config\planner\planTF.yaml
而class PlanningModel(TorchModuleWrapper)似乎被直接用在train中:
    具体情况:在运行以下命令时：
        python run_training.py py_func=train +training=train_planTF
        Hydra 将加载 default_training.yaml 和 train_planTF.yaml 文件，并将它们合并。
        由于 train_planTF.yaml 中使用了 override /model: planTF,  
        Hydra 将加载 planTF\config\model\planTF.yaml 文件中的 model 配置，并覆盖默认的 model 配置。
        所以 run_training.py的main函数里面engine = build_training_engine(cfg, worker)调用的
           build_torch_module_wrapper(cfg.model) 的参数cfg.model 将包含 planTF\config\model\planTF.yaml中的 model 配置项:
           _target_: src.models.planTF.planning_model.PlanningModel,  
也就是说
    run_training.py中通过配置文件 直接调用PlanningModel
    run_simulation.py中通过配置文件 先调用ImitationPlanner间接调用PlanningModel

"""
class ImitationPlanner(AbstractPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _planning(self, current_input: PlannerInput):
        """"
        输入数据格式变化过程:current_input-> planner_feature-> planner_feature_torch(forward输入格式)-> out-> local_trajectory
        """
        self._start_time = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device(self.device)]
        )
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)

        out = self._planner.forward(planner_feature_torch.data)
        local_trajectory = out["output_trajectory"][0].cpu().numpy()

        return local_trajectory.astype(np.float64)

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        ego_state = current_input.history.ego_states[-1]

        if self._last_plan_elapsed_step >= self._replan_interval:
            local_trajectory = self._planning(current_input)
            self._global_trajectory = self._get_global_trajectory(
                local_trajectory, ego_state
            )
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        trajectory = InterpolatedTrajectory(
            trajectory=global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)

        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory
