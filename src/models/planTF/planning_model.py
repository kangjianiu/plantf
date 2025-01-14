import sys
import torch
import torch.nn as nn
import numpy as np
import logging
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
from .diffusion_model import DiffusionModel
# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=20,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes

        # 包含的模块
        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.trajectory_decoder_diffu = DiffusionModel(feature_dim=dim, num_modes=num_modes, future_steps=future_steps)

        # self.trajectory_decoder = TrajectoryDecoder(
        #     embed_dim=dim,
        #     num_modes=num_modes,
        #     future_steps=future_steps,
        #     out_channels=4,
        # )
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.apply(self._init_weights)
        # self.print_model_info()
        # sys.exit(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def print_model_info(self):
        logger = logging.getLogger(__name__)
        logger.info("===== 模型层信息 =====")
        # print("===== 模型层信息 =====")
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 仅打印叶子节点
                num_params = sum(p.numel() for p in module.parameters())
                logger.info(f"层名称: {name}, 类型: {module.__class__.__name__}, 参数数量: {num_params}")
        logger.info("======================\n")
    
    def forward(self, data
                ):
        
        """
        lightning_trainer引用方法:
        def forward(self, features: FeaturesType) -> TargetsType
        res = self.forward(features["feature"].data)
        data的键:dict_keys(['agent', 'map', 'current_state', 'origin', 'angle'])
        data[agent]的键:[position, heading, velocity, shape, category, valid_mask, target]
        """
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        # 打印data[current_state]的键.结果是一个纯数字tensor

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)                             # x: [32, 255, 128]  (batch_size, num_agents + num_polygons, embed_dim)
        ego_instance_feature = x[:, 0].unsqueeze(1)  # [32, 1, 128]   主代理的嵌入表示,形状为 (batch_size, 1, embed_dim)
        map_instance_feature = x[:, A:]              # [32, 222, 128] 地图特征,形状为 (batch_size, num_polygons, embed_dim)


        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        # prediction  [32, 32, 80, 2] (batch_size, num_agents, future_steps, 2) 表示模型对其他代理未来状态的预测。

        # 策略1：读取256个锚点，[num_modes, bs, future_steps, 4]
        npy_file_path = '/data/datasets/niukangjia/plantf/traj_data/kmeans/cluster_centers_plan_style_20_80_vxy.npy'
        traj_anchors = self.load_cluster_centers(npy_file_path)# shape (20, 80, 4)
        traj_anchors = np.array(traj_anchors)  # 将列表转换为 numpy.ndarray


        traj_anchors = torch.tensor(traj_anchors, dtype=torch.float32).to(ego_instance_feature.device)
        # traj_anchors_np通过复制增加一维变为[20,bs,80,4]
        traj_anchors = traj_anchors.unsqueeze(1).expand(-1, bs, -1, -1)
        

        # 初始化diffusion_losses= [],里面只有一个元素，是tensor，值为0，形状为[1]
        diffusion_losses = [torch.tensor(0).to(ego_instance_feature.device)]

        # trajectory, probability = self.trajectory_decoder(x[:, 0]) # 主代理的嵌入表示,形状为 (batch_size, embed_dim)
        trajectory, probability, diffusion_losses = self.trajectory_decoder_diffu(ego_instance_feature, map_instance_feature, traj_anchors)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "diffusion_losses": diffusion_losses,
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )
            # print("最终输出的轨迹形状:", out["output_trajectory"].shape)
        return out

    def load_cluster_centers(self,npy_file_path: str):
        """
        加载 K-Means 聚类中心并返回轨迹锚点。
        假设文件格式与 kmeans_plan.py 输出的类似，形状可能是 ( K, 6, 2)。
        """
        cluster_centers = np.load(npy_file_path)
        # print("聚类中心的形状:", cluster_centers.shape)
        # sys.exit(1)
        # 例如，形状是  (200, 32, 2)，表示 200个聚类中心、每个轨迹 32 个点、每个点 2 坐标
        anchor_points = []
        for i in range(cluster_centers.shape[0]):
            # single_center 的形状是 (32, 2)，即一个轨迹锚点
            anchor_points.append(cluster_centers[i])
        return anchor_points


