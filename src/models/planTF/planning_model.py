import torch
import torch.nn as nn
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
from multimodal_diffusionmodel import MultiModalDiffusionModel
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
        num_modes=6,
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
        self.trajectory_decoder_diffu = MultiModalDiffusionModel(feature_dim=dim, num_modes=num_modes, future_steps=future_steps)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.apply(self._init_weights)

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

    def forward(self, data
                ):
        
        """
        lightning_trainer引用方法:
        def forward(self, features: FeaturesType) -> TargetsType
        res = self.forward(features["feature"].data)
        """
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

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
        x = self.norm(x) # (batch_size, num_agents + num_polygons, embed_dim)

        # trajectory, probability = self.trajectory_decoder(x[:, 0]) # 主代理的嵌入表示,形状为 (batch_size, embed_dim)
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        # probability 表示模型预测的轨迹模式的概率，prediction 表示模型对 其他 代理未来状态的预测。
        # probability 的形状通常为 (batch_size, num_modes)，
        # prediction  的形状通常为 (batch_size, num_agents, future_steps, 2)。

        instance_feature = x[:, 0].unsqueeze(1)  # 主代理的嵌入表示,形状为 (batch_size, 1, embed_dim)
        map_instance_feature = x[:, A:]  # 地图特征,形状为 (batch_size, num_agents - A, embed_dim)
        # 检查 data 中是否包含 "trajs" 这一项
        if "trajs" in data:
            trajs = data["trajs"].to(instance_feature.device)  # 历史轨迹数据
        else:
            raise ValueError("data 中不包含 'trajs' 这一项")
        trajectory, probability = self.trajectory_decoder_diffu(instance_feature, map_instance_feature, trajs)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out
# 我想用diffusion_model.py和diffusion_planning_test共同构成的扩散模型架构替换掉或者说重新设计
# class PlanningModel里面的trajectory_decode, 同时尽量保持trajectory_decoder的输入数据不变,
# 只输出trajectory,不需要输出probability,已知(x[:, 0]) 代表主代理的嵌入表示,形状为 (batch_size, embed_dim)