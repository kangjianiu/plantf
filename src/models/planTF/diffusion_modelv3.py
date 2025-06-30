import sys
import torch
import logging
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .modules.conditional_unet1d import ConditionalUnet1D
# from plantf.src.models.planTF.modules.conditional_unet1d import ConditionalUnet1D
from sklearn.cluster import KMeans
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionModelv3(nn.Module): # 等价于TrajectoryHead ，TrajectoryDecoder
    def __init__(self, feature_dim, num_modes, future_steps):
        super().__init__()
        self.num_modes = num_modes
        self.num_heads = 8
        self.future_steps = future_steps # 80
        self.feature_dim = feature_dim 
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        hidden = 2 * feature_dim
        self.probability_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_modes),
            # # 加一个softmax层
            # nn.Softmax(dim=1)
        )       
        self.fusion_module = FeatureFusion(
            embed_dim=128,
            num_attention_heads=8,
            future_steps=80
        )
        self.noise_pred_net = ConditionalUnet1D(
                input_dim=4,
                global_cond_dim= 128, # 256 修改成 128，取决于-> global_cond=global_feature,#本质上取决于ego,[32, 128]
                down_dims=[128, 256],
                cond_predict_scale=False,
        )
        logger = logging.getLogger(__name__)
        logger.info(f"nkj diffusion modelv3 num_modes:{num_modes}")    
        
    def forward(self, ego_instance_feature, map_instance_feature, other_instance_feature, traj_anchors,prediction):
        # ego [32, 1, 128]         含义：子车代理的嵌入表示(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim] 地图的嵌入表示
        # other_instance_feature [32, 32, 128] [bs, num_agents, embed_dim] 其他代理的嵌入表示
        # traj_anchors [num_modes, bs, future_steps, 4]   -> num_modes=锚点轨迹数：40
        # prediction:  [bs, num_agents, future_steps, 2] 表示其他代理预测模型对其他代理未来状态的预测。
        # 目标；融合 ego_instance_feature, map_instance_feature, other_instance_feature, prediction四个信息，生成全局特征 global_feature
        global_feature = self.fusion_module(
            ego_instance_feature,   # [32,1,128]
            map_instance_feature,   # [32,222,128]
            other_instance_feature,# [32,32,128]
            prediction            # [32,32,80,2]
        )# [32, 1, 128]
        global_feature = global_feature.squeeze(1)# [32, 128]
        # noisy_traj_points = noisy_traj_points.to('cuda')  # 将输入张量移动到 GPU
        global_feature = global_feature.to('cuda')  # 将全局条件张量移动到 GPU
        # timesteps = timesteps.to('cuda')  # 将时间步张量移动到 GPU
        if self.training:
            return self.forward_train(global_feature, traj_anchors)
        else:
            return self.forward_test(global_feature, traj_anchors)        

    def forward_train(self,global_feature, traj_anchors):
        # ego [32, 1, 128]         含义：(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim]
        # traj_anchors [bs, num_modes, future_steps, 4]
        # prediction:  [bs, num_agents, future_steps, 2] 表示模型对其他代理未来状态的预测。
        bs = global_feature.shape[0]
        device = global_feature.device
        # 假设 self.diffusion_scheduler 是你的扩散模型调度器
        self.diffusion_scheduler.set_timesteps(num_inference_steps=4)

        img = traj_anchors.clone()
        noise = torch.randn_like(img, device=device)  # [bs, num_modes, future_steps, 4]
        # trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        timesteps = torch.randint(0,50, (bs,), device=device, dtype=torch.long) # 随机生成timesteps
        img = self.diffusion_scheduler.add_noise(
            original_samples=img,
            noise=noise,
            timesteps=timesteps
        ).float()# [bs, num_modes, future_steps, 4]

        roll_timesteps = (np.arange(0, 2) * 10).round()[::-1].copy().astype(np.int64)#roll_timesteps:[0,1]
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device) # 值：[10,0]
        for t in roll_timesteps[:]:
            trajectory = []
            for mode in range(0,self.num_modes):
                traj_pred = self.noise_pred_net(
                            sample=img[:, mode],# [bs, future_steps, 4]
                            timestep=t,
                            global_cond=global_feature,# [bs, embed_dim]
                )
                # 把预测的轨迹合起来，最后形状是[bs, num_modes, future_steps, 4]
                trajectory.append(traj_pred)
            traj_pred = torch.stack(trajectory, dim=1)  # (bs, num_modes, future_steps, 4)
            #img：([32, 40, 80, 4]) traj_pred.shape:([32, 40, 80, 4])
            img = self.diffusion_scheduler.step(
                model_output=traj_pred,
                timestep=t,
                sample=img
            ).prev_sample

        # 预测cls概率
        probability = self.probability_decoder(global_feature.squeeze(1))  # [bs, num_modes]
        return traj_pred, probability

    def forward_test(self,global_feature, traj_anchors):
        # ego [32, 1, 128]         含义：(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim]
        # traj_anchors [bs, num_modes, future_steps, 4]
        # prediction:  [bs, num_agents, future_steps, 2] 表示模型对其他代理未来状态的预测。
        # target:  [bs,  future_steps, 3] 代表自车未来状态的真实值。x,y,heading
        bs = global_feature.shape[0]
        device = global_feature.device
        # 假设 self.diffusion_scheduler 是你的扩散模型调度器
        self.diffusion_scheduler.set_timesteps(num_inference_steps=4)

        img = traj_anchors.clone()
        noise = torch.randn_like(img, device=device)  # [bs, num_modes, future_steps, 4]
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(
            original_samples=img,
            noise=noise,
            timesteps=trunc_timesteps
        ).float()# [bs, num_modes, future_steps, 4]

        roll_timesteps = (np.arange(0, 2) * 10).round()[::-1].copy().astype(np.int64)#roll_timesteps:[0,1]
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device) # 值：[10,0]
        for t in roll_timesteps[:]:
            trajectory = []
            for mode in range(0,self.num_modes):
                traj_pred = self.noise_pred_net(
                            sample=img[:, mode],# [bs, future_steps, 4]
                            timestep=t,
                            global_cond=global_feature,# [bs, embed_dim]
                )
                # 把预测的轨迹合起来，最后形状是[bs, num_modes, future_steps, 4]
                trajectory.append(traj_pred)
            traj_pred = torch.stack(trajectory, dim=1)  # (bs, num_modes, future_steps, 4)
            #img：([32, 40, 80, 4]) traj_pred.shape:([32, 40, 80, 4])
            img = self.diffusion_scheduler.step(
                model_output=traj_pred,
                timestep=t,
                sample=img
            ).prev_sample

        # 预测cls概率
        probability = self.probability_decoder(global_feature.squeeze(1))  # [bs, num_modes]
        return traj_pred, probability
                
    def normalize_xy_rotation(self, trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape
        vx_vy = trajectory[..., 2:]
        trajectory = trajectory[..., :2]
        downsample_trajectory = trajectory[:, :N, :]
        x_scale = 10
        y_scale = 75
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * torch.pi * i / 10
            rotation_matrix, _ = self.get_rotation_matrices(theta)

            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            rotated_trajectory = self.apply_rotation(downsample_trajectory, rotation_matrix)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0, 2, 1)
        # cat trajectory 和 vx_vy
        trajectory = torch.cat([trajectory, vx_vy], dim=-1)

        return trajectory
    
    def denormalize_xy_rotation(self, trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape# [32, 80, 4]
        vx_vy = trajectory[..., 2:]
        xy = trajectory[..., :2]
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * torch.pi * i / 10
            rotation_matrix, inverse_rotation_matrix = self.get_rotation_matrices(theta)
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
            inverse_rotated_trajectory = self.apply_rotation(trajectory[:, :, 2*i:2*i+2], inverse_rotation_matrix)
            inverse_rotated_trajectories.append(inverse_rotated_trajectory)

        final_trajectory = torch.cat(inverse_rotated_trajectories, 1).permute(0, 2, 1)
        final_trajectory = final_trajectory[:, :, :2]
        final_trajectory[:, :, 0] *= 13
        final_trajectory[:, :, 1] *= 55
        final_trajectory = torch.cat([final_trajectory, vx_vy], dim=-1)# [32, 80, 4]
        
        return final_trajectory

    def pyramid_noise_like(self, trajectory, discount=0.9):
        """
        噪声张量 (noise)，形状与 trajectory 相同 (b, n, c)，并且经过归一化(标准差归一到1)后转为 float 类型。
        会生成一种“多尺度叠加”的噪声，类似在原始尺度和缩放到更小尺寸的尺度上分别加噪，
        而每一次在小尺寸上生成的噪声再被线性插值到原始序列长度并带上一定的衰减系数。
        最终合并后的噪声在多层级时间分辨率上拥有随机扰动，也就是所谓的“金字塔”或“多分辨率”噪声。
        """
        b, n, c = trajectory.shape
        trajectory_reshape = trajectory.permute(0, 2, 1)
        up_sample = torch.nn.Upsample(size=(n), mode='linear')
        noise = torch.randn_like(trajectory_reshape)
        for i in range(10):
            r = torch.rand(1, device=trajectory.device) + 1
            n = max(1, int(n/(r**i)))
            noise += up_sample(torch.randn(b, c, n).to(trajectory_reshape)) * discount**i
            if n == 1: break
        noise = noise.permute(0, 2, 1)
        return (noise/noise.std()).float()

    def get_rotation_matrices(self, theta):
        theta_tensor = torch.tensor(theta)
        cos_theta = torch.cos(theta_tensor)
        sin_theta = torch.sin(theta_tensor)

        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        inverse_rotation_matrix = torch.tensor([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        # 确保 rotation_matrix 是一个张量
        if not isinstance(rotation_matrix, torch.Tensor):
            rotation_matrix = torch.tensor(rotation_matrix)
        
        return rotation_matrix, inverse_rotation_matrix

    def apply_rotation(self, trajectory, rotation_matrix):
        rotated_trajectory = torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)
        return rotated_trajectory

    def add_gaussian_noise(trajectories, noise_std):
        """
        向轨迹添加高斯噪声。

        :param trajectories: 轨迹数据，形状为 (num_modes, trajectory_length, 2)
        :param noise_std: 高斯噪声的标准差
        :return: 添加噪声后的轨迹，形状为 (num_modes, trajectory_length, 2)
        """
        noise = np.random.normal(0, noise_std, trajectories.shape)
        noisy_trajectories = trajectories + noise

        return noisy_trajectories


class FeatureFusion(nn.Module):
    def __init__(self, 
                 embed_dim=128,
                 num_attention_heads=8,
                 future_steps=80):  # 根据实际情况设置默认参数
        super().__init__()
        
        # 轨迹预测编码器（假设future_steps固定）
        self.prediction_encoder = nn.Linear(future_steps*2, embed_dim)
        
        # 特征空间对齐投影
        self.map_proj = nn.Linear(embed_dim, embed_dim)
        self.other_proj = nn.Linear(embed_dim, embed_dim)
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        
        # 交叉注意力模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
    def forward(self, 
                ego_instance_feature,   # [32,1,128]
                map_instance_feature,   # [32,222,128]
                other_instance_feature,# [32,32,128]
                prediction            # [32,32,80,2]
                ):
        
        ###############################
        # 1. 轨迹预测特征编码
        ###############################
        bs, num_agents = prediction.shape[:2]
        # 展平轨迹坐标 [32,32,80*2=160]
        pred_flat = prediction.view(bs, num_agents, -1)
        # 编码为特征向量 [32,32,128]
        pred_feature = self.prediction_encoder(pred_flat)
        
        ###############################
        # 2. 特征空间投影对齐
        ###############################
        map_proj = self.map_proj(map_instance_feature)    # [32,222,128]
        other_proj = self.other_proj(other_instance_feature) # [32,32,128]
        pred_proj = self.pred_proj(pred_feature)          # [32,32,128]
        
        ###############################
        # 3. 构建全局上下文
        ###############################
        # 拼接环境特征 [32,222+32+32=286,128]
        global_context = torch.cat([map_proj, other_proj, pred_proj], dim=1)
        
        ###############################
        # 4. 交叉注意力融合
        ###############################
        # 以自车特征为Query，全局上下文为Key/Value
        global_feature, _ = self.cross_attn(
            query=ego_instance_feature,  # [32,1,128]
            key=global_context,          # [32,286,128]
            value=global_context,
            need_weights=False
        )  # 输出形状 [32,1,128]
        
        return global_feature  # 后续处理可以继续使用该全局特征