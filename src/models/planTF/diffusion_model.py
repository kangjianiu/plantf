import sys
import torch
import torch.nn as nn
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .modules.conditional_unet1d import ConditionalUnet1D
from sklearn.cluster import KMeans
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionModel(nn.Module):
    def __init__(self, feature_dim, num_modes, future_steps):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps # 80
        self.trajectory_decoder = CrossAttentionUnetModel(feature_dim)#feature_dim = 128
        self.scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            trained_betas=None,
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="leading",
            steps_offset=0,
            rescale_betas_zero_snr=False,
        )
        hidden = 2 * feature_dim
        self.probability_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_modes),
            # 加一个softmax层
            nn.Softmax(dim=1)
        )

    def forward(self, ego_instance_feature, map_instance_feature, traj_anchors):
        # ego [32, 1, 128]         含义：(batch_size, 1, embed_dim)
        # map [32, 222, 128] 
        # traj_anchors [256,80,4]
        # 从traj_anchors里面随机采样出来形状为[self.num_modes, bs, self.future_steps, 4]的traj_anchors，由于不够，可以重复使用
        
        bs = ego_instance_feature.shape[0]  # 32
        if bs > 32:
            traj_anchors =  torch.cat([traj_anchors,traj_anchors], dim=0)
        traj_anchors = traj_anchors[:self.num_modes * bs]  # [num_modes * batch_size, future_steps, 4]
        traj_anchors = traj_anchors.view(self.num_modes, bs, self.future_steps, 4)  # [num_modes, batch_size, future_steps, 4]

        trajectories = []
        diffusion_losses = []
        infer_times = 2 #去噪推理次数
        for mode in range(self.num_modes):
            trajs = self.normalize_xy_rotation(traj_anchors[mode], N=self.future_steps, times=1)  # [batch_size, future_steps, 4]
            noise = self.pyramid_noise_like(trajs)  # [batch_size, future_steps, 4]

            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bs, infer_times),
                device=traj_anchors.device
            )  # [batch_size, infer_times]

            # 初始化 traj_pred 
            traj_pred = traj_anchors[mode].clone()

            diffusion_loss = 0
            for k in range(infer_times):
                # 添加噪声
                traj_pred = self.scheduler.add_noise(
                    original_samples=traj_pred,
                    noise=noise,
                    timesteps=timesteps[:, k]
                ).float()
                # 预测噪声
                noise_pred = self.trajectory_decoder(ego_instance_feature, map_instance_feature, timesteps[:, k], traj_pred)
                
                # 由噪声计算轨迹
                # 调用 scheduler.step 时逐样本处理，由于 DDPMScheduler.step 不支持批量时间步，这里通过循环逐样本处理每个时间步：
                traj_pred_step = []
                for b in range(bs):
                    traj_pred_single = traj_pred[b].unsqueeze(0)  # [1, future_steps, 4]
                    noise_pred_single = noise_pred[b].unsqueeze(0)  # [1, future_steps, 4]
                    timestep_single = timesteps[b, k].item()  # 标量

                    step_output = self.scheduler.step(
                        model_output=noise_pred_single,
                        timestep=timestep_single,
                        sample=traj_pred_single
                    )
                    traj_pred_step.append(step_output.prev_sample.squeeze(0))  # [future_steps, 4]
                traj_pred = torch.stack(traj_pred_step, dim=0)  # [batch_size, future_steps, 4]
                
                # 计算损失
                loss_fn = nn.MSELoss()
                diffusion_loss = diffusion_loss + loss_fn(noise_pred, noise)

            # 反归一化轨迹
            trajectory = self.denormalize_xy_rotation(traj_pred, N=self.future_steps, times=1)  # [batch_size, future_steps, 4]
            trajectories.append(trajectory)  # [num_modes, batch_size, future_steps, 4]
            diffusion_losses.append(diffusion_loss)

        trajectories = torch.stack(trajectories, dim=1)  # (batch_size, num_modes, future_steps, 4)
        probability = self.probability_decoder(ego_instance_feature.squeeze(1))  # [batch_size, num_modes]

        return trajectories, probability, diffusion_losses


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

class CrossAttentionUnetModel(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        # 自车可学习参数
        self.ego_feature = nn.Embedding(1, feature_dim)
        self.feature_dim = feature_dim # 128

        #self.map_feature_pos = self.map_bev_pos.weight[None].repeat(batch_size, 1, 1)
        # 位置编码可学习 MAP / instance 
        # Cross-Attention 
        self.ego_instance_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.ego_map_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.map_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        #todo:
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Linear(2 * feature_dim, feature_dim)  
        )

        self.fc2 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Linear(2* feature_dim, feature_dim) 
        )

        self.ins_cond_layernorm_1 = nn.LayerNorm(self.feature_dim)
        self.ins_cond_layernorm_2 = nn.LayerNorm(self.feature_dim)

        self.map_cond_layernorm_1 = nn.LayerNorm(self.feature_dim)
        self.map_cond_layernorm_2 = nn.LayerNorm(self.feature_dim)

        self.noise_pred_net = ConditionalUnet1D(
                input_dim=4,
                global_cond_dim= 128, # 256 修改成 128，取决于-> global_cond=global_feature,#本质上取决于ego,[32, 128]
                down_dims=[128, 256],
                cond_predict_scale=False,
        )
        self.map_feature_pos = nn.Embedding(100, self.feature_dim)
        self.ego_pos_latent = nn.Embedding(1, self.feature_dim)

    """
    本来是这样
        def forward(self, instance_feature,timesteps,noisy_traj_points):
            ego_latent = instance_feature[:,900:,:]

            global_feature = ego_latent
            global_feature = global_feature.squeeze(1)

            noise_pred = self.noise_pred_net(
                        sample=noisy_traj_points,
                        timestep=timesteps,
                        global_cond=global_feature,
            )
            return noise_pred
    """       

    def forward(self, ego_instance_feature, map_instance_feature,timesteps,noisy_traj_points):
        # ego_instance_feature[batch_size, 1, 128], map_instance_feature[batch_size, 222, 128], timesteps[batch_size], noisy_traj_points[batch_size, future_steps, 2]
        # batch_size = ego_instance_feature.shape[0]# 32
        ego_latent = ego_instance_feature[:,:,:] # torch.Size([32, 128])

        global_feature = ego_latent.squeeze(1) # 目前只用了ego的信息
        # global_feature = global_feature.squeeze(1)# 本质上取决于ego_instance_feature：主代理的嵌入表示

        noisy_traj_points = noisy_traj_points.to('cuda')  # 将输入张量移动到 GPU
        global_feature = global_feature.to('cuda')  # 将全局条件张量移动到 GPU
        timesteps = timesteps.to('cuda')  # 将时间步张量移动到 GPU

        # print(f"送入unet1d的global_feature形状:{global_feature.shape},噪声轨迹现状：{noisy_traj_points.shape},")#
        # sys.exit(1)
        noise_pred = self.noise_pred_net(
                    sample=noisy_traj_points,# [batch_size, future_steps, 4]
                    timestep=timesteps,
                    global_cond=global_feature,#本质上取决于ego,[batch_size, embed_dim]
        )
        return noise_pred
    





    # if __name__ == "__main__":
    # map_feature = torch.randn(2, 100, 256)
    # instance_feature = torch.randn(2,901,256)
    # global_cond = instance_feature[:,900:,]
    # anchor_size = 32
    # repeated_tensor=global_cond.repeat(1,anchor_size,1)

    # # print(repeated_tensor.shape)
    # expanded_tensor=repeated_tensor.view(-1,256)
    # # print(expanded_tensor.shape)
    # model = CrossAttentionUnetModel(256)
    # noisy_trajs = torch.randn(anchor_size * 2,6,20)

    # output = model.noise_pred_net(sample=noisy_trajs, 
    #                     timestep=torch.tensor([0]),
    #                     global_cond=expanded_tensor)
    # # print(output.shape)
    # global_feature = model(instance_feature, map_feature)
    # # print(global_feature.shape)