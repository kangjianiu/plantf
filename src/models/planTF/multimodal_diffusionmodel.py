import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from .modules.diffusion_model import CrossAttentionUnetModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
"""
通过引入 MultiModalDiffusionModel 类，在 PlanningModel 类中使用多模态机制来预测多条轨迹。
MultiModalDiffusionModel 类使用 CrossAttentionUnetModel 和 DDPMScheduler 来实现扩散模型，
并通过多次运行扩散过程生成多条轨迹。通过这种方式，模型可以同时输出多条预测的轨迹

修改 MultiModalDiffusionModel 类,将 MultiModalDiffusionModel 类修改为同时
输出 trajectory 和 probability,并确保它们之间的逻辑关系
"""
class MultiModalDiffusionModel(nn.Module):
    def __init__(self, feature_dim, num_modes, future_steps):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps # 80
        self.trajectory_decoder = CrossAttentionUnetModel(feature_dim)
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
        )

    def forward(self, ego_instance_feature, map_instance_feature, traj_anchors):
        trajectories = []
        probabilities = []

            # 截断去噪过程，从噪声轨迹开始，对其进行2步去噪
        for i in range(self.num_modes):

            print("multimodel接受的traj_anchors的形状:", traj_anchors.shape) #[256, 32, 2]
            noise = self.pyramid_noise_like(traj_anchors) 
            
            diffusion_output = noise
            print("multimodel的初始噪声(diffusion_output)形状:", diffusion_output.shape) #[256, 32, 2]

            # # 初始噪声维度扩充为[256, 32, 20]：
            # #   首先，确保 diffusion_output 的形状为 [256, 2, 32]
            # diffusion_output = diffusion_output.permute(0, 2, 1)  # [256, 2, 32]
            # #   生成额外的18个通道的随机噪声
            # additional_noise = torch.randn(diffusion_output.size(0), 18, diffusion_output.size(2)).to(diffusion_output.device)  # [256, 18, 32]
            # #   拼接随机噪声到初始噪声，得到新的 diffusion_output
            # diffusion_output = torch.cat((diffusion_output, additional_noise), dim=1)  # [256, 20, 32]
            # diffusion_output = diffusion_output.permute(0, 2, 1)  # [256, 32, 20]
            # print("扩展后的diffusion_output形状:", diffusion_output.shape)  # 应输出 [256, 32, 20]


            for k in self.scheduler.timesteps[:2]:
                print("multi中k.shape:", k.shape)
                diffusion_output = self.scheduler.scale_model_input(diffusion_output)
                noise_pred = self.trajectory_decoder(ego_instance_feature, map_instance_feature, k.unsqueeze(-1).repeat(ego_instance_feature.shape[0]).to(ego_instance_feature.device), diffusion_output)
                diffusion_output = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=diffusion_output
                ).prev_sample
                """step解释
                假设你正在执行逆扩散过程中的一个步骤：
                输入:
                sample: 当前时间步 $t$ 的样本 $x_t$，形状 [batch_size, channels, height]，例如 [256, 20, 32]。
                model_output: 模型对 $x_t$ 的预测输出，通常是噪声 $\epsilon_\theta(x_t, t)$，形状与 sample 相同。
                timestep: 当前时间步 $t$ 的索引，例如 10。
                过程:
                计算相关的扩散系数 $\alpha_t$ 和 $\beta_t$。
                根据 prediction_type,从 model_output 预测出原始样本 $x_0$ 或调整后的样本。
                使用公式计算预测的前一个时间步的样本 $x_{t-1}$。
                添加噪声（如果 $t > 0$）。
                输出:
                prev_sample: 预测的前一个时间步的样本 $x_{t-1}$，用于下一个步骤的输入。
                pred_original_sample: 预测的原始样本 $x_0$（根据 prediction_type)。
                """

            trajectory = self.denormalize_xy_rotation(diffusion_output, N=self.future_steps, times=10)
            trajectories.append(trajectory)

        # 计算概率
        probability = self.probability_decoder(ego_instance_feature.squeeze(1))
        probabilities.append(probability)

        trajectories = torch.stack(trajectories, dim=1)  # 形状为 (batch_size, num_modes, future_steps, 2)
        probabilities = torch.stack(probabilities, dim=1)  # 形状为 (batch_size, num_modes)
        return trajectories, probabilities

    def normalize_xy_rotation(self, trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :]
        x_scale = 10
        y_scale = 75
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale

        rotated_trajectories = []
        for i in range(times):
            theta = 2 * torch.pi * i / 10
            rotation_matrix, _ = self.get_rotation_matrices(theta)
                # 打印 rotation_matrix 的类型和内容以进行调试
            # print("rotation_matrix type:", type(rotation_matrix),rotation_matrix.shape)
            # print("downsample_trajectory type:", type(downsample_trajectory),downsample_trajectory.shape)

            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            rotated_trajectory = self.apply_rotation(downsample_trajectory, rotation_matrix)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0, 2, 1)
        return trajectory

    def denormalize_xy_rotation(self, trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape
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


    def generate_anchor_trajectories(history_trajectories, num_modes):
        """
        使用K-Means聚类生成锚点轨迹。

        :param history_trajectories: 历史轨迹数据，形状为 (num_samples, trajectory_length, 2)
        :param num_modes: 锚点轨迹的数量
        :return: 锚点轨迹，形状为 (num_modes, trajectory_length, 2)
        """
        # 将轨迹数据展平为二维数组，形状为 (num_samples, trajectory_length * 2)
        flattened_trajectories = history_trajectories.reshape(history_trajectories.shape[0], -1)

        # 使用K-Means聚类生成锚点轨迹
        kmeans = KMeans(n_clusters=num_modes, random_state=0).fit(flattened_trajectories)
        anchor_trajectories = kmeans.cluster_centers_.reshape(num_modes, history_trajectories.shape[1], 2)

        return anchor_trajectories

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

    # # 示例用法
    # history_trajectories = np.random.rand(1000, 30, 2)  # 假设有1000个历史轨迹，每个轨迹有30个时间步，每个时间步有2个坐标
    # num_modes = 5  # 生成5个锚点轨迹
    # noise_std = 0.1  # 高斯噪声的标准差
    # anchor_trajectories = generate_anchor_trajectories(history_trajectories, num_modes)
    # noisy_trajectories = add_gaussian_noise(anchor_trajectories, noise_std)
