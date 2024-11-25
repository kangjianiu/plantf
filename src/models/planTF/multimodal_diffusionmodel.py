import torch
import torch.nn as nn
from planTF.src.models.planTF.modules.diffusion_model import CrossAttentionUnetModel
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
        self.future_steps = future_steps
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

    def forward(self, instance_feature, map_instance_feature, trajs):
        trajectories = []
        probabilities = []

        for mode in range(self.num_modes):
            trajs_temp = self.normalize_xy_rotation(trajs.squeeze(0))
            noise = self.pyramid_noise_like(trajs_temp)
            diffusion_output = noise

            for k in self.scheduler.timesteps[:]:
                diffusion_output = self.scheduler.scale_model_input(diffusion_output)
                noise_pred = self.trajectory_decoder(instance_feature, map_instance_feature, k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(instance_feature.device), diffusion_output)
                diffusion_output = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=diffusion_output
                ).prev_sample

            trajectory = self.denormalize_xy_rotation(diffusion_output, N=self.future_steps, times=10)
            trajectories.append(trajectory)

        # 计算概率
        probability = self.probability_decoder(instance_feature.squeeze(1))
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

        return rotation_matrix, inverse_rotation_matrix

    def apply_rotation(self, trajectory, rotation_matrix):
        rotated_trajectory = torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)
        return rotated_trajectory