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
        # traj_anchors [32,6,80,4]

        #TODO :在之前没有考虑bathsize的问题,现在需要重新考虑一下轨迹锚点和batchsize的关系（2024.12.27）

        # 截断去噪过程，从噪声轨迹开始，对其进行2步去噪
        # print("diffusion_model接受的traj_anchors的形状:", traj_anchors.shape) #[clusters, future_steps, 4]
        # 随机取num_modes * batch_size 个轨迹锚点,形状组织成为 (num_modes,batch_size, future_steps, 4)
        traj_anchors = traj_anchors[torch.randint(traj_anchors.shape[0], (self.num_modes * ego_instance_feature.shape[0],))]# [num_modes * batch_size, future_steps, 4]
        traj_anchors = traj_anchors.view(self.num_modes, ego_instance_feature.shape[0], self.future_steps, 4)# [num_modes, batch_size, future_steps, 4]
        trajectories = []
        diffusion_losses = []
        for mode in range(self.num_modes):
            # 取出traj_anchors里面第 mode 组轨迹锚点
            # print(f"\n========================第{mode+1}次========================")
            trajs = self.normalize_xy_rotation(traj_anchors[mode], N=self.future_steps, times=1) 
            # print(f"normalize_xy_rotation后的trajs形状:{trajs.shape},trajs值[0]:{trajs[0][0]}") # [batch_size, future_steps, 4]
            noise = self.pyramid_noise_like(trajs)                     # [batch_size, future_steps, 4]
            # print(f"noise形状:{noise.shape},noise值[0]:{noise[0][0]}") # [batch_size, future_steps, 4]
            # 在self.scheduler.timesteps里随机取ego_instance_feature.shape[0]个时间步
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (ego_instance_feature.shape[0],), device=traj_anchors.device)
            noisy_traj_points = self.scheduler.add_noise(
                    original_samples=traj_anchors[mode],
                    noise=noise,
                    timesteps=t,
            ).float()
            # print(f"noisy_traj_points形状:{noisy_traj_points.shape},[0]:{noisy_traj_points[0][0]}") # [batch_size, future_steps, 4]
            traj_pred = noisy_traj_points            
            diffusion_output = traj_pred #[batch_size, future_steps, 4]

            for k in self.scheduler.timesteps[:2]:

                diffusion_output = self.scheduler.scale_model_input(diffusion_output)
                noise_pred = self.trajectory_decoder(ego_instance_feature, map_instance_feature, 
                            k.unsqueeze(-1).repeat(ego_instance_feature.shape[0]).to(ego_instance_feature.device), diffusion_output)
                # print(f"diffusion_model中第{mode+1}次,第  {k}  步的 预测噪声 形状:{noise_pred.shape}",)# [32，80, 4](batch_size, future_steps, 4)
                traj_pred = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=traj_pred
                ).prev_sample
                diffusion_output = traj_pred
                # print(f"diffusion_model中第 {mode+1}次,第 {k}  步的 推理轨迹 形状:{diffusion_output.shape}\n",)# [32，80, 4](batch_size, future_steps, 4)
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

            # 用 traj_pred与noisy_traj_points计算扩散模型的损失
            loss_fn = nn.MSELoss()
            diffusion_loss = loss_fn(traj_pred, noisy_traj_points)

            # print(f"diffusion_model最终输出的轨迹形状:{diffusion_output.shape},轨迹值[0]:{diffusion_output[0][0]}")#[32，80, 4](batch_size, future_steps, 4)
            trajectory = self.denormalize_xy_rotation(diffusion_output, N=self.future_steps, times=1)#[32，80, 4][batch_size, future_steps, 4]
            # print(f"denormalize_xy转换后的轨迹形状:{trajectory.shape}，轨迹值[0]:{trajectory[0][0]}")#[batch_size, future_steps, 4]
            trajectories.append(trajectory)# [num_modes, batch_size, future_steps, 4]
            diffusion_losses.append(diffusion_loss)

        # print(f"trajectories类型形状长度:{type(trajectories)},{trajectories[0].shape},{len(trajectories)}")# [num_modes, batch_size, future_steps, 4]
        trajectories = torch.stack(trajectories, dim=1)  # 形状为 (batch_size, num_modes, future_steps, 4)
        # print(f"trajectories类型形状长度:{type(trajectories)},{trajectories[0].shape},{len(trajectories)}")
        # sys.exit(1)
        # 计算概率
        probability = self.probability_decoder(ego_instance_feature.squeeze(1))
        # print(f"概率形状:{probability.shape}，概率值:{probability[0]}")#[batch_size, num_modes]
        

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
        batch_size = ego_instance_feature.shape[0]# 32
        ego_latent = ego_instance_feature[:,:,:] # torch.Size([32, 128])
        
        
        # # 原来的
        # map_pos = self.map_feature_pos.weight[None].repeat(batch_size, 1, 1) #shape: [32, 100, 128] ??
        # ego_pos = self.ego_pos_latent.weight[None].repeat(batch_size, 1, 1)#shape: [32, 1, 128]
        # ego_latent = self.ego_feature.weight[None].repeat(batch_size, 1, 1)#shape: [32, 1, 128]

        # # ego_instance_decoder 把实例特征作为查询、键和值进行处理。然后，经过层归一化和全连接层处理        
        # ego_latent = self.ego_instance_decoder(
        #     query = ego_latent,
        #     key = ego_instance_feature,
        #     value = ego_instance_feature,
        # )[0]
        # ego_latent = self.ins_cond_layernorm_1(ego_latent)
        # ego_latent = self.fc1(ego_latent)
        # ego_latent = self.ins_cond_layernorm_2(ego_latent)
        # ego_latent = ego_latent.unsqueeze(1)
        # # print(instance_feature.shape)
        # # print(ego_latent.shape)

        # # map_decoder 将地图特征作为查询、键和值进行处理。然后，经过层归一化和全连接层处理。
        # map_instance_feature = self.map_decoder(
        #     query = map_instance_feature + map_pos,
        #     key = map_instance_feature + map_pos,
        #     value = map_instance_feature,
        # )[0]
        # map_instance_feature = self.fc1(map_instance_feature)
        # map_instance_feature = self.ins_cond_layernorm_1(map_instance_feature)

        # # ego_map_decoder 将实例特征和地图特征进行交互处理。然后，经过层归一化和全连接层处理。
        # # 将处理后的实例特征作为全局特征，并移除多余的维度
        # ego_latent = self.ego_map_decoder(
        #     query = ego_latent + ego_pos,
        #     key = map_instance_feature,
        #     value = map_instance_feature,
        # )[0]
        # ego_latent = self.map_cond_layernorm_1(ego_latent)
        # ego_latent = self.fc2(ego_latent)
        # ego_latent = self.map_cond_layernorm_2(ego_latent)
        
        # # TODO：融合ego和map信息


        global_feature = ego_latent # 目前只用了ego的信息
        global_feature = global_feature.squeeze(1)# 本质上取决于ego_instance_feature：主代理的嵌入表示

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