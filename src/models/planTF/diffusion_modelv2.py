import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .modules.conditional_unet1d import ConditionalUnet1D
# from plantf.src.models.planTF.modules.conditional_unet1d import ConditionalUnet1D
from sklearn.cluster import KMeans
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionModelv2(nn.Module): # 等价于TrajectoryHead ，TrajectoryDecoder
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
        self.prob_decoder_anchor = nn.Sequential(
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
        
        """       
        # 特征处理模块============================================================
        # 把CrossAttentionUnetModel对特征的处理挪出来：

        # 自车可学习参数
        self.ego_feature = nn.Embedding(1, feature_dim)
        self.feature_dim = feature_dim # 128

        #self.map_feature_pos = self.map_bev_pos.weight[None].repeat(bs, 1, 1)
        # 位置编码可学习 MAP / instance 
        # Cross-Attention 
        self.ego_instance_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=self.num_heads, batch_first=True)
        # self.cross_agent_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.map_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=self.num_heads, batch_first=True)
        # 添加多头注意力机制模块，用来学习其他代理预测信息
        self.prediction_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=self.num_heads, batch_first=True)
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

        self.map_feature_pos = nn.Embedding(100, self.feature_dim)
        self.ego_pos_latent = nn.Embedding(1, self.feature_dim)

        # 添加降维模块，将 prediction 的最后维度从 prediction_dim 映射到 feature_dim
        self.prediction_fc = nn.Linear(160, feature_dim)
        #特征处理模块================================================
        """ 

    def forward(self, ego_instance_feature, map_instance_feature, other_instance_feature, traj_anchors,prediction, target):
        # ego [32, 1, 128]         含义：子车代理的嵌入表示(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim] 地图的嵌入表示
        # other_instance_feature [32, 32, 128] [bs, num_agents, embed_dim] 其他代理的嵌入表示
        # traj_anchors [num_modes, bs, future_steps, 4]   -> num_modes=锚点轨迹数：40
        # prediction:  [bs, num_agents, future_steps, 2] 表示其他代理预测模型对其他代理未来状态的预测。
        # 目标；融合 ego_instance_feature, map_instance_feature, other_instance_feature, prediction四个信息，生成全局特征 global_feature
        """
        #==================================================
        map_encoded, _ = self.map_decoder(
            query=ego_instance_feature,# [32, 1, 128]
            key=map_instance_feature,# [32, 222, 128]
            value=map_instance_feature#[32, 222, 128]
        )
        map_encoded = self.ins_cond_layernorm_1(map_encoded)
        map_encoded = self.fc1(map_encoded)
        map_encoded = self.ins_cond_layernorm_2(map_encoded)# [32, 1, 128]

        # 将 4-D tensor 的 prediction 降为 3-D tensor，并进行降维
        bs, num_agents, future_steps, dim = prediction.shape    # [32, 32, 80, 2]
        prediction = prediction.view(bs, num_agents, future_steps * dim)  #[32, 32, 160]
        
        # 通过全连接层降维 prediction，从 160 降到 128
        prediction = self.prediction_fc(prediction)            # [32, 32, 128]

        # 融合 prediction 信息
        prediction_encoded, _ = self.prediction_decoder(
            query=ego_instance_feature,#[32, 1, 128]
            key=prediction,#[32, 32, 128]
            value=prediction#[32, 32, 128]
        )
        prediction_encoded = self.map_cond_layernorm_1(prediction_encoded)
        prediction_encoded = self.fc2(prediction_encoded)
        prediction_encoded = self.map_cond_layernorm_2(prediction_encoded)# [32, 1, 128]

        # 生成 global_feature
        global_feature = ego_instance_feature + map_encoded + prediction_encoded# [32, 1, 128]
        #==================================================
        """
        global_feature = self.fusion_module(
            ego_instance_feature,   # [32,1,128]
            map_instance_feature,   # [32,222,128]
            other_instance_feature,# [32,32,128]
            traj_anchors,           # [32,32,80,4]
            prediction,            # [32,32,80,2]
            target
        )# [32, 1, 128]
        global_feature = global_feature.squeeze(1)# [32, 128]

        # noisy_traj_points = noisy_traj_points.to('cuda')  # 将输入张量移动到 GPU
        global_feature = global_feature.to('cuda')  # 将全局条件张量移动到 GPU
        # timesteps = timesteps.to('cuda')  # 将时间步张量移动到 GPU

        if self.training:
            return self.forward_train(global_feature, traj_anchors, target)
        else:
            return self.forward_test(global_feature, traj_anchors, target)        


    def forward_train(self,global_feature, traj_anchors, target):
        # ego [32, 1, 128]         含义：(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim]
        # traj_anchors [bs, num_modes, future_steps, 4]
        # prediction:  [bs, num_agents, future_steps, 2] 表示模型对其他代理未来状态的预测。
        # target:  [bs,  future_steps, 3] 代表自车未来状态的真实值。x,y,heading
        bs = global_feature.shape[0]  # 32
        device = global_feature.device
        trajectories = []
        diffu_loss = torch.tensor([0], device=device)
        # 预测cls概率
        probability = self.probability_decoder(global_feature.squeeze(1))  # [bs, num_modes]
        anchor_prob = self.prob_decoder_anchor(global_feature.squeeze(1))  # [bs, num_modes]
        # 生成一个随机时间步，和随机噪声
        noise = torch.randn(traj_anchors.shape, device=device)# [bs, num_modes, future_steps, 4]
        timesteps = torch.randint(
            0,50,
            (bs,),
            device=device
        )# [bs]
        noisy_traj = self.diffusion_scheduler.add_noise(
            original_samples=traj_anchors,
            noise=noise,
            timesteps=timesteps
        ).float()# [bs, num_modes, future_steps, 4]
        # 预测干净样本
        #noise_pred_net接受的输入是[bs, future_steps, 4]，所以需要用for循环每次预测一个模态
        for mode in range(self.num_modes):
            traj_pred = self.noise_pred_net(
                        sample=noisy_traj[:, mode],# [bs, future_steps, 4]
                        timestep=timesteps,
                        global_cond=global_feature,# [bs, embed_dim]
            )
            # 把预测的轨迹合起来，最后形状是[bs, num_modes, future_steps, 4]
            trajectories.append(traj_pred)
        traj_pred = torch.stack(trajectories, dim=1)  # (bs, num_modes, future_steps, 4)

        ego_target_pos, ego_target_heading = target[:, :, :2], target[:, :, 2] # ego_target_pos:[bs, ego_num, future_steps, 2]
        target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        bs, num_mode, ts, d = traj_pred.shape
        target_traj = target
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - traj_anchors[...,:2], dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)# [bs]
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(traj_pred, 1, mode_idx).squeeze(1)
        anchor_cls_loss = F.cross_entropy(anchor_prob, cls_target) #[bs, num_modes] [bs] -> [bs]
        anchor_reg_loss = F.smooth_l1_loss(best_reg, target_traj)# shape of anchor_reg_loss: [bs, future_steps, 3]

        return traj_pred, probability, anchor_cls_loss, anchor_reg_loss

    def forward_test(self,global_feature, traj_anchors, target):
        # ego [32, 1, 128]         含义：(bs, 1, embed_dim)
        # map [32, 222, 128]  [bs, num_polygons, embed_dim]
        # traj_anchors [bs, num_modes, future_steps, 4]
        # prediction:  [bs, num_agents, future_steps, 2] 表示模型对其他代理未来状态的预测。
        # target:  [bs,  future_steps, 3] 代表自车未来状态的真实值。x,y,heading
        bs = global_feature.shape[0]
        device = global_feature.device
        trajectories = []
        # 假设 self.diffusion_scheduler 是你的扩散模型调度器
        self.diffusion_scheduler.set_timesteps(num_inference_steps=4)

        # 迭代加num_inference_steps次噪声
        # Iteratively add noise for num_inference_steps times
        img = traj_anchors.clone()
        for t in self.diffusion_scheduler.timesteps:
            noise_step = torch.randn_like(img, device=device)
            img = self.diffusion_scheduler.add_noise(
            original_samples=img,
            noise=noise_step,
            timesteps=torch.full((bs,), t, device=device, dtype=torch.long)
            ).float()

        # noise = torch.randn(traj_anchors.shape, device=device)
        # timesteps = torch.ones((bs), device=device, dtype=torch.long) * 20 # [bs, infer_times]
        # img = self.diffusion_scheduler.add_noise(
        #     original_samples=traj_anchors,
        #     noise=noise,
        #     timesteps=timesteps
        # ).float()
        # 迭代去噪预测
        # for k in [1,5,10]:
        #     timesteps = k * torch.ones((bs,), device=device, dtype=torch.long) # [bs]

        timesteps_2 = self.diffusion_scheduler.timesteps
        for t in self.diffusion_scheduler.timesteps:
            trajectories = []
            for mode in range(0,self.num_modes):
                traj_pred = self.noise_pred_net(
                            sample=img[:, mode],# [bs, future_steps, 4]
                            timestep=t,
                            global_cond=global_feature,# [bs, embed_dim]
                )
                # 把预测的轨迹合起来，最后形状是[bs, num_modes, future_steps, 4]
                trajectories.append(traj_pred)
            traj_pred = torch.stack(trajectories, dim=1)  # (bs, num_modes, future_steps, 4)
            #img：([32, 40, 80, 4]) traj_pred.shape:([32, 40, 80, 4])
            img = self.diffusion_scheduler.step(
                model_output=traj_pred,
                timestep=t,
                sample=img
            ).prev_sample

        # 预测cls概率
        probability = self.probability_decoder(global_feature.squeeze(1))  # [bs, num_modes]

        ego_target_pos, ego_target_heading = target[:, :, :2], target[:, :, 2] # ego_target_pos:[bs, ego_num, future_steps, 2]
        target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        bs, num_mode, ts, d = traj_pred.shape
        target_traj = target
        # print(f"traj_anchors shape: {traj_anchors.shape}")#[80,80,2]
        # print(f"target_traj shape: {target_traj.shape}")
# traj_anchors shape: torch.Size([1, 80, 80, 4])
# target_traj shape: torch.Size([1, 0, 4])        
#RuntimeError: The size of tensor a (0) must match the size of tensor b (80) at non-singleton dimension 2
        # sys.exit(1)
        dim = target_traj.shape[1]
        # print("dim:",dim)
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - traj_anchors[...,:dim,:2], dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)# [bs]
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(traj_pred, 1, mode_idx).squeeze(1)
        anchor_cls_loss = F.cross_entropy(probability, cls_target) #[bs, num_modes] [bs] -> [bs]
        # print("best_reg",best_reg.shape)# [bs, 80, 4],
        #best_reg第1个维度取前dim项
        best_reg = best_reg[:,:dim,:]
        # print("best_reg",best_reg.shape)# [bs, 80, 4]
        # print("target_traj",target_traj.shape)# [bs, 80, 4]
        anchor_reg_loss = F.l1_loss(best_reg, target_traj)# shape of anchor_reg_loss: [bs, future_steps, 3]

        return traj_pred, probability, anchor_cls_loss, anchor_reg_loss
                
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
                traj_anchors,
                prediction,            # [32,32,80,2]
                target):
        
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