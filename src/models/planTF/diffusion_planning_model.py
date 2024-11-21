import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from diffusion_model import CrossAttentionUnetModel
from conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.optim as optim
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from tqdm import tqdm

device = torch.device("cuda:0")

class DiffusionPlanningModel(TorchModuleWrapper):
    def __init__(self, feature_dim=256, num_timesteps=100, device='cuda:6'):
        super().__init__()
        self.device = torch.device(device)
        self.model = CrossAttentionUnetModel(feature_dim).to(self.device)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
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

    def forward(self, data):
        instance_feature = data["agent"].to(self.device)
        map_instance_feature = data["map"].to(self.device)
        trajs = data["trajs"].to(self.device)

        trajs_temp = self.normalize_xy_rotation(trajs.squeeze(0))
        noise = self.pyramid_noise_like(trajs_temp)
        diffusion_output = noise

        for k in self.scheduler.timesteps[:]:
            diffusion_output = self.scheduler.scale_model_input(diffusion_output)
            noise_pred = self.model(instance_feature, map_instance_feature, k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device), diffusion_output)
            diffusion_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output
            ).prev_sample

        diffusion_output = self.denormalize_xy_rotation(diffusion_output, N=6, times=10)
        return {"trajectory": diffusion_output}

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

# 示例用法
if __name__ == "__main__":
    filename = '/home/users/xingyu.zhang/workspace/SD-origin/scripts/planning_eval_all_de_with_anchor.pkl'
    features = pickle.load(open(filename, 'rb'))
    instance_features = [features[i]['instance_feature'] for i in range(len(features))]
    map_instance_features = [features[i]['map_instance_features'] for i in range(len(features))]

    class FeaturesDataset(Dataset):
        def __init__(self, labels_file):
            with open(labels_file, 'rb') as f:
                self.labels = pickle.load(f)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            instance_feature = instance_features[idx].to(device)
            map_instance_feature = map_instance_features[idx].to(device)
            trajs = self.labels[idx]['ego_trajs'].to(device)
            return {
                "agent": instance_feature,
                "map": map_instance_feature,
                "trajs": trajs
            }

    dataset = FeaturesDataset('/home/users/xingyu.zhang/workspace/SD-origin/scripts/features_eval_all_de_with_anchor.pkl')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = DiffusionPlanningModel(feature_dim=256, device='cuda:6')
    checkpoint = torch.load('/home/users/xingyu.zhang/workspace/SparseDrive/diffusion_head/checkpoint_loss_0.005207525296216111_epoch_80_0903.pth')
    model.model.load_state_dict(checkpoint['model_state_dict'])

    diffusion_outputs = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader)):
            if batch_idx == 100:
                break
            output = model(data)
            diffusion_outputs.append(output["trajectory"])

    with open('generated_trajs_0912_new_test.pkl', 'wb') as f:
        pickle.dump(diffusion_outputs, f)