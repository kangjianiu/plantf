from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math
import sys

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class Downsample1d(nn.Module):
    """
    输入: [batch_size, 128, 32]
    卷积操作: 使用 kernel_size=3, stride=2, padding=1
    输出: [batch_size, 128, 16]
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, # 2    第二轮128
            out_channels, #128   第二轮256
            cond_dim, # 384
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # 第一轮：x:[256, 4, 80] -> [batch_size,4,future_steps] , cond:[32, 384]  
        # 第二轮：x:[256, 128, 16]                                cond:[32, 384]
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]
            
            returns:
            out : [ batch_size x out_channels x horizon ]
            
        '''
        out = self.blocks[0](x)          #  [32, 128, 80]     

        embed = self.cond_encoder(cond)  # [32, 128, 1]       
        # 计算out和embed的最后一个维度的除法
        # time = out.shape[-1]//embed.shape[-1] # 32//1=32
        # embed = embed.repeat(8, 1, time)   # [256, 128, 32]     
        # print( "out shape:", out.shape,"embed shape:", embed.shape)          

        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed #
        out = self.blocks[1](out)#  [32, 128, 32]
        out = out + self.residual_conv(x)# [256, 128, 32]
        # # print("out shape:", out.shape)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,# 4
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)# input_dim=4, down_dims=[128,256],所以all_dims=[4,128,256]
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))# in_out=[(4, 128), (128, 256)]
        # print(f"in_out:{in_out}")
        # print(f"dsed:{dsed}\ncond_dim:{cond_dim}\nglobal_cond_dim:{global_cond_dim}\nstart_dim:{start_dim}\nlocal_cond_dim:{local_cond_dim}")
        #  dsed:256     cond_dim:384     global_cond_dim:128    start_dim:128

        # global_feature = global_feature + global_cond     = 256 + 128 = 384 (在后面)
        # cond_dim       = desd           + global_cond_dim = 256 + 128 = 384
        # global_feature 由 timestep 经过 diffusion_step_encoder 得到，所以维度与参数 dsed 相同。

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]# 256
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):# in_out=[(4, 128), (128, 256)]
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, #[batch_size, future_steps, 4][32,80,4]
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):#global_cond :[batch_size, embed_dim]
        """
        x: (B,T,input_dim)                     #[256, 32, 2]
        timestep: (B,) or int, diffusion step  #[32]
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)       #ego信息[32, 128]
        output: (B,T,input_dim)                #应该是[256, 32, 2]
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')#[batch_size,4,future_steps] 
        #print(f"============unet1d开始===========\ntimesteps1:{timestep.shape}") # timesteps1: torch.Size([32])
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
 
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(global_cond.shape[0]) 
        # print("============unet1d开始===========\ntimesteps3:",timesteps.shape) # timesteps1: torch.Size([32])

        global_feature = self.diffusion_step_encoder(timesteps)

        # print("global_feature shape:", global_feature.shape)# 由时间步解码得到 shape:[32, 256]
        # print("global_cond shape:", global_cond.shape)      # 本质上是ego信息  shape: [32, 128]

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        # 此后global_feature 同时包含了时间步信息和ego信息[32, 384]
        # print("cat之后的global_feature:", global_feature.shape)# cat之后[32, 384]

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        # print("x.shape:", x.shape) #[batch_size,4,future_steps] 
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            # print(f"=====x即将通过第{idx+1}轮=====")
            x = resnet(x, global_feature)  #[batch_size,4,future_steps] ,  [32, 384]
            # print(f"x已经通过第{idx+1}个resnet模块,shape:{x.shape}")# [32, 128, 80]
            # print(f"第 {idx+1} 个 resnet模块 的dim_in:{resnet.in_channels} , dim_out:{resnet.out_channels} , cond_dim: {resnet.cond_dim}")

            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            # print(f"x已经通过第{idx+1}个resnet2模块,shape:{x.shape}")
            # print(f"第{idx+1}个resnet2模块的dim_in:{resnet2.in_channels},dim_out:{resnet2.out_channels},cond_dim:{resnet2.cond_dim}")
            h.append(x)
            x = downsample(x)
            # print(f"x已经通过第{idx+1}个downsample模块,shape:{x.shape}")

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        # print("x.shape:", x.shape,"\n=============unet1d结尾============\n")
        return x


if __name__ == "__main__":
    noise = torch.randn(2, 6, 2)
    global_feature = torch.randn(2, 256)
    timestep = torch.randint(0, 1000, (2,))
    noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=256,
                down_dims=[128,256],
                cond_predict_scale=False,
    )
    noise_pred = noise_pred_net(noise, timestep=timestep, global_cond=global_feature)
    ## print(noise_pred.shape)

