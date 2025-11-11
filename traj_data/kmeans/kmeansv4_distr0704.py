import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def load_trajectories(json_file_path: str):
    """读取轨迹数据"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return [t for trajectories in data.values() for t in trajectories if t]

def process_traj_cumsum(trajectory, num_points=6):
    """处理轨迹：累加坐标和速度"""
    if len(trajectory) < num_points:
        return None
    
    positions = np.array([p['position'] for p in trajectory], dtype=np.float32)
    velocities = np.array([p['velocity'] for p in trajectory], dtype=np.float32)
    
    # 累加处理
    positions_cumsum = np.cumsum(positions, axis=0)[:num_points]
    velocities_cumsum = np.cumsum(velocities, axis=0)[:num_points]
    
    return np.hstack([positions_cumsum, velocities_cumsum]).reshape(-1)

def calculate_curvature(positions):
    """正确计算曲率"""
    if len(positions) < 3:
        return []
    
    curvatures = []
    for i in range(1, len(positions)-1):
        # 三点向量计算
        v1 = positions[i] - positions[i-1]
        v2 = positions[i+1] - positions[i]
        
        # 向量模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 1e-5 and norm_v2 > 1e-5:
            # 角度计算 (弧度)
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            
            # 曲率 = 角度 / 平均距离
            avg_distance = (norm_v1 + norm_v2) / 2
            curvatures.append(angle / avg_distance)
    
    return curvatures

def get_trajectory_stats(trajectory):
    """获取轨迹的曲率和速度序列（保持长度一致）"""
    positions = np.array([p['position'] for p in trajectory])
    velocities = np.array([p['velocity'] for p in trajectory])
    
    # 只取中间点（确保曲率和速度点一一对应）
    curvatures = calculate_curvature(positions)
    
    # 获取中间点的速度 (从索引1到len-1)
    if len(velocities) > 2:
        speeds = np.linalg.norm(velocities[1:-1], axis=1)
    else:
        speeds = []
    
    # 确保数组长度匹配
    min_len = min(len(curvatures), len(speeds))
    if min_len > 0:
        return curvatures[:min_len], speeds[:min_len]
    return [], []  # 显式返回两个空列表

def main():
    # 参数设置
    num_points = 80
    n_clusters = 128
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative_reshape.json'
    output_npy_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/cluster_centers_plan_style_{n_clusters}_{num_points}_v4.npy"
    output_plot_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/plan_style_kmeans_{n_clusters}_{num_points}_v4.png"
    output_stats_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/distribution_comparison_{n_clusters}_{num_points}_v4.png"
    
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    print("加载轨迹数据...")
    all_trajectories = load_trajectories(json_file_path)
    print(f"总轨迹数：{len(all_trajectories)}")

    # 计算真实轨迹统计指标（修复长度不一致问题）
    print("计算真实轨迹统计指标...")
    real_curvatures = []
    real_speeds = []
    
    sample_size = min(5000, len(all_trajectories))
    sampled_indices = np.random.choice(len(all_trajectories), sample_size, replace=False)
    
    for i in tqdm(sampled_indices, desc="处理真实轨迹"):
        curv, spd = get_trajectory_stats(all_trajectories[i])
        # 正确检查数组是否非空
        if len(curv) > 0 and len(spd) > 0:
            real_curvatures.extend(curv)
            real_speeds.extend(spd)
    
    # 计算统计量
    real_curv_mean = np.mean(real_curvatures) if real_curvatures else 0
    real_speed_var = np.var(real_speeds) if real_speeds else 0
    print(f"真实轨迹统计 - 曲率均值: {real_curv_mean:.4f}, 速度方差: {real_speed_var:.4f}")

    # 处理有效轨迹用于聚类
    valid_trajs = []
    for trajectory in tqdm(all_trajectories, desc="处理轨迹聚类"):
        traj_cumsum = process_traj_cumsum(trajectory, num_points)
        if traj_cumsum is not None:
            valid_trajs.append(traj_cumsum)
    
    if not valid_trajs:
        print("没有满足长度要求的轨迹，终止。")
        return

    valid_trajs = np.array(valid_trajs)

    # K-Means聚类
    print(f"执行 K-Means 聚类，聚类数量: {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(valid_trajs)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = cluster_centers.reshape(n_clusters, num_points, 4)
    cluster_centers = cluster_centers[np.random.permutation(n_clusters)]

    # 计算锚点轨迹统计指标
    print("计算锚点轨迹统计指标...")
    anchor_curvatures = []
    anchor_speeds = []
    
    for i in tqdm(range(n_clusters), desc="处理锚点轨迹"):
        # 提取位置和速度
        positions = cluster_centers[i, :, :2]
        velocities = cluster_centers[i, :, 2:]
        
        # 计算曲率
        curvatures = calculate_curvature(positions)
        
        # 计算速度 (仅中间点)
        speeds = np.linalg.norm(velocities[1:-1], axis=1) if len(positions) > 2 else []
        
        # 确保长度一致
        min_len = min(len(curvatures), len(speeds))
        if min_len > 0:
            anchor_curvatures.extend(curvatures[:min_len])
            anchor_speeds.extend(speeds[:min_len])
    
    anchor_curv_mean = np.mean(anchor_curvatures) if anchor_curvatures else 0
    anchor_speed_var = np.var(anchor_speeds) if anchor_speeds else 0
    print(f"锚点轨迹统计 - 曲率均值: {anchor_curv_mean:.4f}, 速度方差: {anchor_speed_var:.4f}")

    # 可视化部分
    print("创建可视化图表...")
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # 子图1：聚类中心轨迹
    ax1 = fig.add_subplot(gs[0])
    for i in range(min(50, n_clusters)):  # 只显示前50个避免混乱
        ax1.plot(cluster_centers[i, :, 0], cluster_centers[i, :, 1], alpha=0.6)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"锚点轨迹聚类 (n={n_clusters})")
    ax1.grid(True)
    
    # 子图2：曲率-速度分布对比
    ax2 = fig.add_subplot(gs[1])
    
    # 仅当有有效数据时绘制
    valid_real_data = False
    if len(real_curvatures) > 0 and len(real_speeds) > 0:
        # 限制数据范围避免异常值
        valid_indices = (np.array(real_curvatures) < 1.0) & (np.array(real_speeds) < 50)
        if np.any(valid_indices):
            sns.kdeplot(
                x=np.array(real_speeds)[valid_indices],
                y=np.array(real_curvatures)[valid_indices],
                cmap="Blues",
                fill=True,
                thresh=0.05,
                alpha=0.5,
                label='真实轨迹分布',
                ax=ax2
            )
            valid_real_data = True
    
    valid_anchor_data = False
    if len(anchor_curvatures) > 0 and len(anchor_speeds) > 0:
        # 限制数据范围避免异常值
        valid_indices = (np.array(anchor_curvatures) < 1.0) & (np.array(anchor_speeds) < 50)
        if np.any(valid_indices):
            ax2.scatter(
                np.array(anchor_speeds)[valid_indices],
                np.array(anchor_curvatures)[valid_indices],
                color='red',
                s=40,
                alpha=0.7,
                label='锚点轨迹'
            )
            valid_anchor_data = True
    
    # 添加统计标注
    stats_text = (f"真实轨迹: μ_c={real_curv_mean:.4f}, σ_v²={real_speed_var:.4f}\n"
                  f"锚点轨迹: μ_c={anchor_curv_mean:.4f}, σ_v²={anchor_speed_var:.4f}")
    
    # 计算KL散度 (仅当有足够数据时)
    kl_div = "N/A"
    if valid_real_data and valid_anchor_data:
        try:
            # 使用相同边界
            _, xedges, yedges = np.histogram2d(
                real_speeds, real_curvatures, bins=20, density=True
            )
            hist_anchor = np.histogram2d(
                anchor_speeds, anchor_curvatures, bins=[xedges, yedges], density=True
            )[0]
            
            # 添加小值避免除零
            hist_real = np.histogram2d(
                real_speeds, real_curvatures, bins=[xedges, yedges], density=True
            )[0] + 1e-10
            hist_anchor += 1e-10
            
            # 计算KL散度
            kl_div = np.sum(hist_real * np.log(hist_real / hist_anchor))
            stats_text += f"\nKL散度: {kl_div:.4f}"
        except Exception as e:
            print(f"计算KL散度时出错: {e}")
    
    # 添加统计文本
    ax2.text(
        0.05, 0.95, stats_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax2.set_title("曲率-速度分布对比")
    ax2.set_xlabel("速度 (m/s)")
    ax2.set_ylabel("曲率 (1/m)")
    if valid_real_data or valid_anchor_data:
        ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化图已保存至 {output_plot_path}")
    
    # 单独保存分布对比图
    if valid_real_data or valid_anchor_data:
        fig2 = plt.figure(figsize=(10, 8))
        if valid_real_data:
            sns.kdeplot(
                x=np.array(real_speeds)[valid_indices],
                y=np.array(real_curvatures)[valid_indices],
                cmap="Blues",
                fill=True,
                thresh=0.05,
                alpha=0.5,
                label='真实轨迹分布'
            )
        if valid_anchor_data:
            plt.scatter(
                np.array(anchor_speeds)[valid_indices],
                np.array(anchor_curvatures)[valid_indices],
                color='red',
                s=40,
                alpha=0.7,
                label='锚点轨迹'
            )
        plt.title("曲率-速度分布对比")
        plt.xlabel("速度 (m/s)")
        plt.ylabel("曲率 (1/m)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.text(
            0.05, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        plt.savefig(output_stats_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"分布对比图已单独保存至 {output_stats_path}")
    else:
        print("没有有效数据创建分布对比图")

    # 保存聚类中心
    np.save(output_npy_path, cluster_centers)
    print(f"聚类中心已保存到: {output_npy_path}")
    print("全部操作完成。")

if __name__ == "__main__":
    main()