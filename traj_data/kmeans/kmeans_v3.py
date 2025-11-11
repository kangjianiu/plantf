import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import math

def load_trajectories(json_file_path: str):
    """
    读取 JSON 文件并提取所有轨迹数据。
    每个轨迹是一个包含多个采样点的列表，点内含有 'time_us', 'position', 'velocity' 等键。
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    all_trajectories = []
    for db_file, trajectories in data.items():
        for trajectory in trajectories:
            if trajectory:  # 跳过空轨迹
                all_trajectories.append(trajectory)
    return all_trajectories

def process_traj_cumsum(trajectory, num_points=6):
    """
    对每条轨迹的 position 坐标和 velocity 做累加 (cumsum)，
    然后仅取前 num_points 个采样点用于聚类（若轨迹长度不足则跳过）。
    返回形如 (num_points * 4,) 的数组，其中每个点包含 (x, y, vx, vy)。
    """
    if len(trajectory) < num_points:
        return None
    
    # 提取原始坐标和速度
    positions = np.array([p['position'] for p in trajectory], dtype=np.float32)
    velocities = np.array([p['velocity'] for p in trajectory], dtype=np.float32)
    
    # 累加坐标和速度
    positions_cumsum = np.cumsum(positions, axis=0)
    velocities_cumsum = np.cumsum(velocities, axis=0)

    # 仅保留前 num_points 帧
    positions_cumsum = positions_cumsum[:num_points]
    velocities_cumsum = velocities_cumsum[:num_points]

    # 合并位置和速度，形成 (num_points, 4) 的数组
    combined = np.hstack([positions_cumsum, velocities_cumsum])  # shape (num_points, 4)

    return combined.reshape(-1)  # shape (num_points * 4,)

def calculate_curvature(positions):
    """
    计算轨迹的曲率，使用三点法拟合圆
    positions: 形状为 (N, 2) 的数组，包含x和y坐标
    返回：曲率数组，长度为 N-2
    """
    curvatures = []
    for i in range(1, len(positions)-1):
        x0, y0 = positions[i-1]
        x1, y1 = positions[i]
        x2, y2 = positions[i+1]
        
        # 计算两条线段的中垂线
        mid1_x = (x0 + x1) / 2
        mid1_y = (y0 + y1) / 2
        if x1 != x0:
            slope1 = (y1 - y0) / (x1 - x0)
            perp_slope1 = -1 / slope1
        else:
            perp_slope1 = 0
        
        mid2_x = (x1 + x2) / 2
        mid2_y = (y1 + y2) / 2
        if x2 != x1:
            slope2 = (y2 - y1) / (x2 - x1)
            perp_slope2 = -1 / slope2
        else:
            perp_slope2 = 0
        
        # 解两条中垂线的交点，即圆心
        A = perp_slope1
        B = -1
        C = mid1_y - perp_slope1 * mid1_x
        
        D = perp_slope2
        E = -1
        F = mid2_y - perp_slope2 * mid2_x
        
        det = A * E - B * D
        if det == 0:
            curvatures.append(0.0)
            continue
        
        center_x = (B * F - E * C) / det
        center_y = (D * C - A * F) / det
        
        radius = np.sqrt((center_x - x1)**2 + (center_y - y1)** 2)
        if radius < 1e-6:
            curvatures.append(0.0)
        else:
            curvatures.append(1.0 / radius)
    
    return np.array(curvatures)

def calculate_speed_distribution(velocities):
    """计算速度大小的分布"""
    speed_magnitudes = np.linalg.norm(velocities, axis=1)
    return speed_magnitudes

def main():
    # 参数设置
    num_points = 80        # 参考 kmeans_plan.py，取 80 帧
    n_clusters = 128       # 参考 kmeans_plan.py，聚类数为 256
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative_reshape.json'
    output_npy_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/cluster_centers_plan_style_{n_clusters}_{num_points}_v4.npy"
    output_plot_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/plan_style_kmeans_{n_clusters}_{num_points}_v4.png"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    print("加载轨迹数据...")
    all_trajectories = load_trajectories(json_file_path)
    print(f"总轨迹数：{len(all_trajectories)}")

    # 将每条轨迹转为累加坐标和速度，并仅取前 num_points 帧
    valid_trajs = []
    all_curvatures = []
    all_speeds = []
    for trajectory in tqdm(all_trajectories, desc="处理轨迹"):
        traj_cumsum = process_traj_cumsum(trajectory, num_points)
        if traj_cumsum is not None:
            valid_trajs.append(traj_cumsum)
        
        # 计算原始轨迹的曲率和速度分布
        if len(trajectory) >= num_points:
            positions = np.array([p['position'] for p in trajectory[:num_points]], dtype=np.float32)
            velocities = np.array([p['velocity'] for p in trajectory[:num_points]], dtype=np.float32)
            
            curvatures = calculate_curvature(positions)
            if curvatures.size > 0:
                all_curvatures.extend(curvatures)
            
            speeds = calculate_speed_distribution(velocities)
            all_speeds.extend(speeds)

    if not valid_trajs:
        print("没有满足长度要求的轨迹，终止。")
        return

    valid_trajs = np.array(valid_trajs)  # shape (N, num_points*4)

    # 执行 K-Means 聚类
    print(f"执行 K-Means 聚类，聚类数量: {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters,  init='k-means++', random_state=42)
    kmeans.fit(valid_trajs)
    cluster_centers = kmeans.cluster_centers_  # shape (n_clusters, num_points*4)
    cluster_centers = cluster_centers.reshape(n_clusters, num_points, 4)  # 恢复成 (n_clusters, num_points, 4)
    # 在第一个维度随机打乱顺序
    cluster_centers = cluster_centers[np.random.permutation(n_clusters)]

    # 计算聚类中心的曲率和速度分布
    print("计算锚点轨迹统计指标...")
    cluster_curvatures = []
    cluster_speeds = []
    for center in cluster_centers:
        positions = center[:, :2]  # x, y
        velocities = center[:, 2:]  # vx, vy
        
        curvatures = calculate_curvature(positions)
        if curvatures.size > 0:
            cluster_curvatures.extend(curvatures)
        
        speeds = calculate_speed_distribution(velocities)
        cluster_speeds.extend(speeds)

    # 绘制分布对比图
    print("绘制曲率和速度分布对比图...")
    plt.figure(figsize=(12, 5))
    
    # 曲率分布
    plt.subplot(1, 2, 1)
    plt.hist(all_curvatures, bins=50, alpha=0.5, label='真实轨迹')
    plt.hist(cluster_curvatures, bins=50, alpha=0.5, label='锚点轨迹')
    plt.xlabel('曲率')
    plt.ylabel('频率')
    plt.title('曲率分布对比')
    plt.legend()
    
    # 速度分布
    plt.subplot(1, 2, 2)
    plt.hist(all_speeds, bins=50, alpha=0.5, label='真实轨迹')
    plt.hist(cluster_speeds, bins=50, alpha=0.5, label='锚点轨迹')
    plt.xlabel('速度大小')
    plt.ylabel('频率')
    plt.title('速度分布对比')
    plt.legend()
    
    plt.tight_layout()
    distribution_plot_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/distribution_comparison_{n_clusters}_{num_points}.png"
    plt.savefig(distribution_plot_path, bbox_inches='tight')
    plt.close()
    print(f"分布对比图已保存至 {distribution_plot_path}")

    print("K-Means 聚类完成。开始可视化并保存...")

    # 可视化：画出每个聚类中心的轨迹
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        # 仅绘制位置部分 (x, y)
        plt.plot(cluster_centers[i, :, 0], cluster_centers[i, :, 1], label=f"cluster_{i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Clusters with Cumulative Sum of Position and Velocity")
    # 为避免图例过多导致混乱，可以选择不显示图例或只显示部分
    plt.legend(ncol=4, fontsize='x-small')  # 可选
    plt.grid(True)
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"可视化图已保存至 {output_plot_path}")

    # 保存聚类中心
    np.save(output_npy_path, cluster_centers)
    print(f"聚类中心已保存到: {output_npy_path}")
    print("全部操作完成。")

if __name__ == "__main__":
    main()

