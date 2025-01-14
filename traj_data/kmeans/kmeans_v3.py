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

def main():
    # 参数设置
    num_points = 80        # 参考 kmeans_plan.py，取 80 帧
    n_clusters = 256       # 参考 kmeans_plan.py，聚类数为 256
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative_reshape.json'
    output_npy_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/cluster_centers_plan_style_{n_clusters}_{num_points}_vxy.npy"
    output_plot_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/plan_style_kmeans_{n_clusters}_{num_points}_vxy.png"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    print("加载轨迹数据...")
    all_trajectories = load_trajectories(json_file_path)
    print(f"总轨迹数：{len(all_trajectories)}")

    # 将每条轨迹转为累加坐标和速度，并仅取前 num_points 帧
    valid_trajs = []
    for trajectory in tqdm(all_trajectories, desc="处理轨迹"):
        traj_cumsum = process_traj_cumsum(trajectory, num_points)
        if traj_cumsum is not None:
            # Flatten => (num_points * 4,) 形状，用于 K-Means
            valid_trajs.append(traj_cumsum)

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