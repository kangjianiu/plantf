import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

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

def process_positions_cumsum(trajectory, num_points=6):
    """
    参考 kmeans_plan.py 的做法，对每条轨迹的 position 坐标做累加 (cumsum)，
    然后仅取前 num_points 个采样点用于聚类（若轨迹长度不足则跳过）。
    返回形如 (num_points, 2) 的数组。
    """
    if len(trajectory) < num_points:
        return None
    
    # 提取原始坐标，示例中不再做 mod 100 的相对坐标处理，如有需要可再加
    positions = np.array([p['position'] for p in trajectory], dtype=np.float32)
    # 累加坐标，模拟类似 planning 轨迹累加的做法
    positions_cumsum = np.cumsum(positions, axis=0)

    # 仅保留前 num_points 帧
    positions_cumsum = positions_cumsum[:num_points]

    return positions_cumsum  # shape (num_points, 2)

def main():
    # 参数设置
    num_points = 32        # 参考 kmeans_plan.py，取 6 帧
    n_clusters = 256        # 参考 kmeans_plan.py，聚类数为 6
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative_reshape.json'
    output_npy_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/cluster_centers_plan_style_{n_clusters}.npy"
    output_plot_path = f"/data/datasets/niukangjia/plantf/traj_data/kmeans/plan_style_kmeans_{n_clusters}.png"


    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    print("加载轨迹数据...")
    all_trajectories = load_trajectories(json_file_path)
    print(f"总轨迹数：{len(all_trajectories)}")

    # 将每条轨迹转为累加坐标，并仅取前 num_points 帧
    valid_trajs = []
    for trajectory in tqdm(all_trajectories):
        traj_cumsum = process_positions_cumsum(trajectory, num_points)
        if traj_cumsum is not None:
            # Flatten => (num_points * 2,) 形状，用于 K-Means
            valid_trajs.append(traj_cumsum.reshape(-1))

    if not valid_trajs:
        print("没有满足长度要求的轨迹，终止。")
        return

    valid_trajs = np.array(valid_trajs)  # shape (N, num_points*2)

    # 执行 K-Means 聚类
    print(f"执行 K-Means 聚类，聚类数量: {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(valid_trajs)
    cluster_centers = kmeans.cluster_centers_  # shape (n_clusters, num_points*2)
    cluster_centers = cluster_centers.reshape(n_clusters, num_points, 2)  # 恢复成 (n_clusters, num_points, 2)

    print("K-Means 聚类完成。开始可视化并保存...")

    # 可视化：画出每个聚类中心


    plt.figure()
    for i in range(n_clusters):
        plt.scatter(cluster_centers[i, :, 0], cluster_centers[i, :, 1], label=f"cluster_{i}")
    # 表明横纵坐标的坐标单位
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend()
    plt.title("Trajectory Clusters (Plan Style)")
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"可视化图已保存至 {output_plot_path}")

    # 保存聚类中心
    np.save(output_npy_path, cluster_centers)
    print(f"聚类中心已保存到: {output_npy_path}")
    print("全部操作完成。")

if __name__ == "__main__":
    main()