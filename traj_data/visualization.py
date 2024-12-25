import json
import matplotlib.pyplot as plt

def read_trajectories_from_file(file_path: str, db_file_key: str):
    """读取文件中的特定轨迹数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[db_file_key]

def process_position(position):
    """处理坐标数据"""
    return [round(position[0] % 100, 1), round(position[1] % 100, 1)]

def average(values):
    """计算平均值"""
    return sum(values) / len(values)

def visualize_trajectories(trajectories, output_path: str):
    """可视化轨迹数据"""
    for i, trajectory in enumerate(trajectories):
        if i % 10 == 0:
            print(f"Visualizing trajectory {i+1}")
        positions = [process_position(point['position']) for point in trajectory]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # x_max = average(x_coords) + 2
        # x_min = average(x_coords) - 2
        # y_max = average(y_coords) + 10
        # y_min = average(y_coords) - 10

        plt.figure()
        plt.plot(x_coords, y_coords, marker='o')
        
        # 为每个点标注序号
        for point_idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.annotate(str(point_idx), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory {i+1}')
        plt.savefig(f'{output_path}/trajectory_{i+1}.png')
        plt.close()

def main():
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/reformatted_all_db_trajectories.json'  # 替换为实际的 JSON 文件路径
    db_file_key = '/data/datasets/niukangjia/nuplan/dataset/nuplan-v1.1/splits/mini/2021.06.09.11.54.15_veh-12_04366_04810.db'
    output_path = '/data/datasets/niukangjia/plantf/traj_data/visualization'  # 替换为实际的输出路径

    trajectories = read_trajectories_from_file(json_file_path, db_file_key)
    visualize_trajectories(trajectories, output_path)

if __name__ == "__main__":
    main()