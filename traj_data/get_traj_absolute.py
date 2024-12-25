import os
import sys
import json
from pathlib import Path
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_sensor_token_by_index_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_token_map_name_from_db,
    SensorDataSource
)

def get_all_db_files(directory: str) -> list:
    """获取指定目录下的所有.db文件"""
    return [str(path) for path in Path(directory).rglob('*.db')]

def process_db_file(db_file: str):
    """处理单个数据库文件，获取历史轨迹"""
    sensor_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token','MergedPointCloud')
    
    # 获取 initial_lidar_token 和 initial_lidar_timestamp
    initial_lidar_token = get_sensor_token_by_index_from_db(db_file, sensor_source, 0)
    initial_lidar_timestamp = get_sensor_data_token_timestamp_from_db(db_file, sensor_source, initial_lidar_token)
    map_name = get_sensor_token_map_name_from_db(db_file, sensor_source, initial_lidar_token)
    map_version = "nuplan-maps-v1.0"  # 假设地图版本是固定的

    scenario = NuPlanScenario(
        data_root='/data/datasets/niukangjia/nuplan/dataset/nuplan-v1.1',
        log_file_load_path=db_file,
        initial_lidar_token=initial_lidar_token,
        initial_lidar_timestamp=initial_lidar_timestamp,
        scenario_type='scenario_type',  # 需要替换为实际的场景类型
        map_root='/data/datasets/niukangjia/nuplan/maps',
        map_version=map_version,
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name="scenario_name", scenario_duration=20, extraction_offset=1, subsample_ratio=0.5
        ),  # 或者提供实际的 ScenarioExtractionInfo 对象
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=None
    )

    # 保存所有历史轨迹
    all_trajectories = []

    # 获取全部iteration历史轨迹
    # 打印scenario的iteration数量
    # print(f"Number of scenario iterations: {scenario.get_number_of_iterations()}")
    for iteration in range(scenario.get_number_of_iterations()):
        past_trajectory = list(scenario.get_ego_future_trajectory(iteration, time_horizon=2))  # 例如，获取10秒的轨迹
        iteration_trajectory = []
        for state in past_trajectory:
            trajectory_point = {
                "time_us": state.time_point.time_us,
                "position": (round(state.rear_axle.x, 6), round(state.rear_axle.y, 6)),
                "velocity": (round(state.dynamic_car_state.rear_axle_velocity_2d.x, 6), round(state.dynamic_car_state.rear_axle_velocity_2d.y, 6))
            }
            iteration_trajectory.append(trajectory_point)
            # print(f"Time: {trajectory_point['time_us'] / 1e6:.6f}, Position: {trajectory_point['position']}, Velocity: {trajectory_point['velocity']}")
            
        all_trajectories.append(iteration_trajectory)
        # print(iteration)
        # # for循环只运行4次
        # if iteration == 4:
        #     break


    return all_trajectories

def main():
    db_directory = '/data/datasets/niukangjia/nuplan/dataset/nuplan-v1.1/splits/mini'
    db_files = get_all_db_files(db_directory)
    all_db_trajectories = {}
    for i, db_file in enumerate(db_files):
        # if i >= 3:
        #     break
        # 打印“正在处理第i个of总数”
        print(f"Processing {i+1} of {len(db_files)}")
        all_db_trajectories[db_file] = process_db_file(db_file)
    # 新建程序所在目录的一个json文件，把all_db_trajectories存到里面
    with open('all_db_trajectories_absolute.json', 'w') as f:
        json.dump(all_db_trajectories, f)
    

    # with open('all_db_trajectories.txt', 'w') as f:
    #     for db_file, trajectories in all_db_trajectories.items():
    #         f.write(f"{db_file}\n")
    #         for trajectory in trajectories:
    #             f.write(f"{trajectory}\n")


if __name__ == "__main__":
    main()