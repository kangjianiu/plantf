import json

def reformat_json_layout(json_file_path: str, output_file_path: str):
    """将 JSON 文件的布局格式修改为类似 temp.txt 的布局格式"""
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    with open(output_file_path, 'w') as output_file:
        output_file.write("{\n")
        for db_file, trajectory_list in data.items():
            output_file.write(f'    "{db_file}": \n')
            output_file.write("    [\n")
            for trajectory in trajectory_list:
                output_file.write("        [\n")
                for point in trajectory:
                    output_file.write(f'            {json.dumps(point)}')
                    if point != trajectory[-1]:
                        output_file.write(",")
                    output_file.write("\n")
                output_file.write("        ]")
                if trajectory != trajectory_list[-1]:
                    output_file.write(",")
                output_file.write("\n")
            output_file.write("    ]")
            if db_file != list(data.keys())[-1]:
                output_file.write(",")
            output_file.write("\n")
        output_file.write("}\n")

def main():
    json_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative.json'  # 替换为实际的 JSON 文件路径
    output_file_path = '/data/datasets/niukangjia/plantf/traj_data/all_db_trajectories_relative_reshape.json'  # 替换为实际的输出文件路径
    reformat_json_layout(json_file_path, output_file_path)

if __name__ == "__main__":
    main()