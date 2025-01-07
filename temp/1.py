import yaml

def main():
    temp_file = "/data/datasets/niukangjia/plantf/temp/temp.txt"
    config_file = "/data/datasets/niukangjia/plantf/output/training/planTF/2025.01.07.15.43.33/code/hydra/config.yaml"

    # 读取 temp.txt
    with open(temp_file, 'r') as f:
        temp_entries = set(line.strip() for line in f if line.strip())

    # 读取 config.yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    val_entries = set(config.get('val', []))

    # 找出匹配的文件夹名称
    matches = temp_entries.intersection(val_entries)

    print("在 config.yaml 的 val: 部分中出现的文件夹名称：")
    for match in matches:
        print(match)

if __name__ == "__main__":
    main()