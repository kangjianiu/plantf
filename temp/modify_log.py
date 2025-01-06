import re

# 定义输入和输出文件路径
input_file = '/data/datasets/niukangjia/nuplan/exp/exp/training/planTF/2025.01.03.18.15.21/run_training.log'
output_file = '/data/datasets/niukangjia/nuplan/exp/exp/training/planTF/2025.01.03.18.15.21/run_training_2.log'

# 编译正则表达式
pattern = re.compile(r'\[INFO\]')

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if not pattern.search(line):
            outfile.write(line)