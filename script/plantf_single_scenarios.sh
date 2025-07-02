# 修改后脚本
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan_challenge \
    scenario_filter=single_right_turn \
    worker=sequential \
    verbose=true \
    planner.imitation_planner.planner_ckpt='/data/datasets/niukangjia/plantf/output/training/planTF/2025.07.01.11.04.10/checkpoints/epoch-8-val_minFDE-0.000.ckpt'