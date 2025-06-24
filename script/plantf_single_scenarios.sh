cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan_challenge \
    scenario_filter=mini \
    worker=sequential \
    verbose=true \
    planner.imitation_planner.planner_ckpt="/data/datasets/niukangjia/plantf/output/training/planTF/2025.03.14.19.26.54/checkpoints/last.ckpt"