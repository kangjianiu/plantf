cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
# CKPT_ROOT="/data/datasets/niukangjia/plantf/output/training/planTF/2025.03.14.19.26.54/checkpoints"
export PYTHONPATH="$cwd/src:$PYTHONPATH"
# source /home/ustc/anaconda3/bin/activate pluto
PLANNER="planTF"
SPLIT=$1
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

# # ray start --head --env PYTHONPATH="$PYTHONPATH"
# ray start --head --env PYTHONPATH="/data/datasets/niukangjia/plantf:$PYTHONPATH"
export PYTHONPATH="/data/datasets/niukangjia/plantf:$PYTHONPATH"
ray start --head
for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=2 \
        experiment_uid=$SPLIT/$PLANNER \
        verbose=true \
        planner.imitation_planner.planner_ckpt="/data/datasets/niukangjia/plantf/output/training/planTF/2025.06.28.12.42.52/checkpoints/epoch=35-val_minFDE=0.000.ckpt"
done

ray stop
