cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
export PYTHONPATH="$cwd/src:$PYTHONPATH"
source /home/ustc/anaconda3/bin/activate pluto
PLANNER="planTF"
SPLIT=$1
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

ray start --head --env PYTHONPATH="$PYTHONPATH"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=2 \
        experiment_uid=$SPLIT/$PLANNER \
        verbose=true \
        planner.imitation_planner.planner_ckpt="$CKPT_ROOT/$PLANNER.ckpt"
done

ray stop
