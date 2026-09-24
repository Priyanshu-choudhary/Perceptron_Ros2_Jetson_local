#!/bin/bash
# Inside the chroot on the edge board (files staged in /work, see README):
#   /work/run.sh NAME --launch-arg ... --params-file /work/params/X.yaml
source /opt/ros/humble/setup.bash
export PYTHONPATH=/work/pylib:$PYTHONPATH
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 HOME=/work
NAME=$1; shift
exec python3 -u /work/profile_run.py --name "$NAME" --launch-file /work/edge_nav.launch.py \
  --zmq-url tcp://127.0.0.1:5555 --pose "${POSE:-1.085,-0.575,2.5787}" --goal "${GOAL:-0.2035,0.2768,2.5787}" \
  --out /work/results --extra-pattern "jetson_robot_bridge.py(serial)=jetson_robot_bridge.py" \
  --idle 30 --nav 30 --nomotion-hz 0 --goal-latest "$@"
