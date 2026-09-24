#!/bin/bash
# One measured run of nav.launch.py (via profile_nav.launch.py, motors disconnected).
#   ./run_laptop.sh NAME [--launch-arg k:=v ...] [--params-file params/X.yaml] [--amcl-set JSON]
# POSE/GOAL: where the parked robot is (map frame) and a goal ~1.2 m ahead of it.
# JETSON: 192.168.1.7 for the live robot, 127.0.0.1 with replay/fake_jetson.py.
source /opt/ros/humble/setup.bash
source $HOME/perceptron_test_ws/install/setup.bash
H=$(dirname "$(readlink -f "$0")")
JETSON=${JETSON:-192.168.1.7}
POSE=${POSE:-1.085,-0.575,2.5787}
GOAL=${GOAL:-0.2035,0.2768,2.5787}
NAME=$1; shift
exec python3 -u $H/profile_run.py --name "$NAME" --pose "$POSE" --goal "$GOAL" \
  --zmq-url tcp://$JETSON:5555 --launch-arg jetson:=$JETSON \
  --launch-arg map:=$HOME/perceptron_test_ws/src/perceptron_navigation/maps/room_map.yaml \
  --launch-arg overlay:=false --idle 30 --nav 25 --settle 10 --nomotion-hz 5 --out $H/results "$@"
