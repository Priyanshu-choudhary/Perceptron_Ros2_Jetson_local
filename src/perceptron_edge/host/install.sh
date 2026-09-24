#!/bin/bash
# One-time (and after-update) setup on the Jetson host. Run with sudo:
#
#   sudo /mnt/sdcard/ros2_chroot/opt/perceptron_ws/src/perceptron_edge/host/install.sh
#
# Expects the ROS 2 Humble chroot at /mnt/sdcard/ros2_chroot (host/setup_chroot.sh
# next to this file) and this workspace's src/ synced to /opt/perceptron_ws/src
# inside it. Builds the workspace, installs the `robot` command, and links
# ~/robot and ~/robot_maps. Touches nothing outside /mnt/sdcard except those two
# links in /home/jetson.
set -eu
R=/mnt/sdcard/ros2_chroot
WS=/opt/perceptron_ws
BASE=/mnt/sdcard/perceptron
HERE=$(cd "$(dirname "$0")" && pwd)
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

# A user 'jetson' (uid 1000) inside the chroot, in dialout for the CT6B port,
# so ROS runs as the same unprivileged user as on the host - never as root.
grep -q '^jetson:' "$R/etc/passwd" || echo "jetson:x:1000:1000:jetson:$WS/home:/bin/bash" >> "$R/etc/passwd"
grep -q '^jetson:' "$R/etc/group" || echo "jetson:x:1000:" >> "$R/etc/group"
grep -q '^dialout:.*jetson' "$R/etc/group" || sed -i 's/^\(dialout:x:20:\)\(.*\)$/\1\2,jetson/; s/:,jetson$/:jetson/' "$R/etc/group"

mkdir -p "$R$WS"/{src,maps,config,home,log}
chown -R 1000:1000 "$R$WS"

for m in proc sys dev dev/pts dev/shm; do mountpoint -q "$R/$m" || mount --bind "/$m" "$R/$m"; done

echo "building workspace (a few minutes on the Nano)..."
chroot --userspec=1000:1000 "$R" /usr/bin/env -i HOME=$WS/home LANG=C.UTF-8 PATH=/usr/bin:/bin \
    /bin/bash -c "source /opt/ros/humble/setup.bash && cd $WS && \
    colcon build --symlink-install --packages-select perceptron_hardware perceptron_navigation \
      perceptron_robot_description perceptron_robot_control perceptron_robot_bringup ct6b_teleop perceptron_edge 2>&1 | tail -4"

mkdir -p "$BASE"/{bin,logs,run}
install -m 755 "$HERE/robot" "$BASE/bin/robot"
install -m 644 "$HERE/robot_cpu.py" "$BASE/bin/robot_cpu.py"
chown -R 1000:1000 "$BASE"
ln -sfn "$BASE/bin/robot" /home/jetson/robot
ln -sfn "$R$WS/maps" /home/jetson/robot_maps
echo "installed: ~/robot  (try: ~/robot help)"
