#!/bin/bash
# Self-contained ROS 2 Humble (Ubuntu 22.04 arm64) chroot on the data SD card.
# Nothing outside /mnt/sdcard/ros2_chroot is modified except temporary bind mounts.
set -e
R=/mnt/sdcard/ros2_chroot

S() { sudo "$@"; }
mkdir -p /mnt/sdcard/ros2_dl
cd /mnt/sdcard/ros2_dl
if [ ! -f ubuntu-base.tar.gz ]; then
  wget -q --limit-rate=2m -O ubuntu-base.tar.gz \
    http://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release/ubuntu-base-22.04.5-base-arm64.tar.gz
fi
if [ ! -d $R/etc ]; then
  S mkdir -p $R
  S tar -xpzf ubuntu-base.tar.gz -C $R
fi
S cp /etc/resolv.conf $R/etc/resolv.conf
for m in proc sys dev dev/pts; do
  mountpoint -q $R/$m || S mount --bind /$m $R/$m
done
mountpoint -q $R/dev/shm || S mount --bind /dev/shm $R/dev/shm
S mkdir -p $R/work
cat > /tmp/inside.sh <<'IN'
set -e
export DEBIAN_FRONTEND=noninteractive
echo 'Acquire::http::Dl-Limit "2000";' > /etc/apt/apt.conf.d/99limit
apt-get update -q
apt-get install -y -q --no-install-recommends curl gnupg ca-certificates locales
locale-gen en_US.UTF-8
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list
apt-get update -q
apt-get install -y -q --no-install-recommends \
  ros-humble-ros-base ros-humble-launch-ros ros-humble-rclcpp-components \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher ros-humble-robot-localization \
  ros-humble-nav2-common ros-humble-nav2-msgs ros-humble-nav2-util ros-humble-nav2-lifecycle-manager \
  ros-humble-nav2-map-server ros-humble-nav2-amcl ros-humble-nav2-costmap-2d \
  ros-humble-nav2-controller ros-humble-nav2-planner ros-humble-nav2-smoother ros-humble-nav2-behaviors \
  ros-humble-nav2-bt-navigator ros-humble-nav2-behavior-tree ros-humble-nav2-waypoint-follower \
  ros-humble-nav2-velocity-smoother ros-humble-dwb-core ros-humble-dwb-plugins ros-humble-dwb-critics \
  ros-humble-nav2-rotation-shim-controller ros-humble-nav2-smac-planner ros-humble-nav2-navfn-planner \
  ros-humble-nav2-theta-star-planner ros-humble-nav2-mppi-controller \
  python3-zmq python3-msgpack python3-numpy python3-psutil python3-yaml
echo INSIDE_DONE
IN
S cp /tmp/inside.sh $R/work/inside.sh
S chroot $R /bin/bash /work/inside.sh
echo SETUP_DONE
