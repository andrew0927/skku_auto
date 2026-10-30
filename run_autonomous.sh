#!/usr/bin/env bash
set -eo pipefail
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
if [[ ! -f "$WS_DIR/install/local_setup.bash" ]]; then
  echo "먼저 실행: cd \"$WS_DIR\" && colcon build --symlink-install" >&2
  exit 1
fi
source "$WS_DIR/install/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
exec ros2 launch "$WS_DIR/autonomous_driving.launch.py" "$@"
