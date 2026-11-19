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
# Keep this laptop's nodes to itself: teammates on the same Wi-Fi with the same domain would
# otherwise also command the car (their idle keyboard node sends stop commands -> staccato).
# /opt/ros/humble/setup.bash sets ROS_LOCALHOST_ONLY=0 itself, so a default would never apply:
# set it explicitly (CAR_LOCALHOST_ONLY=0 ./run_traffic_light.sh ... to talk to other computers).
export ROS_LOCALHOST_ONLY="${CAR_LOCALHOST_ONLY:-1}"
# 키보드 창(gnome-terminal)은 systemd 서비스가 띄우는데, 그 환경에 DISPLAY가
# 빠져 있으면 "Cannot open display"로 창이 뜨지 않음. 현재 화면 정보를 넘겨줌.
if [[ -n "${DISPLAY:-}" ]]; then
  dbus-update-activation-environment --systemd DISPLAY XAUTHORITY 2>/dev/null || true
fi
# 숫자 하나만 주면 주행 속도: ./run_autonomous.sh 200  ==  cruise_speed:=200
args=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    if (( arg > 255 )); then
      echo "속도는 0~255 사이로 지정하세요: $arg" >&2
      exit 1
    fi
    arg="cruise_speed:=$arg"
  fi
  args+=("$arg")
done
set -- "${args[@]}"
# cam_num:=N 이나 CAMERA_INDEX 가 없으면 차량 카메라(C920)를 자동으로 찾음.
# 노트북 내장 웹캠(video0)으로 주행하는 것을 막기 위함.
if [[ -z "${CAMERA_INDEX:-}" && " $* " != *" cam_num:="* && " $* " != *" data_source:="* ]]; then
  cam_link="$(ls /dev/v4l/by-id/*C920*-video-index0 2>/dev/null | head -n1 || true)"
  if [[ -n "$cam_link" ]]; then
    cam_dev="$(readlink -f "$cam_link")"
    export CAMERA_INDEX="${cam_dev#/dev/video}"
    echo "차량 카메라 C920: $cam_dev (cam_num:=$CAMERA_INDEX)"
  else
    echo "C920 카메라를 찾지 못했습니다. USB 연결을 확인하거나 cam_num:=번호 를 지정하세요." >&2
    exit 1
  fi
fi
exec ros2 launch "$WS_DIR/autonomous_driving.launch.py" "$@"
