#!/usr/bin/env bash
# 기록한 주행을 차 없이 재생: 저장된 카메라 영상만으로 action을 다시 뽑고 정답과 비교.
#   ./replay_lap.sh laps/lap_20261108_153000            (가장 최근 기록: ./replay_lap.sh)
#   ./replay_lap.sh laps/lap_... steering_gain:=0.1     (설정을 바꿔서 비교)
# 카메라·아두이노·키보드는 전혀 열지 않음.
set -eo pipefail
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lap="${1:-$(ls -td "$WS_DIR"/laps/lap_* 2>/dev/null | head -n1)}"
[[ $# -gt 0 ]] && shift
if [[ -z "$lap" || ! -d "$lap/bag" ]]; then
  echo "기록이 없습니다. 먼저: ./run_traffic_light.sh 150 record:=true" >&2
  exit 1
fi
lap="$(cd "$lap" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/local_setup.bash"
# 실차 세션(도메인 10)과 섞이지 않도록 별도 도메인 사용
export ROS_DOMAIN_ID="${REPLAY_DOMAIN_ID:-77}"
mapfile -t recorded < "$lap/launch_args.txt"
shopt -s nullglob
before=("$lap"/replay_*)
ros2 launch "$WS_DIR/autonomous_driving.launch.py" "${recorded[@]}" \
  enable_camera:=false enable_serial:=false enable_teleop:=false \
  use_sim_time:=true record:=false replay_bag:="$lap/bag" show_image:=false "$@" || true
after=("$lap"/replay_*)
if (( ${#after[@]} <= ${#before[@]} )); then
  echo "재생 결과가 저장되지 않았습니다." >&2
  exit 1
fi
replay="$(ls -td "$lap"/replay_* | head -n1)"
python3 "$WS_DIR/tools/compare_lap.py" "$lap" "$replay"
if [[ -f "$lap/labels.csv" ]] && grep -qE "^[0-9]+,[^,]*,-?[0-9]" "$lap/labels.csv"; then
  echo; echo "=== 라벨(정답)과 비교"
  python3 "$WS_DIR/tools/compare_labels.py" "$lap" "$replay"
fi
