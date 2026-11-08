#!/usr/bin/env bash
# 신호등 버전: 차선 주행 + 빨간불 정지. 다른 옵션은 run_autonomous.sh 와 동일하게 뒤에 붙이면 됨.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_autonomous.sh" obey_traffic_lights:=true "$@"
