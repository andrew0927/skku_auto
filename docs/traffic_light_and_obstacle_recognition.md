# 신호등 및 장애물 인식

자율주행 중 **빨간불에서 횡단보도 앞에 서고**, **내 차선 앞에 차가 있으면 옆 차선으로 옮겨 계속 달리는** 기능을 정리한 문서입니다.
코드 위치는 `ros2_ws` 기준이며 줄 번호는 2026-11-19 기준입니다.

```bash
./run_traffic_light.sh 150 lane:=2 record:=true   # 2차선 출발, 신호등 + 장애물 회피(기본 켜짐), 기록
./run_traffic_light.sh 150 lane:=1                # 1차선 출발
./run_traffic_light.sh 150 avoid_obstacles:=false # 장애물 회피 끄기
```

## 1. 전체 흐름

```
카메라(C920, 10Hz)
 ├─ yolov8_node (best.pt seg: Trafficlight, crosswalk, lane1, lane2)
 │    ├─ lane_info_extractor_node : 차선 중심 경로, 내 차선(seg), 앞차의 차선 판정
 │    │    └─ path_planner_node → motion_planner_node
 │    └─ traffic_light_detector_node : 신호 색, 횡단보도 가까움/보임
 └─ vehicle_detector_node (HybridNets: 차량 박스 + 점선으로 본 차선)
          └─ lane_info_extractor_node
motion_planner_node : 신호등 정지/출발, 차선 변경 상태 기계, 비상 정지 → 키보드 노드 → 시리얼 → 아두이노
```

| 노드 | 파일 |
| --- | --- |
| 신호등·횡단보도 | [traffic_light_detector_node.py](../src/camera_perception_pkg/camera_perception_pkg/traffic_light_detector_node.py) |
| 신호 램프 직접 읽기 | [lamp_change.py](../src/camera_perception_pkg/camera_perception_pkg/lamp_change.py) |
| 차량 검출 | [vehicle_detector_node.py](../src/camera_perception_pkg/camera_perception_pkg/vehicle_detector_node.py), [hybridnets_vehicles.py](../src/camera_perception_pkg/camera_perception_pkg/hybridnets_vehicles.py) |
| 가운데 점선 | [center_line.py](../src/camera_perception_pkg/camera_perception_pkg/center_line.py) |
| 차선·내 차선·앞차 판정 | [lane_info_extractor_node.py](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py), [lane_geometry.py](../src/camera_perception_pkg/camera_perception_pkg/lane_geometry.py) |
| 판단(정지·차선 변경·비상 정지) | [motion_planner_node.py](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py) |
| 실행 인자 | [autonomous_driving.launch.py](../autonomous_driving.launch.py) |

---

## 2. 신호등 인식과 정지

### 2.1 신호 색 읽기 — `traffic_light_detector_node.py`
- best.pt가 찾은 신호등 박스 **안쪽** 픽셀만으로 램프 색(빨강/노랑(호박색)/초록)을 읽습니다. 박스 밖 초록 매트가 섞이지 않게 하기 위함입니다.
  → `read_lamps()` [lamp_change.py:34](../src/camera_perception_pkg/camera_perception_pkg/lamp_change.py#L34)
- 점수 `light_min_score`(0.5) 미만의 신호등 박스는 무시합니다.

### 2.2 YOLO가 신호등을 놓쳤을 때 — 램프 직접 찾기
횡단보도가 보이는데 신호등 박스가 없으면 화면 위쪽에서 켜진 빨간/노란 램프 덩어리를 직접 찾습니다.
→ `find_lamp_blob()` [lamp_change.py:46](../src/camera_perception_pkg/camera_perception_pkg/lamp_change.py#L46), 호출 [traffic_light_detector_node.py:103](../src/camera_perception_pkg/camera_perception_pkg/traffic_light_detector_node.py#L103)

| 조건 | 이유 |
| --- | --- |
| 가운데의 10% 이상이 하얗게 포화(V>245) | 켜진 LED는 카메라에서 포화됨(18~81%), 주황 벤치·기둥은 0% → 벤치를 노란불로 오인해 직선에서 서던 문제 해결 |
| 면적 1000px 이상 | 횡단보도 근처의 실제 램프는 4000~28000px, 자판기의 빨간 화면은 140~280px |
| 둥글기 0.55 이상, 단 **화면 끝에 닿은 덩어리는 0.25 이상** | 차가 꺾어 신호등이 화면 끝에 잘리면 둥글지 않음(11-17 110초, 11-18 21:09 46초에 빨간불을 놓친 원인) |

정지 중에는 신호등 위치의 픽셀 변화를 비교해, 신호등이 안 잡혀도 초록으로 바뀐 것을 알아챕니다 → `LampChangeDetector` [lamp_change.py:90](../src/camera_perception_pkg/camera_perception_pkg/lamp_change.py#L90).

### 2.3 횡단보도
- `crosswalk_ahead`: 횡단보도가 보이면 True → 플래너가 150으로 감속합니다.
- `crosswalk_near`: 횡단보도 박스 아래쪽이 정지선 `crosswalk_stop_y`(440)를 넘으면 True. **높이 60px(`crosswalk_min_height`) 미만은 제외**합니다. 화면 맨 아래 가로 흰 선을 얇은 횡단보도로 오인한 경우를 막기 위함입니다.
  → [traffic_light_detector_node.py:123-126](../src/camera_perception_pkg/camera_perception_pkg/traffic_light_detector_node.py#L123)

### 2.4 정지와 출발 — `motion_planner_node.py`
| 기능 | 내용 | 위치 |
| --- | --- | --- |
| 빨간불 확인 | 빨강/노랑이 **2번**(`red_confirm`) 보고돼야 빨간불로 봄(신호 없는 프레임은 세지 않음) | `light_callback` [:251](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L251), [:264](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L264) |
| 정지 | 빨간불 + `crosswalk_near` → 정지. 정지 직전 조향을 ±1로 기억 | `must_stop_for_red` [:504](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L504) |
| 출발 | 초록 **3번**(`green_confirm`) 연속, 또는 빨강이 `red_release_time`(3초) 동안 안 보이면 | [:266](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L266) |
| 건너기 | 출발 후 `crossing_time`(150 기준 2.5초) 동안 차선이 안 보여도 기억한 조향으로 직진 | `can_cross_crosswalk` [:467](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L467) |

---

## 3. 장애물(차량) 인식

### 3.1 차량 검출 — HybridNets
- best.pt에는 차 클래스가 없어서 차량은 별도 모델 HybridNets(공식 가중치를 TorchScript로 변환)로 찾습니다. GPU 약 26ms, CPU 약 150ms.
  → `HybridNetsVehicles.detect()` [hybridnets_vehicles.py:87](../src/camera_perception_pkg/camera_perception_pkg/hybridnets_vehicles.py#L87), 모델 `models/hybridnets_384x512/`, 변환 `tools/export_hybridnets_torchscript.py`
- 최근 5장 중 2장 이상 보여야 차로 인정하고, 화면 맨 아래의 납작한 박스(내 차 보닛)는 버립니다. → `keep_box()` [vehicle_detector_node.py:29](../src/camera_perception_pkg/camera_perception_pkg/vehicle_detector_node.py#L29)
- 박스마다 가운데 점선으로 본 차선(3.3)을 `Detection.id`에 넣어 보냅니다. → [vehicle_detector_node.py:84](../src/camera_perception_pkg/camera_perception_pkg/vehicle_detector_node.py#L84)

### 3.2 내 차선 인식 (지금 몇 차선인지)
여러 방법을 같은 화면 507장에서 비교한 결과(11-18 기록):

| 방법 | 판단한 비율 | 결과 |
| --- | --- | --- |
| **best.pt seg: 보닛 바로 위가 lane1/lane2 중 어느 마스크인지** | 77% | 기준. 다른 방법과 다를 때 모두 이것이 맞음 |
| 가운데 점선이 차의 왼쪽/오른쪽 | 50% | seg와 92% 일치. 주차구역에서 주차칸 선을 잘못 잡음 |
| HybridNets 도로 seg(도로의 왼쪽/오른쪽 절반) | 39% | seg와 62% 일치. 초록 매트까지 도로로 칠해 사용 안 함 |
| 출발 차선 + 차선 변경 기록 | 100% | seg와 90% 일치. 잘못된 차선 변경 뒤에 어긋남 |

그래서 다음 순서로 정합니다.
1. **seg**: 보닛 위(x 270~370, y 400~445)가 한 마스크로 50% 이상, 다른 마스크 20% 미만이면 그 차선. 최근 1초(10장) 중 70% 이상 같은 값이면 확정 → `lane_under_car()` [lane_info_extractor_node.py:149](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py#L149), `my_lane()` [:92](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py#L92), `/lane_under_car`로 발행
2. **seg가 판단 못하면**(차선 경계 위, 사람·차에 가림): 출발 차선(`lane:=`) + 차선 변경 기록
3. **둘이 1초(`lane_sync_time`) 이상 다르면** 차선 변경 중이 아닐 때 기록을 seg에 맞춤 → `sync_lane()` [motion_planner_node.py:288](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L291)

### 3.3 앞차가 어느 차선에 있는지 — 가운데 점선
→ [center_line.py](../src/camera_perception_pkg/camera_perception_pkg/center_line.py)
- 점선 찾기 `center_points()` [:14](../src/camera_perception_pkg/camera_perception_pkg/center_line.py#L14): 흰 페인트 중 **양옆이 모두 어두운 도로**인 것(가장자리 선은 한쪽이 매트·바닥).
- 직선 맞추기 `ransac_lines()` [:38](../src/camera_perception_pkg/camera_perception_pkg/center_line.py#L38).
- 판정 `car_side()` [:62](../src/camera_perception_pkg/camera_perception_pkg/center_line.py#L62): 앞차 박스 아래쪽 **바로 옆**(위 15px~아래 110px, 차 폭 1.5배 이내)의 점선만 사용 → 차가 점선 왼쪽이면 lane1, 오른쪽이면 lane2, 걸쳐 있으면 'line'. 멀리 있는 주차칸 선을 잡지 않도록 "바로 옆" 조건을 둠.

best.pt의 lane1/lane2 마스크로 앞차 차선을 정하지 않는 이유: 차가 마스크를 가리거나 감싸서 반대 차선 차를 내 차선 차로 판단한 사례가 있었음(11-18 79.5·89.6초).

### 3.4 "내 차선 앞에 차가 있다" 판정 — `vehicle_callback` [lane_info_extractor_node.py:101](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py#L101)
| 경우 | 판정 |
| --- | --- |
| 박스 아래쪽 y < 260(`vehicle_near_y`) | 멀어서 무시 |
| 점선 판정 = 내 차선(3.2) | 내 차선 차 |
| 점선 위에 걸침('line')·점선 없음 | 먼저 차 아래의 best.pt 차선 마스크로 차선 판정(11-19 225.8초: 화면 끝 lane1 차가 'line'으로 읽혀 그쪽으로 꺾음). 그래도 'line'이면 화면 가운데(x 200~440)에 40px 이상 걸칠 때만 내 차선 |
| 점선 판정 = 반대 차선 | 반대 차선 차 |
| 점선 없음 | y ≥ 330, 화면 정면(x=320을 가로지름), **그 아래가 도로(차선 마스크)**일 때만 내 차선 차. 매트 위에 세운 차 제외 |
| 가깝고(y ≥ 400) 화면 가운데(x 200~440)에 40px 미만 걸침 | **옆에서 지나가는 차** → 무시 [:130](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py#L130) |

결과는 `/camera_obstacle_info`(내 차선), `/camera_obstacle_other`(반대 차선), `/camera_obstacle_box`(내 차선의 가장 가까운 차와 화면의 가장 가까운 차 박스)로 발행합니다.

---

## 4. 차선 변경 (회피) — `motion_planner_node.py`

### 4.1 시작 조건 — `avoidance()` [:322](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L322)
- **확신할 때만 움직임**: 앞차가 내 차선으로 연속 3번(`avoid_confirm`, 약 0.3초) 판정돼야 차선 변경·후진을 함(사용자 11-19). 확신이 없으면(차선 모름, 다른 차선으로 읽힌 정면 차) 멈추기만 하고, 확신이 생기면 그때 출발.
- 시작 위치: 확신한 내 차선 차의 박스 아래쪽이 **y 315**(`avoid_start_y`, 약 1.8m)에 오면 **달리면서 바로 꺾음**(11-19 lap 210352 첫 장애물에서 후진이 끝난 위치 y 309~316: 사용자 "거기나 더 뒤에서 바로 꺾어도 됨, 길이 보이게"). **신호등을 지난 직후**(초록불에 건너는 중이거나 횡단보도를 지난 지 5초 안)에는 확신하자마자 거리와 관계없이 꺾음. 횡단보도 앞(정지선 전)·빨간불에서는 꺾지 않음(212007 8초: 정지선 전에 신호등 너머 4m+ 차를 보고 미리 꺾음).
- 확신한 내 차선 차면 화면 어디에 보이든 꺾음(커브에서는 내 차선 차가 화면 옆에 보임). '경로를 막을 때만 꺾는다'는 규칙은 213025 9~14초에 충돌을 일으켜 삭제.
- 내 차선은 상태(`avoid_lane`)로 판단하고, 세그멘테이션은 1초 이상 다를 때만 상태를 고침(212007 128.5초: 한 번 튄 lane2 판독으로 이미 있는 lane1로 '변경'). 새 차선 도착 판정도 5 step·s 이상 돈 뒤에만.
- 정지 위치: 내 차선 차가 y 315(`avoid_stop_y`, 약 1.8m; 212007 26초 사용자가 y 339에서 너무 가깝다고 세움)까지 왔는데 차선 변경이 시작되지 않았거나, 정면(박스 중심 x 220~420) 차의 차선이 확실하지 않으면 정지. 사용자가 Space로 세운 위치는 y 374(약 0.8m). 서 있다가 확신이 생기면 바퀴를 먼저 꺾고(STOP) 출발.
- 순서: OUT(새 차선 쪽으로 15 step·s) → HOLD(앞차가 옆으로 지나갈 때까지 도로 방향) → BACK, UNWIND(도로 방향 복귀). **후진은 확신한 차가 0.5m 안으로 들어왔을 때만**(4.5).
- 0.5m 안(y ≥ 380)의 경로 위 차를 검출기가 놓치면 그 박스를 최대 3초 유지(210352 49초: 가까워지자 검출이 사라져 충돌).
- 간격 모델: 간격 ≈ 538/(y − 169) − 1.82 (지평선 y≈169, 보닛 끝 y≈465가 간격 0, y 310이 2m).
- 모든 각도는 **도로 기준**: 시작 직전 1초의 차선 추종 조향(커브)을 기준(`base`)으로 더하고 뺍니다(커브에서 직선 기준으로 15를 더 꺾어 도로를 벗어난 80초·186초 장면). 간격 모델: 간격 ≈ 538/(y − 169) − 1.82 (지평선 y≈169). 간격 0은 우리 보닛 끝 y≈465(11-19 충돌 장면의 박스 아래쪽 468~474), 2m는 y 310(사용자 관찰). → 1m = y 360, 0.5m = y 400. (예전 값 440은 카메라 기준 1m라서 실제 간격은 거의 0이었고, 그래서 부딪혔음)
- 가까워져 차선을 못 읽게 된 앞차도 같은 박스(IoU ≥ 0.3, 0.5초 이내)면 계속 내 차선 차로 유지 [:54](../src/camera_perception_pkg/camera_perception_pkg/lane_info_extractor_node.py#L54). 11-19 두 번의 충돌은 y 380~450에서 차선을 못 읽어 '내 차선 차'에서 빠진 것이 원인.
- **시작하지 않는 경우** [:348](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L348): 빨간불을 본 직후·빨간불 대기 중(신호 대기 차는 정상이고, 꺾으면 신호등이 화면에서 가려짐) / 차선을 놓쳐 정지했거나 빨간불에 서 있을 때 / 차선 변경 직후 3초(같은 차 무시).
- 옮겨 갈 차선에도 가까운 차가 있으면 → 정지("obstacle in both lanes").

### 4.2 방향 결정 [:362](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L362)
- 지금 차선(3.2의 seg 우선, 없으면 기록)의 **반대 차선**으로. lane1은 왼쪽 차선이므로 **lane2 → 왼쪽(−), lane1 → 오른쪽(+)**.
- 결정한 차선은 `/target_lane`으로 차선 추출 노드에 보내 그 차선을 따라가게 합니다.

### 4.3 꺾고 돌아오기 — `maneuver_command()` [:398](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L398)
도로·차선이 안 보여도 동작하도록 계산된 동작이 기본이고, seg가 새 차선을 확인하면 그때 바로 넘깁니다.

| 단계 | 조향 | 끝나는 조건 |
| --- | --- | --- |
| 나가기(out) | 새 차선 쪽 최대 ±7(`avoid_steer`), 0.15초에 1단계씩(`avoid_steer_rate` 6.5/s) | 방향 변화량 ∫조향 dt가 **15 단계·초**(`avoid_heading`)에 도달(되돌리는 중 더 도는 양 포함). 단, seg가 아직 **옛 차선을 분명히** 보이면 계속(최대 25, `avoid_heading_max`, 사람 최대 23). 나가기가 4초(150 기준, `avoid_out_max`)를 넘으면 돌아오기로 |
| 유지(hold) | 0(직진, 방향 유지) | 회전량이 25에 도달했거나 나가기가 4초를 넘었는데 앞차가 아직 옆으로 비켜나지 않았을 때. 2초(150 기준) 넘게 안 비켜나면 정지 |
| 돌아오기(back) | 반대쪽 최대 ±7 | **앞차가 옆으로 비켜난 뒤에만 시작**(왼쪽으로 지날 때 앞차 왼쪽 끝 x ≥ 470, 오른쪽으로 지날 때 오른쪽 끝 x ≤ 170, 또는 0.5초 이상 안 보임 — 사람 run_04와 같음, 11-19 104초에 비켜나기 전에 되돌아 앞차 쪽으로 꺾은 문제 수정). 방향 변화량이 0으로 돌아오기 직전까지 |
| 풀기(unwind) | 0(직진)으로 | 조향이 0 근처 |
| 넘겨주기 | 새 차선 차선추종 | **seg가 새 차선을 보이면 어느 단계든 즉시** / 돌아오기·풀기가 끝나면(돌아오기는 항상 끝까지 수행해 방향 복원) / 안전 상한 12초 |

- 조향 1단계당 회전 속도 ≈ 1.4°/s(속도 150): 녹화 영상의 지면 움직임(버드아이 변환 + 특징점 정합)으로 측정. 회전이 조향에 비례하므로 방향 변화량 = ∫조향 dt.
- 15의 근거: run_04에서 사람이 새 차선에 도착(seg가 바뀐 시점)할 때까지 돌린 양이 6과 23. 장애물이 S자 곡선 위에 있어 "도로 기준 옆 이동량"(22 vs 54)은 일정하지 않아 쓰지 않음.
- 적분은 주행 거리 기준(시간 × 속도/150)이라 속도를 올려도 지면 위 궤적이 같습니다.
- run_04 실제 프레임으로 비교(코드 vs 사람): lane1→오른쪽 +7 유지 후 93.5초부터 되돌림(사람 94.1초), lane2→왼쪽 −7을 106초까지 유지 후 seg가 lane1을 보여 차선추종으로 넘김(사람 107초부터 풂). 이후 차선추종 조향 −5→−1→+2(사람 −7→−1→+2).

### 4.4 도로(차선)가 안 보일 때
| 상황 | 동작 | 위치 |
| --- | --- | --- |
| 차선 변경 중 차선·seg가 안 보임 | 4.3의 계산된 나가기→돌아오기로 진행 후 차선추종 | `maneuver_command` |
| 평소 주행 중 차선을 놓침(가로 흰 선, 반사, 앞차에 가림) | 직전 1초 차선추종 조향의 중앙값으로 150 속도로 1.5초(`lost_hold_time`, 150 기준) 더 간 뒤 정지 | [:575](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L575), `lost_limit` [:447](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L447) |
| 횡단보도 건너는 중 차선 없음 | 정지 직전 기억한 조향(±1)으로 직진 | [:559](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L559) |

### 4.5 절대 부딪히지 않기 — 정지·후진 [:736](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L736), 경로 판정 `car_in_path()` [:534](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L534), 후진 `backoff_command()` [:499](../src/decision_making_pkg/decision_making_pkg/motion_planner_node.py#L499)
- 내 차선 차가 1m(y 360)에 왔는데 차선 변경이 시작되지 않았으면(신호등 앞, 방금 차선 변경을 끝낸 직후 등) → 정지.
- 어떤 차든 0.5m(y ≥ 400, `front_stop_y`)에서 진행 경로(x 220~420)를 60px 이상 막으면 → 정지.
  - 내 차선 차 또는 차선 모름 → **후진 후 다시 틀기**: SETTLE(0.3초 정지, 바퀴를 새 차선 반대쪽으로 끝까지) → REVERSE(그 바퀴로 후진, 앞머리가 새 차선 쪽으로 돎; 앞차 박스가 y 345 위로 갈 때까지, 0.5~2초) → OUT(바퀴를 새 차선 쪽 끝까지 바로 꺾고 전진). 후진 중 돈 각도도 H에 더해져 OUT·BACK 계산에 들어감.
  - 차선 변경 중에도 0.5m에 들어오면 다시 후진. 한 차에 3번(`avoid_backoff_max`)을 넘으면 정지하고, 그 차가 경로에서 사라질 때까지 다시 후진하지 않음.
  - 다른 차선 차(커브에서 경로에 걸침) → 정지해서 기다림. 2초(`avoid_wait_max`) 넘게 안 비키면 후진.
- 후진 명령은 '마지막 명령 유지'로 다시 쓰이지 않음.
- 검증(11-18·11-19 녹화 20개 충돌·접촉 장면): 모두 가장 가까워지기 전에 차선 변경 시작 또는 정지·후진. 가까워지지 않은(0.5m 밖) 다른 차선·차선 모름 차 9개 장면은 동작 없음.
- 기록 확인: 걸린 장면은 모두 실제 충돌 직전, 사람 시범 run_04(앞차 옆으로 가깝게 지나감)에서는 한 번도 안 걸림.

---

## 5. 주요 파라미터 (launch 인자)

| 인자 | 기본 | 의미 |
| --- | --- | --- |
| `lane` | 2 | 출발 차선 |
| `avoid_obstacles` | true | 차선 변경 회피 |
| `avoid_start_y` | 315 | 확신한 내 차선 차가 이 행(약 1.8m)에 오면 달리면서 차선 변경 |
| `avoid_stop_y` | 315 | 변경이 시작 안 됐거나 차선이 불확실한 정면 차가 이 행(약 1.8m)에 오면 정지 |
| `avoid_confirm` | 3 | 내 차선 판정이 연속 이만큼 나와야 움직임 |
| `front_stop_y` | 400 | 경로의 차가 이 행(간격 약 0.5m)에 닿으면 정지·후진 |
| `avoid_steer` | 7 | 차선 변경 최대 조향 |
| `avoid_speed` | 150 | 차선 변경 중 속도 상한 |
| `crosswalk_stop_y` | 440 | 정지선 |
| `red_release_time` | 3.0 | 빨강이 안 보이면 출발하는 시간 |
| `light_min_score` | 0.5 | 신호등 박스 최소 점수 |
| `vehicle_backend` | hybridnets | 차량 검출 모델(yolo로 바꾸면 COCO yolov8n) |

노드 내부 값(`avoid_heading` 15, `avoid_heading_max` 25, `lane_sync_time` 1.0, `red_confirm` 2, `green_confirm` 3, `crosswalk_min_height` 60, `vehicle_near_y` 260 등)은 각 노드의 `declare_parameter`에 근거와 함께 적혀 있습니다.

---

## 6. 검증 방법

| 방법 | 명령/파일 |
| --- | --- |
| 단위 테스트(93개) | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests` — 차선 변경: `tests/test_lane_loss_control.py`, 점선·차량: `tests/test_vehicle_lanes.py`, 램프: `tests/test_lamp_change.py` |
| 기록 재생(전체 파이프라인) | `./replay_lap.sh laps/lap_...` → 원래 주행과 판단 비교(`comparison.csv`, `summary.txt`). 예전 튜닝값을 빼고 현재 기본값으로: `REPLAY_DROP='avoid_[a-z_]*' ./replay_lap.sh laps/lap_...` |
| 사람 시범과 비교 | `run_04_lane2_20261118_201039_098241/` (신호등 이후 구간) |

2026-11-19 기준 기록 16개 재생 결과:
- 빨간불 정지: 원래 주행과 같은 곳에서 정지. 원래 놓쳤던 11-17 110초, 11-18 21:09 48초도 정지. 벤치 오인(11-18 193427 33~70초)·신호등 없는 곳(11-13 192초) 정지는 사라짐.
- 노드 종료/오류 없음.
- 차선 변경 22건(현재 기본값): seg가 새 차선을 보여 넘김 9건, 계산된 나가기·돌아오기 완료 5건(남은 방향 −0.3~+0.3, 모두 원래 방향 복원), 비상 정지로 중단 5건(모두 녹화상 충돌 직전 장면 — 원래 주행이 앞차로 다가간 영상이라 재생에서도 다가감).
- 재생은 녹화 화면을 그대로 넣는 방식이라, 꺾었을 때 차가 실제로 움직인 결과는 실제 주행 기록으로 확인해야 합니다.

## 7. 알려진 한계
- 차선 변경의 방향 변화량 15는 사람 시범 2회(6, 23)와 성공 사례에서 정한 값입니다. seg가 판단하는 구간에서는 seg가 끝을 정하므로 영향이 작지만, seg가 계속 안 보이는 곡선에서는 덜 가거나 더 갈 수 있습니다. 실제 주행 기록이 쌓이면 다시 계산합니다.
- 점선이 안 보이는 장면(약 절반)에서는 "가깝고 정면 + 도로 위" 차만 내 차선으로 봐서 판단이 늦을 수 있습니다.
- 차선 변경 직후 3초는 같은 차를 무시합니다(비상 정지는 동작).
