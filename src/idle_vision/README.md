# idle_vision

Intel RealSense D435와 Logitech C270i 같은 USB RGB 카메라를 `idle_ws`에서 쓰기 위한 ROS2 패키지입니다.

기능은 크게 다섯 가지입니다.

- D435 또는 Logitech C270i 카메라 실행
- 학습용 RGB/depth 이미지 토픽 제공 및 저장
- HSV 색 범위 튜닝
- 색깔 박스의 `Color, x, y, z, yaw` 추출
- Whisper STT + Ollama/Qwen 자연어 명령으로 목표 박스 선택

## 빌드

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select idle_vision
source install/setup.bash
```

## 코드 구성

- `idle_vision/box_pose_node.py`: 여러 색 박스의 `color, x, y, z, yaw`를 publish하는 메인 node
- `idle_vision/color_segmentation.py`: 공유 HSV preset, B/G/R 비율 mask, custom HSV parser, contour 색 품질 검사
- `idle_vision/vision_utils.py`: timestamp/depth 변환 공통 helper
- `idle_vision/hsv_tuner_node.py`: 화면 ROI의 HSV 통계를 publish하는 튜닝용 node
- `idle_vision/image_learning_node.py`: 학습용 color/depth 이미지 republish 및 저장 node
- `idle_vision/object_depth_node.py`: 단일 색 물체 depth 확인용 node
- `idle_vision/qwen_box_selector_node.py`: Ollama/Qwen으로 자연어를 파싱하고 HSV 검출 박스 중 목표를 선택하는 node
- `idle_vision/voice_command_node.py`: 마이크 음성을 Whisper STT로 텍스트화해서 Qwen command topic에 publish하는 node

## 현재 C270i 포팅 상태

Logitech C270i 경로는 RealSense depth 없이 RGB만 쓰는 상태입니다.

- HSV 색 검출: 동작
- bbox/중심 픽셀/면적: 동작
- 이미지 평면 yaw: 동작
- Qwen/Whisper로 색상 명령을 받아 박스 선택: 동작
- `depth_m`: 없음, C270i가 depth 카메라가 아니라서 `null`
- `x_m/y_m/z_m`: homography 전에는 `null`

즉 지금 C270i로는 “색깔이랑 yaw까지 찾고, 어느 박스를 골랐는지”는 됩니다.
로봇 base 기준 `x/y` 좌표까지 쓰려면 바닥 평면 homography calibration을 추가로 잡아야 합니다.

## 수동 node 실행

일단 calibration과 토픽을 하나씩 보려면 launch보다 아래처럼 터미널을 나눠서 실행하는 게 편합니다.

모든 터미널에서 먼저:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

현재 추천 실행 순서:

터미널 1, D435 카메라:

```bash
ros2 launch idle_vision d435_camera.launch.py publish_tf:=false
```

터미널 2, 최종 base에서 본 `camera_color_optical_frame` static TF:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.067 --y 0.56 --z 0.939 \
  --yaw 3.141592 --pitch 0.0 --roll 3.141592 \
  --frame-id base \
  --child-frame-id camera_color_optical_frame
```

터미널 3, 박스 pose 추출:

```bash
ros2 run idle_vision box_pose_node --ros-args \
  -p target_color:=auto \
  -p base_frame:=base
```

터미널 4, 결과 확인:

```bash
ros2 topic echo --full-length /idle_vision/box_poses
```

`pose_frame_id`가 `base`로 나오면 base 기준 좌표 변환이 적용된 상태입니다.
`pose_frame_id`가 `camera_color_optical_frame`이면 calibration TF를 못 찾은 상태라 카메라 기준 좌표입니다.

카메라 optical frame 좌표는 맞는데 base 변환만 틀리면, calibration 값을 `link`가 아니라
`camera_color_optical_frame` 기준으로 알고 있는 경우일 수 있습니다. 이때는 RealSense 내부 TF를 끄고
optical frame을 base에 직접 붙여서 테스트합니다.

터미널 1, RealSense 내부 TF 끄고 카메라 실행:

```bash
ros2 launch idle_vision d435_camera.launch.py publish_tf:=false
```

터미널 2, base에서 본 `camera_color_optical_frame` 직접 TF:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.067 --y 0.56 --z 0.939 \
  --yaw 3.141592 --pitch 0.0 --roll 3.141592 \
  --frame-id base \
  --child-frame-id camera_color_optical_frame
```

축 부호가 반대면 먼저 `--yaw 0.0 --pitch 0.0 --roll 3.141592`도 비교합니다.

- `roll pi`만 쓰면 대략 `base_x = camera_x + 0.067`, `base_y = -camera_y + 0.56`, `base_z = -camera_z + 0.939`
- `yaw pi + roll pi`를 쓰면 대략 `base_x = -camera_x + 0.067`, `base_y = camera_y + 0.56`, `base_z = -camera_z + 0.939`

확인은:

```bash
ros2 run tf2_ros tf2_echo base camera_color_optical_frame
ros2 topic echo --full-length /idle_vision/box_poses
```

아래는 launch를 쓰지 않고 RealSense driver를 직접 실행하는 대체 방식입니다.

터미널 1, D435 카메라 node:

```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/camera \
  -r __node:=camera \
  -p device_type:=d435 \
  -p enable_color:=true \
  -p rgb_camera.color_profile:="640,480,30" \
  -p rgb_camera.color_format:=RGB8 \
  -p enable_depth:=true \
  -p depth_module.depth_profile:="640,480,30" \
  -p depth_module.depth_format:=Z16 \
  -p align_depth.enable:=true \
  -p enable_sync:=true \
  -p publish_tf:=false
```

터미널 2, 박스 인식 node:

```bash
ros2 run idle_vision box_pose_node --ros-args \
  -p target_color:=auto \
  -p base_frame:=base
```

최종 calibration TF를 직접 넣고 싶으면 별도 터미널에서:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.067 --y 0.56 --z 0.939 \
  --yaw 3.141592 --pitch 0.0 --roll 3.141592 \
  --frame-id base \
  --child-frame-id camera_color_optical_frame
```

`camera_color_optical_frame`을 직접 child로 쓰므로 RealSense 내부 TF는 꺼둡니다.

rqt는 보고 싶은 이미지마다 별도 터미널에서 따로 실행:

```bash
ros2 run rqt_image_view rqt_image_view /idle_vision/box_pose/debug_image
ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw
ros2 run rqt_image_view rqt_image_view /camera/camera/aligned_depth_to_color/image_raw
```

RViz도 따로 실행:

```bash
rviz2 -d ~/idle_ws/install/idle_vision/share/idle_vision/rviz/d435_box_pose.rviz
```

좌표/결과는 터미널에서:

```bash
ros2 topic echo --full-length /idle_vision/box_poses
```

`pose_frame_id`가 `base`로 나오면 base 기준 좌표 변환이 적용된 상태입니다.
`pose_frame_id`가 `camera_color_optical_frame`이면 calibration TF를 못 찾은 상태라 카메라 기준 좌표입니다.

HSV 값만 따로 보고 싶으면 node를 먼저 실행:

```bash
ros2 run idle_vision hsv_tuner_node
```

그리고 다른 터미널에서 확인:

```bash
ros2 run rqt_image_view rqt_image_view /idle_vision/hsv_tuner/debug_image
```

상태 JSON은 또 다른 터미널에서:

```bash
ros2 topic echo /idle_vision/hsv_tuner/status
```

## 색깔 박스 Pose 추출 - launch로 한번에 실행

여러 색 박스를 동시에 인식하고 각 박스의 색깔, 위치, yaw를 JSON으로 publish합니다.

```bash
ros2 launch idle_vision d435_box_pose.launch.py
```

rqt 이미지 창까지 같이 띄우려면:

```bash
ros2 launch idle_vision d435_box_pose_rqt.launch.py
```

이 launch는 기본으로 최종 calibration을 사용합니다.

- RealSense 내부 TF: 꺼짐, `camera_publish_tf:=false`
- static TF: 켜짐, `publish_camera_tf:=true`
- TF: `base -> camera_color_optical_frame`
- translation: `x=0.067, y=0.56, z=0.939`
- rotation: `yaw=3.141592, pitch=0.0, roll=3.141592`

rqt에서는 보통 세 창을 같이 봅니다.

- `/idle_vision/box_pose/debug_image`: color 위에 박스, frame 이름, 색 이름, x/y/z/yaw 표시
- `/camera/camera/color/image_raw`: 원본 color
- `/camera/camera/aligned_depth_to_color/image_raw`: aligned depth

RViz marker가 필요하면:

```bash
ros2 launch idle_vision d435_box_pose_rviz.launch.py
```

RViz launch도 기본으로 같은 최종 calibration을 사용합니다. 값을 다시 override하고 싶으면:

```bash
ros2 launch idle_vision d435_box_pose_rviz.launch.py \
  camera_publish_tf:=false \
  publish_camera_tf:=true \
  base_frame:=base \
  camera_frame:=camera_color_optical_frame \
  camera_x:=0.067 camera_y:=0.56 camera_z:=0.939 \
  camera_yaw:=3.141592 camera_pitch:=0.0 camera_roll:=3.141592
```

결과 확인:

```bash
ros2 topic echo --full-length /idle_vision/box_poses
```

출력 예시:

```json
{
  "count": 1,
  "pose_frame_id": "base",
  "boxes": [
    {
      "color": "Blue",
      "x_m": 0.12,
      "y_m": -0.04,
      "z_m": 0.63,
      "yaw_rad": 0.52,
      "yaw_deg": 30.0,
      "camera_x_m": 0.02,
      "camera_y_m": 0.11,
      "camera_z_m": 0.63
    }
  ]
}
```

좌표계 기준:

- calibration TF가 있으면 `x_m`, `y_m`, `z_m`, `yaw`는 `pose_frame_id` 기준입니다.
- 기본 `base_frame`은 `base`입니다.
- calibration TF가 없으면 `pose_frame_id`가 카메라 optical frame이고, 값도 카메라 기준입니다.
- 카메라 기준 원본 값은 항상 `camera_x_m`, `camera_y_m`, `camera_z_m`, `camera_yaw_deg`에 남습니다.

카메라를 base에 calibration해서 rqt로 실행, 기본값 사용:

```bash
ros2 launch idle_vision d435_box_pose_rqt.launch.py
```

여기서 `camera_x/y/z`는 base frame에서 본 카메라 원점 위치이고,
`camera_yaw/pitch/roll`은 base에서 camera frame으로 가는 회전입니다. 단위는 rad입니다.
`camera_frame:=camera_color_optical_frame`을 쓰는 경우에는 RealSense 내부 TF와 중복되지 않게
`camera_publish_tf:=false`를 같이 둡니다. `camera_frame:=link`처럼 RealSense root/body frame에
static TF를 걸 때는 `camera_publish_tf:=true`를 유지합니다.

RViz calibration을 맞추려면 다음 구조를 정해야 합니다.

- `base_frame` 이름: 예를 들어 `base`, `base_link`
- `camera_frame` 이름: 보통 `link` 또는 `camera_link`
- base 기준 카메라 원점 위치: `camera_x`, `camera_y`, `camera_z`, 단위 m
- base에서 camera frame으로 가는 회전: `camera_yaw`, `camera_pitch`, `camera_roll`, 단위 rad
- 로봇/table에서 x축, y축, z축이 어느 방향인지

출력 토픽:

- `/idle_vision/box_poses`: 감지된 박스 목록 JSON
- `/idle_vision/box_pose_array`: JSON 목록과 같은 순서의 `PoseArray`
- `/idle_vision/box_pose/debug_image`: 회전 박스, frame 이름, 색 이름, x/y/z/yaw 표시 이미지
- `/idle_vision/box_pose/mask`: 색 threshold를 통과한 픽셀만 흰색으로 보이는 이진 이미지
- `/idle_vision/box_pose/markers`: RViz용 색 점, yaw 화살표, x/y/z/yaw 라벨

## Logitech C270i USB 카메라로 박스 검출

C270i는 RGB 웹캠이라 depth가 없습니다. 그래서 기본 실행에서는 색, 중심 픽셀,
bbox, yaw, area를 publish하고, `x_m/y_m/z_m`은 `null`입니다.
나중에 바닥 평면 homography를 넣으면 같은 `/idle_vision/box_poses`에서
base 기준 `x_m/y_m/yaw`까지 채울 수 있습니다.

카메라 + 색 박스 검출 + rqt:

```bash
ros2 launch idle_vision logi_c270_box_pose_rqt.launch.py
```

기본 video device는 현재 직접 확인해서 열린 Logitech 장치입니다.

```text
/dev/video2
```

다른 장치로 바뀌면:

```bash
ros2 launch idle_vision logi_c270_box_pose_rqt.launch.py \
  video_device:=/dev/v4l/by-id/usb-046d_0825_D087B1E0-video-index0
```

이 launch는 내부적으로 다음과 같은 흐름입니다.

- `usb_cam`, C270i RGB 이미지 publish
- `box_pose_node`, `require_depth:=false`로 depth 없이 HSV 박스 검출
- rqt image view, `/image_raw`와 `/idle_vision/box_pose/debug_image` 확인

결과 확인:

```bash
ros2 topic echo --full-length /idle_vision/box_poses
```

Qwen/Whisper는 카메라와 분리된 launch를 그대로 쓰면 됩니다.

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py
```

평면 homography를 알게 되면 아래처럼 pixel-to-base 3x3 행렬을 넣습니다.
행렬은 `[u, v, 1] -> [x_m, y_m, 1]` 변환입니다.

```bash
ros2 launch idle_vision logi_c270_box_pose_rqt.launch.py \
  plane_frame:=base \
  plane_z_m:=0.0 \
  plane_homography_json:='[[h00,h01,h02],[h10,h11,h12],[h20,h21,h22]]'
```

homography가 들어가면 `/idle_vision/box_pose_array`,
`/idle_vision/box_pose/markers`, `/idle_vision/qwen/target_pose`도
좌표가 채워진 상태로 쓸 수 있습니다.

기본값 `target_color:=auto`는 `red`, `green`, `blue` 박스만 자동 탐색합니다.
현재 기본 색 검출은 다음 순서로 후보를 찾습니다.

1. HSV 범위로 색 픽셀을 1차 mask 처리
2. B/G/R 채널 비율로 빨강/초록/파랑 우세 조건 확인
3. D435에서는 aligned depth가 유효한 픽셀만 후보 mask에 유지
4. contour 전체 median HSV/BGR 비율로 최종 후보 검사

기본 HSV preset은 실제 샘플링한 빨강/초록/파랑 값 기준에서 조금 여유 있게 설정되어 있습니다.

- `red`: mask `H 0-12 또는 170-180, S 130-255, V 80-190`
- `green`: mask `H 28-58, S 90-230, V 65-150`
- `blue`: mask `H 85-112, S 130-255, V 110-190`

흰색/반사광이 낮은 채도에서 `blue`처럼 잡히는 것을 막기 위해 후보 contour의 median 색도 추가로 검사합니다.

- `red`: median HSV가 `H 0-18 또는 162-180, S 145-255, V 80-190`이고 R 비율이 충분할 때만 통과
- `green`: median HSV가 `H 28-58, S 100-230, V 65-150`이고 G 비율이 충분할 때만 통과
- `blue`: median HSV가 `H 85-112, S 150-255, V 115-190`이고 B 비율이 충분할 때만 통과

현재 샘플 로그 기준 검증 결과:

- 빈바닥: `137` frame 모두 `count 0`
- 빨강: `127` frame 모두 `Red` 1개
- 초록: `122` frame 모두 `Green` 1개
- 파랑: `177` frame 모두 `Blue` 1개

박스가 정사각형에 가까울 때 OpenCV 회전 사각형의 축이 가끔 90도 뒤집힐 수 있어서,
`box_pose_node`는 이전 frame의 같은 색/가까운 박스와 이어지는 방향으로 yaw를 안정화합니다.
관련 parameter는 `yaw_stabilize_aspect_ratio`, `yaw_track_max_px`입니다.

비교 실험을 위해 ratio/depth 후보 mask를 끌 수 있습니다.

```bash
ros2 run idle_vision box_pose_node --ros-args \
  -p use_color_ratio_mask:=false \
  -p use_depth_candidate_mask:=false
```

흰색/검정까지 포함하려면:

```bash
ros2 launch idle_vision d435_box_pose.launch.py target_color:=all
```

특정 색만 보려면:

```bash
ros2 launch idle_vision d435_box_pose.launch.py target_color:=blue
```

## Ollama/Qwen 자연어 선택

Qwen은 자연어 명령을 `color_key`, `spatial` 같은 JSON 기준으로 파싱만 합니다.
실제 좌표는 HSV/Depth 기반 `/idle_vision/box_poses`에서 선택합니다.

권장 구조는 카메라/좌표 추출 launch와 Qwen/Whisper launch를 분리해서 켜는 방식입니다.

터미널 1, 카메라 + HSV/depth 박스 좌표 추출 + rqt:

```bash
ros2 launch idle_vision d435_box_pose_rqt.launch.py
```

이 launch가 `/idle_vision/box_poses`를 publish합니다.

터미널 2, Ollama 서버:

```bash
ollama serve
```

모델이 아직 없으면 별도 터미널에서 한 번만 받습니다.

```bash
ollama pull qwen2.5:7b
```

터미널 3, Qwen selector + Whisper STT:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py
```

이 launch는 카메라를 켜지 않고 다음만 실행합니다.

- `qwen_box_selector_node`, 자연어 명령 파싱 및 목표 박스 선택
- `voice_command_node`, 마이크 -> Whisper STT -> Qwen command topic

Ollama model 이름이 다르면:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py ollama_model:=qwen2.5:7b
```

Ollama 없이 색/위치 단어 규칙만으로 테스트:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py use_ollama:=false
```

Ollama 서버도 ROS launch에서 같이 켜고 싶으면:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py start_ollama:=true
```

이미 `ollama serve`가 켜져 있으면 `start_ollama:=false` 그대로 둡니다.

명령 보내기:

```bash
ros2 topic pub --once /idle_vision/qwen/command std_msgs/msg/String \
  "{data: '파란 블럭 좌표 알려줘'}"
```

현재 마이크 환경에서는 음성 VAD 최소 임계값 기본값을 `300.0`으로 둡니다.
더 크게/작게 바꾸고 싶으면:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py \
  min_absolute_threshold:=300.0
```

마이크로 말한 문장은 `/idle_vision/voice/transcript`에 뜨고, 같은 문장이
`/idle_vision/qwen/command`로 들어가서 기존 Qwen selector가 목표 박스를 고릅니다.

음성만 따로 테스트:

```bash
ros2 launch idle_vision voice_command.launch.py
```

마이크 장치 번호 확인:

```bash
ros2 launch idle_vision voice_command.launch.py \
  print_audio_devices:=true list_devices_only:=true
```

특정 마이크를 쓰려면:

```bash
ros2 launch idle_vision qwen_voice_selector.launch.py \
  input_device:=2 min_absolute_threshold:=300.0
```

카메라/Qwen/Whisper를 전부 한 launch에서 켜는 기존 통합 launch도 남아있지만,
현장 테스트에서는 위처럼 분리 실행을 권장합니다.

Whisper 패키지가 없으면 Python 환경에 설치가 필요합니다.

```bash
pip install faster-whisper sounddevice
```

예시 명령:

- `빨간 블럭`
- `초록 블럭`
- `파란 블럭`
- `왼쪽 파란 블럭`
- `가장 가까운 초록 블럭`
- `가장 큰 빨간 박스`

출력 확인:

```bash
ros2 topic echo --full-length /idle_vision/qwen/parsed_command
ros2 topic echo --full-length /idle_vision/qwen/selected_box
ros2 topic echo /idle_vision/qwen/target_pose
ros2 topic echo /idle_vision/voice/transcript
```

출력 토픽:

- `/idle_vision/qwen/parsed_command`: Qwen/규칙 파싱 결과 JSON
- `/idle_vision/qwen/selected_box`: 선택된 박스와 `x_m, y_m, z_m, yaw_deg` JSON
- `/idle_vision/qwen/target_pose`: 선택된 박스의 `PoseStamped`
- `/idle_vision/qwen/status`: 선택 상태/에러 JSON
- `/idle_vision/voice/transcript`: Whisper STT가 알아들은 문장
- `/idle_vision/voice/status`: 마이크/VAD/STT 상태 JSON

`selected_box`의 `x_m`, `y_m`, `z_m`, `yaw_deg`는 `/idle_vision/box_poses`와 같은 기준입니다.
현재 최종 calibration이 적용되면 `pose_frame_id`가 `base`로 나옵니다.

## D435 카메라만 실행

```bash
ros2 launch idle_vision d435_camera.launch.py
```

기본 RealSense 토픽:

- `/camera/camera/color/image_raw`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`

## 카메라 + 학습용 이미지 브리지

```bash
ros2 launch idle_vision d435_learning.launch.py
```

학습용 출력 토픽:

- `/idle_vision/learning/color/image_raw`
- `/idle_vision/learning/depth/image_raw`
- `/idle_vision/learning/status`

학습 프레임 저장:

```bash
ros2 launch idle_vision d435_learning.launch.py \
  save_dir:=/home/eomyunbeen/idle_ws/datasets/d435_blocks \
  save_every_n:=10
```

이미 카메라 노드가 실행 중이면:

```bash
ros2 launch idle_vision d435_learning.launch.py start_camera:=false
```

## HSV 범위 튜닝

색상 threshold가 조명에 따라 달라질 수 있으므로, 물체를 화면 중앙 근처에 두고 실행합니다.

```bash
ros2 launch idle_vision hsv_tuner.launch.py
```

HSV 통계 확인:

```bash
ros2 topic echo /idle_vision/hsv_tuner/status
```

`status` JSON에는 다음 값이 들어갑니다.

- `sample_px`: 샘플링한 픽셀 좌표
- `roi_xyxy`: 샘플링 ROI 사각형
- `hsv.min`, `hsv.median`, `hsv.p05`, `hsv.p95`, `hsv.max`
- `suggested_hsv_ranges_json`: `hsv_ranges_json`에 넣어볼 시작 범위

디버그 토픽:

- `/idle_vision/hsv_tuner/debug_image`: 샘플 ROI 표시 이미지
- `/idle_vision/hsv_tuner/mask`: `hsv_ranges_json`을 넣었을 때 mask 미리보기

화면 중앙 대신 특정 픽셀을 샘플링:

```bash
ros2 launch idle_vision hsv_tuner.launch.py sample_u:=320 sample_v:=240
```

후보 HSV 범위를 mask로 미리보기:

```bash
ros2 launch idle_vision hsv_tuner.launch.py \
  hsv_ranges_json:='[[82,60,80,110,255,255]]'
```

## 단일 물체 Depth 추출

가장 크게 감지된 색 물체 하나의 depth와 중심점을 publish합니다.

```bash
ros2 launch idle_vision d435_object_depth.launch.py
```

기본값 `target_color:=auto`는 순수 `red`, `green`, `blue`를 자동 탐색한 뒤
가장 큰 물체 하나를 선택합니다.
선택된 색은 `/idle_vision/object_depth/bbox`의 `detected_color`로 확인합니다.

특정 색만 보려면:

```bash
ros2 launch idle_vision d435_object_depth.launch.py target_color:=red
```

지원 색상 preset:

`red`, `orange`, `yellow`, `green`, `blue`, `purple`, `white`, `black`

흰색/검정까지 자동 탐색하려면:

```bash
ros2 launch idle_vision d435_object_depth.launch.py target_color:=all
```

출력 토픽:

- `/idle_vision/object_depth/depth_m`: mask 내부 물체 depth, 단위 m
- `/idle_vision/object_depth/point`: 물체 중심 `PointStamped`
- `/idle_vision/object_depth/bbox`: bounding box와 depth 통계 JSON
- `/idle_vision/object_depth/mask`: 감지된 물체 mask
- `/idle_vision/object_depth/debug_image`: 감지 결과 overlay 이미지
- `/idle_vision/object_depth/status`: JSON status

확인 예시:

```bash
ros2 topic echo /idle_vision/object_depth/depth_m
ros2 topic echo /idle_vision/object_depth/point
```

커스텀 HSV threshold 사용:

```bash
ros2 launch idle_vision d435_object_depth.launch.py \
  hsv_ranges_json:='[[82,60,80,110,255,255]]'
```
