// CAN <-> ROS bridge node for MIT protocol (Type02 state/error, Type01 command relay).
#include <errno.h>
#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "msgs/msg/motor_cmd.hpp"
#include "msgs/msg/motor_cmd_array.hpp"
#include "msgs/msg/motor_error.hpp"
#include "msgs/msg/motor_error_array.hpp"
#include "msgs/msg/motor_state.hpp"
#include "msgs/msg/motor_state_array.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{
// MIT 확장 ID의 통신 타입 정의
constexpr uint8_t TYPE00_GET_DEVICE_ID = 0x00;
constexpr uint8_t TYPE01_OPERATION_CONTROL = 0x01;
constexpr uint8_t TYPE02_FEEDBACK = 0x02;
constexpr uint8_t TYPE03_ENABLE = 0x03;
constexpr uint8_t TYPE04_STOP = 0x04;
constexpr uint8_t TYPE_MASK = 0x1F;
// Home 복귀 궤적 관련 기본값
constexpr float kHomeReturnDurationSec = 7.0F;
constexpr float kHomeEpsilon = 1e-4F;
// 이 범위 이내면 "이미 홈 포즈"로 간주 — 궤적/j2j3 시퀀스 건너뜀.
constexpr float kHomeNearThreshold = 0.0873F;  // ≈ 5도
// 모터 온도 임계값 (°C)
constexpr float kTempWarnDegC     = 70.0F;   // WARN 로그
constexpr float kTempCriticalDegC = 90.0F;   // safe home 트리거

// 정적 브리지 설정(수정 후 재빌드/재시작 필요)
constexpr char kChannel[] = "can0";
constexpr uint8_t kHostId = 0xFD;
constexpr uint8_t kRs03Id  = 2;  // 어깨(j2) — RS03 모터
constexpr uint8_t kRs00Id1 = 5;
constexpr uint8_t kRs00Id2 = 6;
constexpr uint8_t kRs05Id  = 7;
constexpr uint8_t kJ3MotorId = 3;  // 엘보우(j3) — ±π 경계 wrapping 특수 처리 대상
constexpr uint8_t kJ2MotorId = 2;
constexpr float kJ3SafeHomeRad  = -0.785F;  // -45°: link2_3/joint1 충돌 없는 안전 구간
constexpr float kJ2SafeFloorRad = -0.524F;  // -30°: j3=-45° 일 때 바닥 간섭 없는 최소 j2
constexpr float kHomeSeqSpeedRadPerSec = 0.5F;
constexpr float kHomeSeqMinDurationSec = 0.5F;
constexpr float kHomeSeqDwellSec       = 1.0F;  // 페이즈 완료 후 안착 대기
constexpr int kScanMinId = 1;
constexpr int kScanMaxId = 10;
constexpr int kScanWaitMs = 200;
constexpr int kRxMaxFramesPerTick = 100;
constexpr int kCmdTimeoutMs = 100;
constexpr double kTxHzDefault = 250.0;
const std::unordered_map<int, double> kTxHzByMotor {};
// q_des 슬루레이트 한계 (rad/s): 한 틱 안에 이 속도를 초과하는 q_des 점프를 클램프.
// 홈 시퀀스·timeout-home·외부 명령 모두 적용된다.
const std::unordered_map<int, double> kMaxQdotCmdByMotor {
  {1, 1.5},  // RS02, j1
  {2, 1.0},  // RS03, j2 어깨 — 가장 보수적
  {3, 1.5},  // RS02, j3
  {4, 1.5},  // RS02, j4
  {5, 2.0},  // RS00, j5
  {6, 2.0},  // RS00, j6
  {7, 3.0},  // RS05, 그리퍼
};
// 외부 명령 tau_ff 클램프 한계 (Nm, 대칭).
// plan_node/hold_node와 동일한 DEFAULT_TAU_LIMIT_BY_MOTOR 값.
const std::unordered_map<int, float> kTauLimitByMotor {
  {1, 6.0F},
  {2, 25.0F},
  {3, 10.0F},
  {4, 6.0F},
  {5, 5.0F},
  {6, 5.0F},
  {7, 1.6F},
};
// tau_ff 슬루레이트 한계 (Nm/s): 홈(tau_ff=0)에서 중력 보상값으로 전환 시 급격한 토크 점프를 방지.
const std::unordered_map<int, float> kMaxTauRateByMotor {
  {1, 10.0F},
  {2, 20.0F},
  {3, 15.0F},
  {4, 10.0F},
  {5,  8.0F},
  {6,  8.0F},
  {7,  3.0F},
};
constexpr double kHomeQDes = 0.0;
constexpr double kHomeQdDes = 0.0;
constexpr double kHomeKp = 5.0;
constexpr double kHomeKd = 1.0;
constexpr double kHomeTauFf = 0.0;
struct MotorHomeConfig
{
  double q_des;
  double kp;
  double kd;
  double q_min;
  double q_max;
};

const std::unordered_map<int, MotorHomeConfig> kMotorHomeByMotor {
  {1, {0.0,                        20.0, 1.0, -static_cast<double>(M_PI), static_cast<double>(M_PI)}},
  {2, {0.0,                        70.0, 5.0, -1.78,                     1.78}},
  {3, {0.0,                        20.0, 1.0, -static_cast<double>(M_PI), static_cast<double>(M_PI)}},
  {4, {0.0,                        20.0, 1.0, -static_cast<double>(M_PI), static_cast<double>(M_PI)}},
  {5, {0.0,                        5.0, 0.5, -static_cast<double>(M_PI), static_cast<double>(M_PI)}},
  {6, {0.0,                        5.0, 0.5, -static_cast<double>(M_PI), static_cast<double>(M_PI)}},
  {7, {0.0,                         2.0, 0.5, 0.0,                       1.3599}},
};

// MIT 인코딩/디코딩에 쓰는 모터별 물리 범위
struct MitRanges
{
  float p_min;
  float p_max;
  float v_min;
  float v_max;
  float t_min; 
  float t_max;
  float kp_min;
  float kp_max;
  float kd_min;
  float kd_max;
};

const MitRanges MIT_RS02{
  -4.0F * static_cast<float>(M_PI), 4.0F * static_cast<float>(M_PI),
  -44.0F, 44.0F,
  -17.0F, 17.0F,
  0.0F, 500.0F,
  0.0F, 5.0F,
};

const MitRanges MIT_RS03{
  -4.0F * static_cast<float>(M_PI), 4.0F * static_cast<float>(M_PI),
  -20.0F, 20.0F,
  -60.0F, 60.0F,
  0.0F, 5000.0F,
  0.0F, 100.0F,
};

// RS00: rated 5 Nm, peak 14 Nm, 315 rpm no-load, 10:1 reduction (motors 5, 6)
const MitRanges MIT_RS00{
  -4.0F * static_cast<float>(M_PI), 4.0F * static_cast<float>(M_PI),
  -33.0F, 33.0F,
  -14.0F, 14.0F,
  0.0F, 500.0F,
  0.0F, 5.0F,
};

// RS05: rated 1.6 Nm, peak 5.5 Nm, 480 rpm no-load, 7.75:1 reduction (motor 7 / gripper)
const MitRanges MIT_RS05{
  -4.0F * static_cast<float>(M_PI), 4.0F * static_cast<float>(M_PI),
  -50.0F, 50.0F,
  -5.5F, 5.5F,
  0.0F, 500.0F,
  0.0F, 5.0F,
};

// (type, data2, data1)을 MIT 확장 arbitration ID로 패킹한다.
inline uint32_t pack_ext_id(uint8_t comm_type, uint16_t data2, uint8_t data1)
{
  return (static_cast<uint32_t>(comm_type & TYPE_MASK) << 24) |
         (static_cast<uint32_t>(data2) << 8) |
         static_cast<uint32_t>(data1);
}

// 확장 ID에서 통신 타입을 추출한다.
inline uint8_t comm_type_from_arb(uint32_t arb_id)
{
  return static_cast<uint8_t>((arb_id >> 24) & TYPE_MASK);
}

// 확장 ID에서 data2 필드를 추출한다.
inline uint16_t data2_from_arb(uint32_t arb_id)
{
  return static_cast<uint16_t>((arb_id >> 8) & 0xFFFFU);
}

// 확장 ID에서 data1 필드를 추출한다.
inline uint8_t data1_from_arb(uint32_t arb_id)
{
  return static_cast<uint8_t>(arb_id & 0xFFU);
}

// 주기성(2pi) 후보 중 현재 값에 가장 가까운 목표를 선택한다.
inline float nearest_periodic_target(
  const float target_base,
  const float current,
  const float period,
  const float lo,
  const float hi)
{
  if (period <= 0.0F) {
    return std::clamp(target_base, lo, hi);
  }
  float target = target_base + std::round((current - target_base) / period) * period;
  while (target < lo) {
    target += period;
  }
  while (target > hi) {
    target -= period;
  }
  return std::clamp(target, lo, hi);
}

// 실수값을 [xmin, xmax] 범위의 u16로 인코딩한다.
inline uint16_t float_to_u16(float x, float xmin, float xmax)
{
  const float clamped = std::clamp(x, xmin, xmax);
  const float scaled = (clamped - xmin) * 65535.0F / (xmax - xmin);
  const float rounded = std::round(scaled);
  return static_cast<uint16_t>(rounded);
}

// u16 값을 [xmin, xmax] 실수 범위로 디코딩한다.
inline float u16_to_float(uint16_t u, float xmin, float xmax)
{
  const float ratio = static_cast<float>(u) / 65535.0F;
  return xmin + ratio * (xmax - xmin);
}

// big-endian 2바이트를 u16로 조합한다.
inline uint16_t be_u16(const uint8_t hi, const uint8_t lo)
{
  return static_cast<uint16_t>((static_cast<uint16_t>(hi) << 8) | static_cast<uint16_t>(lo));
}

// ROS Time을 builtin_interfaces::msg::Time으로 변환한다.
inline builtin_interfaces::msg::Time to_builtin_time(const rclcpp::Time & t)
{
  constexpr int64_t kNsecPerSec = 1000000000LL;
  const int64_t ns = t.nanoseconds();
  int64_t sec = ns / kNsecPerSec;
  int64_t nsec = ns % kNsecPerSec;
  if (nsec < 0) {
    --sec;
    nsec += kNsecPerSec;
  }

  builtin_interfaces::msg::Time out {};
  out.sec = static_cast<int32_t>(sec);
  out.nanosec = static_cast<uint32_t>(nsec);
  return out;
}
}  // namespace

class CanBridgeNode : public rclcpp::Node
{
public:
  // 노드 초기화: 정적 정책 적용, CAN 오픈, 토픽/타이머 설정
  CanBridgeNode()
  : Node("can_bridge_node")
  {
    channel_ = kChannel;
    host_id_ = kHostId;
    scan_min_id_ = kScanMinId;
    scan_max_id_ = kScanMaxId;
    scan_wait_ms_ = kScanWaitMs;
    rx_max_frames_per_tick_ = kRxMaxFramesPerTick;
    cmd_timeout_ms_ = std::max(0, kCmdTimeoutMs);

    apply_tx_policy(kTxHzDefault, kTxHzByMotor);

    std::unordered_map<int, double> q_des_by_motor;
    std::unordered_map<int, double> kp_by_motor;
    std::unordered_map<int, double> kd_by_motor;
    std::unordered_map<int, double> q_limit_min_by_motor;
    std::unordered_map<int, double> q_limit_max_by_motor;
    for (const auto & [motor_id, cfg] : kMotorHomeByMotor) {
      q_des_by_motor[motor_id] = cfg.q_des;
      kp_by_motor[motor_id] = cfg.kp;
      kd_by_motor[motor_id] = cfg.kd;
      q_limit_min_by_motor[motor_id] = cfg.q_min;
      q_limit_max_by_motor[motor_id] = cfg.q_max;
    }
    apply_home_policy(
      kHomeQDes,
      kHomeQdDes,
      kHomeKp,
      kHomeKd,
      kHomeTauFf,
      q_des_by_motor,
      kp_by_motor,
      kd_by_motor,
      q_limit_min_by_motor,
      q_limit_max_by_motor);

    if (!open_socketcan(channel_)) {
      throw std::runtime_error("failed to open socketcan channel: " + channel_);
    }

    const auto qos_cmd = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    const auto qos_state = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    const auto qos_reliable = rclcpp::QoS(rclcpp::KeepLast(20)).reliable();

    state_array_pub_ =
      create_publisher<msgs::msg::MotorStateArray>("/motor_state_array", qos_state);
    error_array_pub_ = create_publisher<msgs::msg::MotorErrorArray>(
      "/motor_error_array",
      qos_reliable);
    applied_cmd_array_pub_ =
      create_publisher<msgs::msg::MotorCMDArray>("/motor_cmd_array_applied", qos_cmd);
    cmd_array_sub_ = create_subscription<msgs::msg::MotorCMDArray>(
      "/motor_cmd_array",
      qos_cmd,
      std::bind(&CanBridgeNode::on_cmd_array, this, std::placeholders::_1));

    scan_motor_ids();
    send_enable_once_from_scan();
    RCLCPP_INFO(
      get_logger(),
      "driver parameter write(Type18/Type22) is disabled in can_bridge_node; use maintenance tools only");
    RCLCPP_INFO(
      get_logger(),
      "runtime parameter updates are disabled; tune static constants in can_bridge_node.cpp and restart");
    RCLCPP_INFO(
      get_logger(),
      "tx policy initialized: tx_hz_default=%.3f per_motor=%zu",
      tx_policy_.tx_hz_default,
      tx_policy_.tx_hz_by_motor.size());
    RCLCPP_INFO(
      get_logger(),
      "home policy initialized: default(q_des=%.3f qd_des=%.3f kp=%.3f kd=%.3f tau_ff=%.3f), "
      "per_motor(q_des=%zu kp=%zu kd=%zu qmin=%zu qmax=%zu)",
      home_policy_.q_des_default,
      home_policy_.qd_des_default,
      home_policy_.kp_default,
      home_policy_.kd_default,
      home_policy_.tau_ff_default,
      home_policy_.q_des_by_motor.size(),
      home_policy_.kp_by_motor.size(),
      home_policy_.kd_by_motor.size(),
      home_policy_.q_limit_min_by_motor.size(),
      home_policy_.q_limit_max_by_motor.size());

    io_timer_ = create_wall_timer(
      std::chrono::milliseconds(1),
      std::bind(&CanBridgeNode::poll_can_frames, this));
  }

  // 종료 시 알려진 모터에 STOP을 1회 전송하고 소켓을 닫는다.
  ~CanBridgeNode() override
  {
    send_stop_once_on_shutdown();
    if (sock_fd_ >= 0) {
      close(sock_fd_);
      sock_fd_ = -1;
    }
  }

private:
  // 모터 ID에 따라 모터 모델별 MIT 범위를 선택한다.
  const MitRanges & ranges_for(const uint8_t motor_id) const
  {
    if (motor_id == kRs03Id)                           return MIT_RS03;
    if (motor_id == kRs00Id1 || motor_id == kRs00Id2) return MIT_RS00;
    if (motor_id == kRs05Id)                           return MIT_RS05;
    return MIT_RS02;
  }

  // SocketCAN 채널을 열고 non-blocking으로 설정한다.
  bool open_socketcan(const std::string & channel)
  {
    sock_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock_fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "socket() failed: %s", std::strerror(errno));
      return false;
    }

    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, channel.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock_fd_, SIOCGIFINDEX, &ifr) < 0) {
      RCLCPP_ERROR(
        get_logger(), "ioctl(SIOCGIFINDEX) failed for %s: %s",
        channel.c_str(), std::strerror(errno));
      return false;
    }

    struct sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      RCLCPP_ERROR(get_logger(), "bind() failed for %s: %s", channel.c_str(), std::strerror(errno));
      return false;
    }

    const int flags = fcntl(sock_fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(sock_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
      RCLCPP_ERROR(get_logger(), "fcntl(O_NONBLOCK) failed: %s", std::strerror(errno));
      return false;
    }
    return true;
  }

  // MIT 확장 프레임 1개를 전송한다.
  bool send_ext_frame(
    const uint32_t arb_id, const std::array<uint8_t, 8> & data,
    const uint8_t dlc = 8)
  {
    if (sock_fd_ < 0) {
      return false;
    }
    struct can_frame frame {};
    frame.can_id = (arb_id & CAN_EFF_MASK) | CAN_EFF_FLAG;
    frame.can_dlc = dlc;
    std::memcpy(frame.data, data.data(), dlc);
    const ssize_t n = write(sock_fd_, &frame, sizeof(frame));
    return n == static_cast<ssize_t>(sizeof(frame));
  }

  // CAN 프레임 1개를 non-blocking으로 읽어 온다.
  std::optional<struct can_frame> recv_frame()
  {
    struct can_frame frame {};
    const ssize_t n = read(sock_fd_, &frame, sizeof(frame));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return std::nullopt;
      }
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "read() failed: %s", std::strerror(errno));
      return std::nullopt;
    }
    if (n != static_cast<ssize_t>(sizeof(frame))) {
      return std::nullopt;
    }
    return frame;
  }

  // 모터별 최신 명령/전송 시각/홈 복귀 상태 캐시
  struct CachedCommand
  {
    msgs::msg::MotorCMD cmd {};
    bool valid {false};
    bool has_sent {false};
    bool has_received_cmd {false};
    bool timeout_home_active {false};
    bool home_traj_active {false};
    float home_traj_start_q {0.0F};
    float home_traj_goal_q {0.0F};
    float home_traj_duration_sec {kHomeReturnDurationSec};
    std::chrono::steady_clock::time_point home_traj_start_tp {};
    std::chrono::steady_clock::time_point last_sent {};
    std::chrono::steady_clock::time_point last_received_cmd {};
    // q_des 슬루레이트: 직전 전송된 q_des (실제 모터 위치로 초기화)
    bool slew_initialized {false};
    float last_sent_q_des {0.0F};
    // tau_ff 슬루레이트: 홈(0 Nm)에서 시작해 중력 보상값으로 천천히 증가
    bool tau_slew_initialized {false};
    float last_sent_tau_ff {0.0F};
  };

  // 모터별 송신 주기 정책
  struct TxPolicy
  {
    double tx_hz_default {500.0};
    std::unordered_map<int, double> tx_hz_by_motor {};
  };

  // 홈 복귀와 제한값 관련 정책
  struct HomePolicy
  {
    double q_des_default {0.0};
    double qd_des_default {0.0};
    double kp_default {30.0};
    double kd_default {5.0};
    double tau_ff_default {0.0};
    std::unordered_map<int, double> q_des_by_motor {};
    std::unordered_map<int, double> kp_by_motor {};
    std::unordered_map<int, double> kd_by_motor {};
    std::unordered_map<int, double> q_limit_min_by_motor {};
    std::unordered_map<int, double> q_limit_max_by_motor {};
  };

  // 모터별 ready/state 런타임 상태
  struct MotorRuntime
  {
    bool enable_sent {false};
    bool ready_for_control {false};  // Type03 was sent and Type02 feedback observed afterwards.
    bool has_state {false};
    // 시작 시 실제 위치가 joint limit 밖이면 true: 명령 전송 차단.
    // 수동 복구 후 can_bridge 재시작 필요.
    bool startup_blocked {false};
    // 온도 임계 초과 시 true → safe home 트리거. 온도 정상 복귀 시 자동 해제.
    bool safe_shutdown_thermal {false};
    // fault_bits raised 시 true → safe home 트리거. fault cleared 시 자동 해제.
    bool safe_shutdown_fault {false};
    msgs::msg::MotorState last_state {};
    std::chrono::steady_clock::time_point last_enable_sent {};
  };

  // tx 정책을 교체 적용한다.
  void apply_tx_policy(
    const double tx_hz_default,
    const std::unordered_map<int, double> & tx_hz_by_motor)
  {
    tx_policy_.tx_hz_default = tx_hz_default;
    tx_policy_.tx_hz_by_motor = tx_hz_by_motor;
  }

  // 홈 정책과 모터별 제한값을 교체 적용한다.
  void apply_home_policy(
    const double q_des_default,
    const double qd_des_default,
    const double kp_default,
    const double kd_default,
    const double tau_ff_default,
    const std::unordered_map<int, double> & q_des_by_motor,
    const std::unordered_map<int, double> & kp_by_motor,
    const std::unordered_map<int, double> & kd_by_motor,
    const std::unordered_map<int, double> & q_limit_min_by_motor,
    const std::unordered_map<int, double> & q_limit_max_by_motor)
  {
    home_policy_.q_des_default = q_des_default;
    home_policy_.qd_des_default = qd_des_default;
    home_policy_.kp_default = std::max(0.0, kp_default);
    home_policy_.kd_default = std::max(0.0, kd_default);
    home_policy_.tau_ff_default = tau_ff_default;
    home_policy_.q_des_by_motor = q_des_by_motor;
    home_policy_.kp_by_motor = kp_by_motor;
    home_policy_.kd_by_motor = kd_by_motor;
    home_policy_.q_limit_min_by_motor = q_limit_min_by_motor;
    home_policy_.q_limit_max_by_motor = q_limit_max_by_motor;

    for (auto & kv : home_policy_.kp_by_motor) {
      if (kv.second < 0.0) {
        RCLCPP_WARN(
          get_logger(),
          "home_kp_by_motor[%d]=%.3f is negative; clamped to 0.0",
          kv.first,
          kv.second);
        kv.second = 0.0;
      }
    }
    for (auto & kv : home_policy_.kd_by_motor) {
      if (kv.second < 0.0) {
        RCLCPP_WARN(
          get_logger(),
          "home_kd_by_motor[%d]=%.3f is negative; clamped to 0.0",
          kv.first,
          kv.second);
        kv.second = 0.0;
      }
    }

  }

  // 모터별 override가 있으면 사용하고 없으면 기본값을 사용한다.
  double value_for_motor(
    const std::unordered_map<int, double> & by_motor,
    const uint8_t motor_id,
    const double default_value) const
  {
    const auto it = by_motor.find(static_cast<int>(motor_id));
    if (it == by_motor.end()) {
      return default_value;
    }
    return it->second;
  }

  // 모터별 위치 제한(min/max)을 계산한다.
  std::pair<float, float> position_limits_for_motor(const uint8_t motor_id) const
  {
    const MitRanges & r = ranges_for(motor_id);
    const double qmin_cfg = value_for_motor(
      home_policy_.q_limit_min_by_motor, motor_id, static_cast<double>(r.p_min));
    const double qmax_cfg = value_for_motor(
      home_policy_.q_limit_max_by_motor, motor_id, static_cast<double>(r.p_max));

    float q_min = static_cast<float>(std::max(static_cast<double>(r.p_min), qmin_cfg));
    float q_max = static_cast<float>(std::min(static_cast<double>(r.p_max), qmax_cfg));
    if (q_min > q_max) {
      RCLCPP_WARN_ONCE(
        get_logger(),
        "motor_id=%u invalid q-limit config (min=%.3f max=%.3f): using MIT range [%.3f, %.3f]",
        motor_id,
        qmin_cfg,
        qmax_cfg,
        static_cast<double>(r.p_min),
        static_cast<double>(r.p_max));
      q_min = r.p_min;
      q_max = r.p_max;
    }
    return {q_min, q_max};
  }

  // 홈 복귀 시작점으로 사용할 현재 위치를 제한 범위 안으로 맞춘다.
  float wrap_or_clamp_current_q_for_home(
    const float current_q,
    const float q_min,
    const float q_max) const
  {
    // State is treated as absolute in configured limits. Do not re-wrap here;
    // just clamp to the allowed home window.
    return std::clamp(current_q, q_min, q_max);
  }

  // 모터별 명령 송신 주파수를 반환한다.
  double tx_hz_for_motor(const uint8_t motor_id) const
  {
    const auto it = tx_policy_.tx_hz_by_motor.find(static_cast<int>(motor_id));
    if (it != tx_policy_.tx_hz_by_motor.end() && it->second > 0.0) {
      return it->second;
    }
    return (tx_policy_.tx_hz_default > 0.0) ? tx_policy_.tx_hz_default : kTxHzDefault;
  }

  // 모터별 q_des 슬루레이트 한계 (rad/s)를 반환한다.
  double max_qdot_for_motor(const uint8_t motor_id) const
  {
    const auto it = kMaxQdotCmdByMotor.find(static_cast<int>(motor_id));
    return (it != kMaxQdotCmdByMotor.end()) ? it->second : 0.0;
  }

  // MotorCMD를 MIT Type01 프레임으로 인코딩해 전송한다.
  bool send_mit_command(const msgs::msg::MotorCMD & cmd)
  {
    const uint8_t motor_id = cmd.motor_id;
    const MitRanges & r = ranges_for(motor_id);
    const auto limits = position_limits_for_motor(motor_id);
    const float q_des_in_window = std::clamp(cmd.q_des, limits.first, limits.second);
    if (std::fabs(q_des_in_window - cmd.q_des) > 1e-6F) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "motor_id=%u q_des clamp applied: %.3f -> %.3f (limit=[%.3f, %.3f])",
        motor_id,
        static_cast<double>(cmd.q_des),
        static_cast<double>(q_des_in_window),
        static_cast<double>(limits.first),
        static_cast<double>(limits.second));
    }

    const float q_des_for_encode = q_des_in_window;

    const uint16_t t_u16 = float_to_u16(cmd.tau_ff, r.t_min, r.t_max);
    const uint16_t q_u16 = float_to_u16(q_des_for_encode, r.p_min, r.p_max);
    const uint16_t qd_u16 = float_to_u16(cmd.qd_des, r.v_min, r.v_max);
    const uint16_t kp_u16 = float_to_u16(cmd.kp, r.kp_min, r.kp_max);
    const uint16_t kd_u16 = float_to_u16(cmd.kd, r.kd_min, r.kd_max);

    const uint32_t arb = pack_ext_id(TYPE01_OPERATION_CONTROL, t_u16, motor_id);
    std::array<uint8_t, 8> data {};
    data[0] = static_cast<uint8_t>((q_u16 >> 8) & 0xFF);
    data[1] = static_cast<uint8_t>(q_u16 & 0xFF);
    data[2] = static_cast<uint8_t>((qd_u16 >> 8) & 0xFF);
    data[3] = static_cast<uint8_t>(qd_u16 & 0xFF);
    data[4] = static_cast<uint8_t>((kp_u16 >> 8) & 0xFF);
    data[5] = static_cast<uint8_t>(kp_u16 & 0xFF);
    data[6] = static_cast<uint8_t>((kd_u16 >> 8) & 0xFF);
    data[7] = static_cast<uint8_t>(kd_u16 & 0xFF);

    return send_ext_frame(arb, data);
  }

  // Type03 ENABLE 프레임을 전송한다.
  bool send_enable_frame(const uint8_t motor_id)
  {
    std::array<uint8_t, 8> data {};
    const uint32_t arb = pack_ext_id(TYPE03_ENABLE, host_id_, motor_id);
    return send_ext_frame(arb, data);
  }

  // Type04 STOP 프레임을 전송한다.
  bool send_stop_frame(const uint8_t motor_id, const bool clear_fault = false)
  {
    std::array<uint8_t, 8> data {};
    data[0] = clear_fault ? 1U : 0U;
    const uint32_t arb = pack_ext_id(TYPE04_STOP, host_id_, motor_id);
    return send_ext_frame(arb, data);
  }

  // 종료 시점에 알려진 모터 전체로 STOP을 1회 전송한다.
  void send_stop_once_on_shutdown()
  {
    if (sock_fd_ < 0) {
      return;
    }

    std::set<uint8_t> target_motor_ids = discovered_motor_ids_;
    for (const auto & kv : latest_cmd_by_motor_) {
      target_motor_ids.insert(kv.first);
    }
    for (const auto & kv : runtime_by_motor_) {
      target_motor_ids.insert(kv.first);
    }

    if (target_motor_ids.empty()) {
      RCLCPP_WARN(get_logger(), "shutdown Type04 skipped: no known motor id");
      return;
    }

    int ok = 0;
    int fail = 0;
    for (const uint8_t motor_id : target_motor_ids) {
      if (send_stop_frame(motor_id, false)) {
        ++ok;
      } else {
        ++fail;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    RCLCPP_INFO(
      get_logger(),
      "shutdown Type04 sent once: targets=%zu ok=%d fail=%d",
      target_motor_ids.size(),
      ok,
      fail);
  }

  // 스캔으로 찾은 모터에 ENABLE을 1회 전송한다.
  void send_enable_once_from_scan()
  {
    if (discovered_motor_ids_.empty()) {
      RCLCPP_WARN(get_logger(), "startup Type03 skipped: no discovered id from Type00 scan");
      return;
    }

    int ok = 0;
    int fail = 0;
    for (const uint8_t motor_id : discovered_motor_ids_) {
      MotorRuntime & rt = runtime_by_motor_[motor_id];
      if (send_enable_frame(motor_id)) {
        rt.enable_sent = true;
        rt.last_enable_sent = std::chrono::steady_clock::now();
        ++ok;
      } else {
        ++fail;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    RCLCPP_INFO(
      get_logger(),
      "startup Type03 sent once: targets=%zu ok=%d fail=%d",
      discovered_motor_ids_.size(),
      ok,
      fail);
  }

  // 외부 명령 적용 전, 해당 모터가 ready 상태인지 확인한다.
  bool ensure_motor_ready_for_control(const uint8_t motor_id)
  {
    MotorRuntime & rt = runtime_by_motor_[motor_id];
    if (rt.ready_for_control) {
      return true;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "waiting TYPE02 before applying /motor_cmd_array for motor_id=%u", motor_id);
    return false;
  }

  // 현재 홈 정책으로 모터별 home 명령 기본값을 생성한다.
  msgs::msg::MotorCMD make_home_command(const uint8_t motor_id) const
  {
    msgs::msg::MotorCMD home {};
    home.motor_id = motor_id;
    home.q_des = home_q_des_for_motor(motor_id);
    home.qd_des = static_cast<float>(home_policy_.qd_des_default);
    home.kp = home_kp_for_motor(motor_id);
    home.kd = home_kd_for_motor(motor_id);
    home.tau_ff = static_cast<float>(home_policy_.tau_ff_default);
    return home;
  }

  // 모터별 home 목표 위치를 반환한다.
  float home_q_des_for_motor(const uint8_t motor_id) const
  {
    return static_cast<float>(
      value_for_motor(home_policy_.q_des_by_motor, motor_id, home_policy_.q_des_default));
  }

  // 모터별 home kp를 반환한다.
  float home_kp_for_motor(const uint8_t motor_id) const
  {
    return static_cast<float>(
      value_for_motor(home_policy_.kp_by_motor, motor_id, home_policy_.kp_default));
  }

  // 모터별 home kd를 반환한다.
  float home_kd_for_motor(const uint8_t motor_id) const
  {
    return static_cast<float>(
      value_for_motor(home_policy_.kd_by_motor, motor_id, home_policy_.kd_default));
  }

  // j2/j3 순서 홈 복귀 시퀀서를 시작한다.
  void start_j2j3_sequence(const std::chrono::steady_clock::time_point & now_tp)
  {
    const MotorRuntime & rt2 = runtime_by_motor_[kJ2MotorId];
    const MotorRuntime & rt3 = runtime_by_motor_[kJ3MotorId];
    j2j3_seq_.j2_q0  = rt2.has_state ? rt2.last_state.q : 0.0F;
    j2j3_seq_.j3_q0  = rt3.has_state ? rt3.last_state.q : 0.0F;
    j2j3_seq_.j2_via = std::max(j2j3_seq_.j2_q0, kJ2SafeFloorRad);
    auto phase_dur = [](float dq) -> float {
      return std::max(
        kHomeSeqMinDurationSec, std::fabs(dq) / kHomeSeqSpeedRadPerSec);
    };
    j2j3_seq_.p1_dur = phase_dur(j2j3_seq_.j2_via - j2j3_seq_.j2_q0);
    j2j3_seq_.p2_dur = phase_dur(kJ3SafeHomeRad  - j2j3_seq_.j3_q0);
    j2j3_seq_.p3_dur = phase_dur(0.0F            - j2j3_seq_.j2_via);
    j2j3_seq_.p4_dur = phase_dur(0.0F            - kJ3SafeHomeRad);
    j2j3_seq_.phase_start = now_tp;
    j2j3_seq_.phase = 1;
    j2j3_seq_.done = false;
    RCLCPP_INFO(
      get_logger(),
      "j2/j3 home seq started: j2_q0=%.3f j3_q0=%.3f j2_via=%.3f "
      "dur=[%.1f, %.1f, %.1f, %.1f] s",
      static_cast<double>(j2j3_seq_.j2_q0),
      static_cast<double>(j2j3_seq_.j3_q0),
      static_cast<double>(j2j3_seq_.j2_via),
      static_cast<double>(j2j3_seq_.p1_dur),
      static_cast<double>(j2j3_seq_.p2_dur),
      static_cast<double>(j2j3_seq_.p3_dur),
      static_cast<double>(j2j3_seq_.p4_dur));
  }

  // 경과 시간 기준으로 페이즈를 전진시킨다 (dispatch 루프 시작 전 1회 호출).
  void advance_j2j3_phase(const std::chrono::steady_clock::time_point & now_tp)
  {
    if (j2j3_seq_.phase == 0) {
      return;
    }

    if (j2j3_seq_.in_dwell) {
      const double dwell_elapsed =
        std::chrono::duration<double>(now_tp - j2j3_seq_.dwell_start).count();
      if (dwell_elapsed < static_cast<double>(kHomeSeqDwellSec)) {
        return;
      }
      j2j3_seq_.in_dwell = false;
      const uint8_t next = j2j3_seq_.phase + 1U;
      if (next > 4U) {
        j2j3_seq_.phase = 0;
        j2j3_seq_.done = true;
        RCLCPP_INFO(get_logger(), "j2/j3 home seq complete");
      } else {
        j2j3_seq_.phase = next;
        j2j3_seq_.phase_start = now_tp;
        RCLCPP_INFO(get_logger(), "j2/j3 home seq → phase %u", static_cast<unsigned>(next));
      }
      return;
    }

    const float cur_dur =
      (j2j3_seq_.phase == 1) ? j2j3_seq_.p1_dur :
      (j2j3_seq_.phase == 2) ? j2j3_seq_.p2_dur :
      (j2j3_seq_.phase == 3) ? j2j3_seq_.p3_dur : j2j3_seq_.p4_dur;
    const double elapsed =
      std::chrono::duration<double>(now_tp - j2j3_seq_.phase_start).count();
    if (elapsed >= static_cast<double>(cur_dur)) {
      j2j3_seq_.in_dwell = true;
      j2j3_seq_.dwell_start = now_tp;
      RCLCPP_INFO(
        get_logger(), "j2/j3 home seq phase %u motion done → dwell %.1f s",
        static_cast<unsigned>(j2j3_seq_.phase),
        static_cast<double>(kHomeSeqDwellSec));
    }
  }

  // 현재 페이즈·경과 시간으로 모터별 q_des를 계산한다.
  float j2j3_seq_q_des(
    const uint8_t motor_id,
    const std::chrono::steady_clock::time_point & now_tp) const
  {
    float alpha = 1.0F;
    if (!j2j3_seq_.in_dwell) {
      const float cur_dur =
        (j2j3_seq_.phase == 1) ? j2j3_seq_.p1_dur :
        (j2j3_seq_.phase == 2) ? j2j3_seq_.p2_dur :
        (j2j3_seq_.phase == 3) ? j2j3_seq_.p3_dur : j2j3_seq_.p4_dur;
      const double elapsed =
        std::chrono::duration<double>(now_tp - j2j3_seq_.phase_start).count();
      alpha = std::clamp(
        static_cast<float>(elapsed / static_cast<double>(cur_dur)), 0.0F, 1.0F);
    }
    if (motor_id == kJ2MotorId) {
      switch (j2j3_seq_.phase) {
        case 1: return j2j3_seq_.j2_q0 + alpha * (j2j3_seq_.j2_via - j2j3_seq_.j2_q0);
        case 2: return j2j3_seq_.j2_via;
        case 3: return j2j3_seq_.j2_via + alpha * (0.0F - j2j3_seq_.j2_via);
        case 4: return 0.0F;
        default: return 0.0F;
      }
    } else {
      switch (j2j3_seq_.phase) {
        case 1: return j2j3_seq_.j3_q0;
        case 2: return j2j3_seq_.j3_q0 + alpha * (kJ3SafeHomeRad - j2j3_seq_.j3_q0);
        case 3: return kJ3SafeHomeRad;
        case 4: return kJ3SafeHomeRad + alpha * (0.0F - kJ3SafeHomeRad);
        default: return 0.0F;
      }
    }
  }

  // 현재 위치 기준으로 주기성을 고려한 home 목표 분기를 선택한다.
  float aligned_home_q_des(
    const uint8_t motor_id,
    const float current_q_raw,
    const float current_q_wrapped) const
  {
    const auto limits = position_limits_for_motor(motor_id);
    const float q_base = home_q_des_for_motor(motor_id);
    const float period = 2.0F * static_cast<float>(M_PI);
    const float span = limits.second - limits.first;

    // For motor 3 at +/-pi boundary, choose branch by RAW state sign.
    // Example: raw=-3.178 -> target=-pi, raw=+3.178 -> target=+pi.
    constexpr float kPiBoundaryTol = 0.05F;
    if (
      motor_id == kJ3MotorId &&
      std::fabs(span - period) <= 1e-3F &&
      std::fabs(std::fabs(q_base) - static_cast<float>(M_PI)) <= kPiBoundaryTol)
    {
      const float q_pos_pi = std::clamp(static_cast<float>(M_PI), limits.first, limits.second);
      const float q_neg_pi = std::clamp(-static_cast<float>(M_PI), limits.first, limits.second);
      return (current_q_raw < 0.0F) ? q_neg_pi : q_pos_pi;
    }

    return nearest_periodic_target(
      q_base,
      current_q_wrapped,
      period,
      limits.first,
      limits.second);
  }

  // home 복귀 선형 보간 궤적의 시작/목표 상태를 설정한다.
  void start_home_trajectory(
    const uint8_t motor_id,
    const float current_q,
    const std::chrono::steady_clock::time_point & now_tp,
    const char * reason)
  {
    CachedCommand & cached = latest_cmd_by_motor_[motor_id];
    const auto limits = position_limits_for_motor(motor_id);
    const float q_base = home_q_des_for_motor(motor_id);
    const float q_start = wrap_or_clamp_current_q_for_home(current_q, limits.first, limits.second);
    const float q_goal = aligned_home_q_des(motor_id, current_q, q_start);
    const float abs_dq = std::fabs(q_goal - q_start);

    cached.home_traj_start_q = q_start;
    cached.home_traj_goal_q = q_goal;
    cached.home_traj_duration_sec = kHomeReturnDurationSec;
    cached.home_traj_start_tp = now_tp;
    cached.home_traj_active = abs_dq > kHomeEpsilon;

    if (cached.home_traj_active) {
      RCLCPP_INFO(
        get_logger(),
        "motor_id=%u home trajectory started (%s, q_base=%.3f q_start=%.3f q_goal=%.3f duration=%.3f s)",
        motor_id,
        reason,
        static_cast<double>(q_base),
        static_cast<double>(q_start),
        static_cast<double>(q_goal),
        static_cast<double>(cached.home_traj_duration_sec));
    } else {
      RCLCPP_INFO(
        get_logger(),
        "motor_id=%u home trajectory skipped (%s, |dq|<=%.1e, q_base=%.3f q=%.3f)",
        motor_id,
        reason,
        static_cast<double>(kHomeEpsilon),
        static_cast<double>(q_base),
        static_cast<double>(q_goal));
    }
  }

  // 현재 시각에서 home 궤적의 q_des 샘플을 계산한다.
  float sample_home_trajectory_q_des(
    CachedCommand & cached,
    const std::chrono::steady_clock::time_point & now_tp) const
  {
    if (!cached.home_traj_active || cached.home_traj_duration_sec <= 0.0F) {
      return cached.home_traj_goal_q;
    }

    const double elapsed_sec =
      std::chrono::duration<double>(now_tp - cached.home_traj_start_tp).count();
    const float alpha = std::clamp(
      static_cast<float>(elapsed_sec / static_cast<double>(cached.home_traj_duration_sec)),
      0.0F,
      1.0F);
    const float q_des = cached.home_traj_start_q + alpha *
      (cached.home_traj_goal_q - cached.home_traj_start_q);
    if (alpha >= 1.0F) {
      cached.home_traj_active = false;
      return cached.home_traj_goal_q;
    }
    return q_des;
  }

  // ready 직후 외부 명령이 없을 때 pre-home 스트림을 arm 한다.
  // ① 시작 위치가 joint limit 밖이면 startup_blocked 세우고 차단.
  // ② 이미 홈 근처(kHomeNearThreshold)이면 궤적 없이 현재 위치 홀드.
  // ③ 그 외에는 7초 홈 복귀 궤적 실행.
  void start_home_stream_for_ready_motor(const uint8_t motor_id, const float current_q)
  {
    const auto limits = position_limits_for_motor(motor_id);
    MotorRuntime & rt = runtime_by_motor_[motor_id];

    if (current_q < limits.first || current_q > limits.second) {
      rt.startup_blocked = true;
      RCLCPP_ERROR(
        get_logger(),
        "motor_id=%u STARTUP BLOCKED: actual q=%.4f is outside limits [%.4f, %.4f]. "
        "Manually reposition the joint, then restart can_bridge.",
        motor_id,
        static_cast<double>(current_q),
        static_cast<double>(limits.first),
        static_cast<double>(limits.second));
      return;
    }

    CachedCommand & cached = latest_cmd_by_motor_[motor_id];
    if (cached.has_received_cmd) {
      return;
    }

    cached.cmd = make_home_command(motor_id);
    cached.cmd.stamp = to_builtin_time(now());
    cached.valid = true;
    cached.has_sent = false;
    cached.timeout_home_active = false;

    // 그리퍼(모터7)는 홈 궤적 없이 현재 위치 홀드.
    if (motor_id == 7U) {
      cached.cmd.q_des = current_q;
      cached.home_traj_active = false;
      cached.home_traj_start_q = current_q;
      cached.home_traj_goal_q = current_q;
      RCLCPP_INFO(get_logger(), "motor_id=7 gripper: holding in place (q=%.4f, no home traj)",
        static_cast<double>(current_q));
      return;
    }

    const float q_home = home_q_des_for_motor(motor_id);
    const float abs_dq = std::fabs(q_home - current_q);

    if (abs_dq <= kHomeNearThreshold) {
      // 이미 홈 근처: 실제 위치를 유지하고 궤적 없이 홀드.
      cached.cmd.q_des = current_q;
      cached.home_traj_active = false;
      cached.home_traj_start_q = current_q;
      cached.home_traj_goal_q = current_q;
      RCLCPP_INFO(
        get_logger(),
        "motor_id=%u already near home (q=%.4f, |dq|=%.4f <= %.4f): holding in place",
        motor_id,
        static_cast<double>(current_q),
        static_cast<double>(abs_dq),
        static_cast<double>(kHomeNearThreshold));
      return;
    }

    const auto now_tp = std::chrono::steady_clock::now();
    start_home_trajectory(motor_id, current_q, now_tp, "pre-home");
    cached.cmd.q_des = cached.home_traj_goal_q;
    RCLCPP_INFO(
      get_logger(),
      "motor_id=%u home Type01 stream armed after ready (current_q_raw=%.3f q_start=%.3f q_goal=%.3f duration=%.3f s)",
      motor_id,
      static_cast<double>(current_q),
      static_cast<double>(cached.home_traj_start_q),
      static_cast<double>(cached.home_traj_goal_q),
      static_cast<double>(cached.home_traj_duration_sec));
  }

  // /motor_cmd_array 입력을 모터별 최신 명령 캐시에 반영한다.
  // ① NaN/Inf 포함 시 해당 모터 명령 거부.
  // ② q_des가 joint limit 밖이면 거부.
  // ③ tau_ff는 kTauLimitByMotor로 클램프 (거부하지 않고 제한).
  void on_cmd_array(const msgs::msg::MotorCMDArray::SharedPtr msg)
  {
    if (msg->commands.empty()) {
      return;
    }
    const auto now_tp = std::chrono::steady_clock::now();
    for (auto cmd : msg->commands) {
      // ① NaN/Inf 검사: 어느 필드라도 비정상이면 거부.
      if (
        !std::isfinite(cmd.q_des) || !std::isfinite(cmd.qd_des) ||
        !std::isfinite(cmd.tau_ff) || !std::isfinite(cmd.kp) || !std::isfinite(cmd.kd))
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "motor_id=%u external cmd REJECTED: NaN/Inf in fields "
          "(q_des=%g qd_des=%g tau_ff=%g kp=%g kd=%g)",
          cmd.motor_id,
          static_cast<double>(cmd.q_des), static_cast<double>(cmd.qd_des),
          static_cast<double>(cmd.tau_ff), static_cast<double>(cmd.kp),
          static_cast<double>(cmd.kd));
        continue;
      }
      // ② safe shutdown 중 외부 명령 거부 (fault / 온도 임계 초과).
      {
        const MotorRuntime & rt_check = runtime_by_motor_[cmd.motor_id];
        if (rt_check.safe_shutdown_thermal || rt_check.safe_shutdown_fault) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "[SAFETY] motor_id=%u external cmd REJECTED: safe shutdown active "
            "(thermal=%d fault=%d) — homing in progress",
            cmd.motor_id,
            static_cast<int>(rt_check.safe_shutdown_thermal),
            static_cast<int>(rt_check.safe_shutdown_fault));
          continue;
        }
      }
      // ③ joint position limit 검사.
      const auto limits = position_limits_for_motor(cmd.motor_id);
      if (cmd.q_des < limits.first || cmd.q_des > limits.second) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[SAFETY] motor_id=%u external cmd REJECTED: q_des=%.4f outside limits [%.4f, %.4f]",
          cmd.motor_id,
          static_cast<double>(cmd.q_des),
          static_cast<double>(limits.first),
          static_cast<double>(limits.second));
        continue;
      }
      // ④ tau_ff 클램프.
      {
        const auto it = kTauLimitByMotor.find(static_cast<int>(cmd.motor_id));
        if (it != kTauLimitByMotor.end()) {
          const float limit = it->second;
          const float clamped = std::clamp(cmd.tau_ff, -limit, limit);
          if (std::fabs(clamped - cmd.tau_ff) > 1e-4F) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "[SAFETY] motor_id=%u tau_ff clamped: %.4f -> %.4f (limit=±%.1f Nm)",
              cmd.motor_id,
              static_cast<double>(cmd.tau_ff),
              static_cast<double>(clamped),
              static_cast<double>(limit));
            cmd.tau_ff = clamped;
          }
        }
      }
      // ⑤ qd_des 클램프.
      {
        const double max_qd = max_qdot_for_motor(cmd.motor_id);
        if (max_qd > 0.0) {
          const float clamped_qd = std::clamp(
            cmd.qd_des,
            static_cast<float>(-max_qd),
            static_cast<float>(max_qd));
          if (std::fabs(clamped_qd - cmd.qd_des) > 1e-4F) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "[SAFETY] motor_id=%u qd_des clamped: %.4f -> %.4f (limit=±%.2f rad/s)",
              cmd.motor_id,
              static_cast<double>(cmd.qd_des),
              static_cast<double>(clamped_qd),
              max_qd);
            cmd.qd_des = clamped_qd;
          }
        }
      }
      CachedCommand & cached = latest_cmd_by_motor_[cmd.motor_id];
      cached.cmd = cmd;
      cached.valid = true;
      cached.has_received_cmd = true;
      cached.last_received_cmd = now_tp;
      cached.timeout_home_active = false;
      cached.home_traj_active = false;
    }
  }

  // 캐시된 최신 명령을 모터별 주기에 맞춰 전송한다.
  void dispatch_cached_commands()
  {
    const auto now_tp = std::chrono::steady_clock::now();
    msgs::msg::MotorCMDArray applied_batch {};
    applied_batch.stamp = to_builtin_time(now());
    for (auto & kv : latest_cmd_by_motor_) {
      const uint8_t motor_id = kv.first;
      CachedCommand & cached = kv.second;
      if (!cached.valid) {
        continue;
      }

      // 시작 위치가 joint limit 밖이면 모든 명령 전송을 차단한다.
      if (runtime_by_motor_[motor_id].startup_blocked) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "motor_id=%u STARTUP BLOCKED — no commands sent. Reposition and restart.",
          motor_id);
        continue;
      }

      const bool is_pre_home_stream = !cached.has_received_cmd;
      if (!is_pre_home_stream && !ensure_motor_ready_for_control(motor_id)) {
        continue;  // Hold external /motor_cmd_array until Type02 is observed.
      }

      const double tx_hz = tx_hz_for_motor(motor_id);
      if (tx_hz <= 0.0) {
        continue;
      }

      const auto period = std::chrono::duration<double>(1.0 / tx_hz);
      if (cached.has_sent) {
        const auto elapsed = now_tp - cached.last_sent;
        if (elapsed < period) {
          continue;
        }
      }

      bool cmd_timed_out = false;
      if (cmd_timeout_ms_ > 0 && cached.has_received_cmd) {
        const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now_tp - cached.last_received_cmd).count();
        cmd_timed_out = age_ms >= cmd_timeout_ms_;
      }

      // safe shutdown(fault/온도) 중이면 home 복귀 강제.
      const MotorRuntime & rt_dispatch = runtime_by_motor_[motor_id];
      const bool force_home = rt_dispatch.safe_shutdown_thermal || rt_dispatch.safe_shutdown_fault;
      const bool home_requested = cmd_timed_out || force_home;

      msgs::msg::MotorCMD outgoing = cached.cmd;
      if (is_pre_home_stream) {
        outgoing = make_home_command(motor_id);
        outgoing.q_des = sample_home_trajectory_q_des(cached, now_tp);
        outgoing.stamp = to_builtin_time(now());
      } else if (home_requested) {
        outgoing = make_home_command(motor_id);
        if (!cached.timeout_home_active) {
          cached.timeout_home_active = true;
          const char * reason = force_home ? "safe-shutdown" : "timeout-home";
          if (rt_dispatch.has_state) {
            start_home_trajectory(motor_id, rt_dispatch.last_state.q, now_tp, reason);
          } else {
            const auto limits = position_limits_for_motor(motor_id);
            const float q_home = std::clamp(home_q_des_for_motor(motor_id), limits.first, limits.second);
            cached.home_traj_active = false;
            cached.home_traj_start_q = q_home;
            cached.home_traj_goal_q = q_home;
            cached.home_traj_duration_sec = kHomeReturnDurationSec;
            cached.home_traj_start_tp = now_tp;
          }
          if (force_home) {
            RCLCPP_WARN(
              get_logger(),
              "[SAFETY] motor_id=%u safe shutdown: homing "
              "(thermal=%d fault=%d q_start=%.3f q_goal=%.3f duration=%.3f s)",
              motor_id,
              static_cast<int>(rt_dispatch.safe_shutdown_thermal),
              static_cast<int>(rt_dispatch.safe_shutdown_fault),
              static_cast<double>(cached.home_traj_start_q),
              static_cast<double>(cached.home_traj_goal_q),
              static_cast<double>(cached.home_traj_duration_sec));
          } else {
            RCLCPP_WARN(
              get_logger(),
              "motor_id=%u cmd timeout (%d ms): switching to home trajectory "
              "(q_start=%.3f q_goal=%.3f duration=%.3f s)",
              motor_id,
              cmd_timeout_ms_,
              static_cast<double>(cached.home_traj_start_q),
              static_cast<double>(cached.home_traj_goal_q),
              static_cast<double>(cached.home_traj_duration_sec));
          }
        }
        outgoing.q_des = sample_home_trajectory_q_des(cached, now_tp);
        outgoing.stamp = to_builtin_time(now());
      } else if (cached.timeout_home_active) {
        cached.timeout_home_active = false;
        cached.home_traj_active = false;
        RCLCPP_INFO(
          get_logger(),
          "motor_id=%u cmd restored: leaving home timeout mode (q_goal=%.3f duration=%.3f s)",
          motor_id,
          static_cast<double>(cached.home_traj_goal_q),
          static_cast<double>(cached.home_traj_duration_sec));
      }

      // q_des 슬루레이트 클램프: 모든 경로(홈/타임아웃/외부 명령)에 적용.
      // 첫 전송 시 실제 모터 위치로 초기화해 시작 직후 점프를 방지한다.
      {
        const MotorRuntime & rt_slew = runtime_by_motor_[motor_id];
        if (!cached.slew_initialized) {
          cached.last_sent_q_des =
            rt_slew.has_state ? rt_slew.last_state.q : outgoing.q_des;
          cached.slew_initialized = true;
        }
        const double max_qdot = max_qdot_for_motor(motor_id);
        if (max_qdot > 0.0 && cached.has_sent) {
          // dt를 50ms로 클램프해 일시 정지 후 재시작 시 큰 점프를 방지한다.
          const double dt = std::min(
            std::chrono::duration<double>(now_tp - cached.last_sent).count(), 0.05);
          const double max_dq = max_qdot * dt;
          const float q_des_requested = outgoing.q_des;
          const double dq =
            static_cast<double>(q_des_requested) - static_cast<double>(cached.last_sent_q_des);
          if (std::fabs(dq) > max_dq) {
            outgoing.q_des =
              cached.last_sent_q_des + static_cast<float>(std::copysign(max_dq, dq));
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "motor_id=%u q_des slew clamped: requested=%.4f sent=%.4f "
              "(max_qdot=%.2f rad/s dt=%.4f s)",
              motor_id,
              static_cast<double>(q_des_requested),
              static_cast<double>(outgoing.q_des),
              max_qdot,
              dt);
          }
        }
      }

      // tau_ff 슬루레이트 클램프: 0(홈)에서 중력 보상값으로의 급격한 전환을 방지.
      // 정상 운행 중엔 중력 보상값이 연속적으로 변하므로 실질적으로 발동되지 않는다.
      {
        if (!cached.tau_slew_initialized) {
          cached.last_sent_tau_ff = 0.0F;  // 홈 명령은 tau_ff=0에서 시작
          cached.tau_slew_initialized = true;
        }
        const auto it = kMaxTauRateByMotor.find(static_cast<int>(motor_id));
        if (it != kMaxTauRateByMotor.end() && cached.has_sent) {
          const float max_tau_rate = it->second;
          const double dt = std::min(
            std::chrono::duration<double>(now_tp - cached.last_sent).count(), 0.05);
          const float max_dtau = max_tau_rate * static_cast<float>(dt);
          const float tau_requested = outgoing.tau_ff;
          const float dtau = tau_requested - cached.last_sent_tau_ff;
          if (std::fabs(dtau) > max_dtau) {
            outgoing.tau_ff = cached.last_sent_tau_ff + std::copysign(max_dtau, dtau);
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "motor_id=%u tau_ff slew clamped: requested=%.3f sent=%.3f "
              "(max_rate=%.1f Nm/s dt=%.4f s)",
              motor_id,
              static_cast<double>(tau_requested),
              static_cast<double>(outgoing.tau_ff),
              static_cast<double>(max_tau_rate),
              dt);
          }
        }
      }

      if (!send_mit_command(outgoing)) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(), 2000, "failed to send TYPE01 command frame");
      } else {
        applied_batch.commands.push_back(outgoing);
      }
      cached.has_sent = true;
      cached.last_sent = now_tp;
      cached.last_sent_q_des = outgoing.q_des;
      cached.last_sent_tau_ff = outgoing.tau_ff;
    }
    if (!applied_batch.commands.empty()) {
      applied_cmd_array_pub_->publish(applied_batch);
    }
  }

  // Type00 브로드캐스트 스캔으로 대상 모터 ID 목록을 수집한다.
  void scan_motor_ids()
  {
    if (scan_min_id_ > scan_max_id_) {
      std::swap(scan_min_id_, scan_max_id_);
    }

    std::array<uint8_t, 8> zeros {};
    for (int id = scan_min_id_; id <= scan_max_id_; ++id) {
      const uint32_t arb = pack_ext_id(TYPE00_GET_DEVICE_ID, host_id_, static_cast<uint8_t>(id));
      (void)send_ext_frame(arb, zeros);
    }

    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(scan_wait_ms_);
    std::set<int> found;
    while (std::chrono::steady_clock::now() < deadline) {
      auto frame_opt = recv_frame();
      if (!frame_opt.has_value()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      const auto & frame = frame_opt.value();
      if ((frame.can_id & CAN_EFF_FLAG) == 0) {
        continue;
      }
      const uint32_t arb = frame.can_id & CAN_EFF_MASK;
      const uint8_t type = comm_type_from_arb(arb);
      if (type != TYPE00_GET_DEVICE_ID) {
        continue;
      }
      int motor_id = -1;
      const uint8_t id1 = data1_from_arb(arb);
      const uint8_t id2 = static_cast<uint8_t>(data2_from_arb(arb) & 0xFF);
      // Manual form: Type00 reply uses bit7~0 = 0xFE and bit15~8 = motor id.
      if (id1 == 0xFE) {
        motor_id = static_cast<int>(id2);
      } else if (id2 == 0xFE) {
        // Be tolerant to swapped encodings.
        motor_id = static_cast<int>(id1);
      } else {
        // Fallback for legacy format.
        motor_id = static_cast<int>(id1);
      }
      if (motor_id >= scan_min_id_ && motor_id <= scan_max_id_) {
        found.insert(motor_id);
      }
    }

    discovered_motor_ids_.clear();
    for (const int id : found) {
      discovered_motor_ids_.insert(static_cast<uint8_t>(id));
    }

    if (discovered_motor_ids_.empty()) {
      RCLCPP_WARN(
        get_logger(), "Type00 scan found no motor in id range [%d, %d]", scan_min_id_,
        scan_max_id_);
      return;
    }

    std::string ids;
    for (const uint8_t id : discovered_motor_ids_) {
      if (!ids.empty()) {
        ids += ",";
      }
      ids += std::to_string(static_cast<int>(id));
    }
    RCLCPP_INFO(get_logger(), "Type00 scan discovered motor ids: [%s]", ids.c_str());
  }

  // 주기 tick에서 Rx(Type02)를 배치 처리하고 Tx 디스패치를 수행한다.
  void poll_can_frames()
  {
    // Per tick, keep only the last sample/event per motor_id.
    std::map<uint8_t, msgs::msg::MotorState> state_batch_by_motor;
    std::map<uint8_t, msgs::msg::MotorError> error_batch_by_motor;

    int processed = 0;
    while (processed < rx_max_frames_per_tick_) {
      auto frame_opt = recv_frame();
      if (!frame_opt.has_value()) {
        break;
      }
      ++processed;
      const auto & frame = frame_opt.value();

      if ((frame.can_id & CAN_EFF_FLAG) == 0) {
        continue;
      }
      if (frame.can_dlc != 8) {
        continue;
      }
      const uint32_t arb = frame.can_id & CAN_EFF_MASK;
      if (comm_type_from_arb(arb) != TYPE02_FEEDBACK) {
        continue;  // Type02 only
      }

      const uint16_t data2 = data2_from_arb(arb);
      const uint8_t motor_id = static_cast<uint8_t>(data2 & 0xFF);
      const uint8_t flags = static_cast<uint8_t>((data2 >> 8) & 0xFF);
      const uint8_t mode_status = static_cast<uint8_t>((flags >> 6) & 0x03);
      const uint8_t fault_bits = static_cast<uint8_t>(flags & 0x3F);

      const uint16_t q_raw = be_u16(frame.data[0], frame.data[1]);
      const uint16_t qd_raw = be_u16(frame.data[2], frame.data[3]);
      const uint16_t tau_raw = be_u16(frame.data[4], frame.data[5]);
      const uint16_t temp_raw = be_u16(frame.data[6], frame.data[7]);
      const MitRanges & r = ranges_for(motor_id);
      const auto stamp = to_builtin_time(now());

      msgs::msg::MotorState st {};
      st.stamp = stamp;
      st.motor_id = motor_id;
      st.q = u16_to_float(q_raw, r.p_min, r.p_max);
      st.qd = u16_to_float(qd_raw, r.v_min, r.v_max);
      st.tau = u16_to_float(tau_raw, r.t_min, r.t_max);
      st.temp_c = static_cast<float>(temp_raw) / 10.0F;
      state_batch_by_motor[motor_id] = st;

      MotorRuntime & rt = runtime_by_motor_[motor_id];
      rt.has_state = true;
      rt.last_state = st;
      if (!rt.ready_for_control && rt.enable_sent) {
        if (mode_status == 2U) {
          rt.ready_for_control = true;
          RCLCPP_INFO(
            get_logger(), "motor_id=%u ready: TYPE02 run-mode observed after TYPE03", motor_id);
          start_home_stream_for_ready_motor(motor_id, st.q);
        } else {
          RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "motor_id=%u waiting run mode TYPE02 after TYPE03 (mode_status=%u)",
            motor_id,
            mode_status);
        }
      }

      // 온도 모니터링: 임계 초과 시 safe home 트리거, 정상 복귀 시 해제.
      if (st.temp_c >= kTempCriticalDegC) {
        if (!rt.safe_shutdown_thermal) {
          rt.safe_shutdown_thermal = true;
          RCLCPP_ERROR(
            get_logger(),
            "[SAFETY] motor_id=%u thermal critical: temp=%.1f°C >= %.1f°C — safe home triggered",
            motor_id,
            static_cast<double>(st.temp_c),
            static_cast<double>(kTempCriticalDegC));
        }
      } else if (st.temp_c >= kTempWarnDegC) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[SAFETY] motor_id=%u temperature warning: %.1f°C (critical=%.1f°C)",
          motor_id,
          static_cast<double>(st.temp_c),
          static_cast<double>(kTempCriticalDegC));
      } else if (rt.safe_shutdown_thermal) {
        rt.safe_shutdown_thermal = false;
        RCLCPP_INFO(
          get_logger(),
          "motor_id=%u temperature normal: %.1f°C — thermal safe shutdown cleared",
          motor_id,
          static_cast<double>(st.temp_c));
      }

      const uint8_t prev_fault = last_fault_bits_[motor_id];
      const bool changed = (prev_fault != fault_bits);
      last_fault_bits_[motor_id] = fault_bits;
      if (!changed) {
        continue;
      }

      msgs::msg::MotorError err {};
      err.stamp = stamp;
      err.motor_id = motor_id;
      err.mode_status = mode_status;
      err.prev_fault_bits = prev_fault;
      err.fault_bits = fault_bits;
      if (prev_fault == 0 && fault_bits != 0) {
        err.event = msgs::msg::MotorError::EVENT_RAISED;
        rt.safe_shutdown_fault = true;
        RCLCPP_ERROR(
          get_logger(),
          "[SAFETY] motor_id=%u fault raised: bits=0x%02X — safe home triggered",
          motor_id,
          static_cast<unsigned>(fault_bits));
      } else if (prev_fault != 0 && fault_bits == 0) {
        err.event = msgs::msg::MotorError::EVENT_CLEARED;
        rt.safe_shutdown_fault = false;
        RCLCPP_INFO(
          get_logger(),
          "motor_id=%u fault cleared (was 0x%02X) — fault safe shutdown cleared",
          motor_id,
          static_cast<unsigned>(prev_fault));
      } else {
        err.event = msgs::msg::MotorError::EVENT_CHANGED;
        RCLCPP_WARN(
          get_logger(),
          "[SAFETY] motor_id=%u fault changed: 0x%02X → 0x%02X",
          motor_id,
          static_cast<unsigned>(prev_fault),
          static_cast<unsigned>(fault_bits));
      }
      error_batch_by_motor[motor_id] = err;
    }

    if (!state_batch_by_motor.empty()) {
      msgs::msg::MotorStateArray batch {};
      batch.stamp = to_builtin_time(now());
      batch.states.reserve(state_batch_by_motor.size());
      for (const auto & kv : state_batch_by_motor) {
        batch.states.push_back(kv.second);
      }
      state_array_pub_->publish(batch);
    }

    if (!error_batch_by_motor.empty()) {
      msgs::msg::MotorErrorArray batch {};
      batch.stamp = to_builtin_time(now());
      batch.errors.reserve(error_batch_by_motor.size());
      for (const auto & kv : error_batch_by_motor) {
        batch.errors.push_back(kv.second);
      }
      error_array_pub_->publish(batch);
    }

    // Rx is event-driven; Tx is rate-controlled per motor using latest cached command.
    dispatch_cached_commands();
  }

  int sock_fd_{-1};
  std::string channel_;
  uint8_t host_id_{0xFD};
  int scan_min_id_{1};
  int scan_max_id_{10};
  int scan_wait_ms_{200};
  int rx_max_frames_per_tick_{100};
  int cmd_timeout_ms_{100};

  // j2/j3 순서 홈 복귀 (4단계, 거리 비례 duration + 페이즈별 dwell)
  // P1: j2→via(-30°)  P2: j3→-45°  P3: j2→0°  P4: j3→0°
  struct J2J3Seq {
    uint8_t phase {0};    // 0=idle, 1~4=각 페이즈
    bool done {false};    // 한 사이클 완료 후 재기동 방지
    bool in_dwell {false};  // 보간 완료 후 안착 대기 중
    std::chrono::steady_clock::time_point phase_start {};
    std::chrono::steady_clock::time_point dwell_start {};
    float j2_q0  {0.0F};
    float j3_q0  {0.0F};
    float j2_via {0.0F};  // P1 목표: max(j2_q0, kJ2SafeFloorRad)
    float p1_dur {1.0F};
    float p2_dur {1.0F};
    float p3_dur {1.0F};
    float p4_dur {1.0F};
  } j2j3_seq_ {};

  std::unordered_map<uint8_t, uint8_t> last_fault_bits_;
  std::unordered_map<uint8_t, CachedCommand> latest_cmd_by_motor_;
  std::unordered_map<uint8_t, MotorRuntime> runtime_by_motor_;
  std::set<uint8_t> discovered_motor_ids_;
  TxPolicy tx_policy_;
  HomePolicy home_policy_;
  rclcpp::Publisher<msgs::msg::MotorStateArray>::SharedPtr state_array_pub_;
  rclcpp::Publisher<msgs::msg::MotorErrorArray>::SharedPtr error_array_pub_;
  rclcpp::Publisher<msgs::msg::MotorCMDArray>::SharedPtr applied_cmd_array_pub_;
  rclcpp::Subscription<msgs::msg::MotorCMDArray>::SharedPtr cmd_array_sub_;
  rclcpp::TimerBase::SharedPtr io_timer_;
};

// ROS2 엔트리포인트
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CanBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
