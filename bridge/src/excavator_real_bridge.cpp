#include <excavator_api/excavator_control.hpp>
#include <excavator_api/excavator_receive.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using clock_t = std::chrono::system_clock;

constexpr int kProtocolVersion = 1;
constexpr int kActionDim = 4;
constexpr int kLowerAxisCount = 8;

struct Options {
    std::string host{"127.0.0.1"};
    int port{8765};
    std::string can_if{"can0"};
    std::string imu_if{"can1"};
    std::string can_shm{"canlib_shm_bridge"};
    std::string imu_shm{"imu_canlib_shm_bridge"};
    bool create_mapping{true};
    bool can_simulation{true};
    bool imu_simulation{true};
    bool can_bus_enabled{false};
    bool watchdog_enabled{true};
    int heartbeat_timeout_ms{200};
    int read_timeout_ms{100};
    int image_width{160};
    int image_height{120};
    std::string pid_yaml{};
    bool one_shot{false};
    excavator_api::ControlMode control_mode{excavator_api::ControlMode::ClosedLoopVelocityScalar};
};

std::uint64_t nowNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock_t::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

std::uint64_t steadyNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string envString(const char* name, const std::string& fallback = "") {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    return std::string(raw);
}

bool bucketImu0RollProfileEnabled() {
    const std::string raw = lowerCopy(envString("EXCAVATOR_BUCKET_IMU0_PROFILE", "legacy_y"));
    return raw == "roll_ccw90" || raw == "rotated_ccw90" || raw == "imu0_roll" || raw == "roll";
}

bool parseBool(const std::string& raw) {
    const std::string v = lowerCopy(raw);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("invalid bool value: " + raw);
}

int parseInt(const std::string& raw, const std::string& name, int min_value, int max_value) {
    std::size_t idx = 0;
    const int value = std::stoi(raw, &idx, 10);
    if (idx != raw.size() || value < min_value || value > max_value) {
        std::ostringstream oss;
        oss << name << " must be in [" << min_value << "," << max_value << "]";
        throw std::runtime_error(oss.str());
    }
    return value;
}

void printHelp(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "JSON/TCP bridge between testbed bridge_tcp and control/excavator_api.\n"
        << "Safety defaults keep real CAN disabled unless explicitly enabled.\n"
        << "\n"
        << "Options:\n"
        << "  --host <ip>                         listen host (default 127.0.0.1)\n"
        << "  --port <1-65535>                    listen port (default 8765)\n"
        << "  --can-if <canX>                     excavator CAN interface (default can0)\n"
        << "  --imu-if <canX|usbcan[dev][:ch][@bitrate]>\n"
        << "                                      IMU CAN interface (default can1; e.g. usbcan0@250000)\n"
        << "  --can-shm <name>                    excavator shared memory name\n"
        << "  --imu-shm <name>                    IMU shared memory name\n"
        << "  --create-mapping <bool>             create SHM mapping (default true)\n"
        << "  --can-simulation <bool>             simulate excavator CAN (default true)\n"
        << "  --imu-simulation <bool>             simulate IMU CAN (default true)\n"
        << "  --can-bus-enabled <bool>            allow CAN writes (default false)\n"
        << "  --control-mode <mode>               closed_loop_velocity_scalar|open_loop_motor_speed\n"
        << "  --heartbeat-timeout-ms <ms>         watchdog timeout (default 200)\n"
        << "  --disable-watchdog                  disable zero-command watchdog\n"
        << "  --read-timeout-ms <ms>              receive timeout for read_state (default 100)\n"
        << "  --image-width <px>                  placeholder fpv width (default 160)\n"
        << "  --image-height <px>                 placeholder fpv height (default 120)\n"
        << "  --pid-yaml <path>                   load 9x8 PID/feedforward vectors\n"
        << "  --one-shot                          stop after one client disconnects\n"
        << "  --help                              show this message\n";
}

bool parseVector8Values(const std::string& line, std::vector<double>& out_values) {
    const std::size_t l = line.find('[');
    const std::size_t r = line.find(']');
    if (l == std::string::npos || r == std::string::npos || r <= l) {
        return false;
    }
    std::string body = line.substr(l + 1, r - l - 1);
    for (char& c : body) {
        if (c == ',') c = ' ';
    }
    std::istringstream iss(body);
    out_values.clear();
    double v = 0.0;
    while (iss >> v) {
        out_values.push_back(v);
    }
    return out_values.size() == kLowerAxisCount;
}

bool loadPidVectorsFromYaml(const std::string& yaml_path,
                            std::vector<std::vector<double>>& pid_vectors) {
    std::ifstream fin(yaml_path);
    if (!fin.is_open()) {
        return false;
    }
    const std::vector<std::string> keys = {
        "position_kp",
        "position_ki",
        "position_kd",
        "velocity_kp",
        "velocity_ki",
        "velocity_kd",
        "velocity_scalar_max",
        "feedforward_scalar_threshold_pos",
        "feedforward_scalar_threshold_neg",
    };
    pid_vectors.assign(keys.size(), std::vector<double>{});
    std::string line;
    while (std::getline(fin, line)) {
        const std::size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const std::string key = keys[i] + ":";
            if (line.find(key) == std::string::npos) {
                continue;
            }
            std::vector<double> vals;
            if (!parseVector8Values(line, vals)) {
                return false;
            }
            pid_vectors[i] = std::move(vals);
        }
    }
    for (const auto& vals : pid_vectors) {
        if (vals.size() != kLowerAxisCount) {
            return false;
        }
    }
    return true;
}

std::string nextArgValue(int& i, int argc, char** argv, const std::string& key) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
    }
    ++i;
    return argv[i];
}

Options parseArgs(int argc, char** argv) {
    Options opt{};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        const std::size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            value = arg.substr(eq + 1);
            arg = arg.substr(0, eq);
        }

        auto valueOrNext = [&]() {
            return eq == std::string::npos ? nextArgValue(i, argc, argv, arg) : value;
        };

        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            std::exit(0);
        } else if (arg == "--host") {
            opt.host = valueOrNext();
        } else if (arg == "--port") {
            opt.port = parseInt(valueOrNext(), "port", 1, 65535);
        } else if (arg == "--can-if") {
            opt.can_if = valueOrNext();
        } else if (arg == "--imu-if") {
            opt.imu_if = valueOrNext();
        } else if (arg == "--can-shm") {
            opt.can_shm = valueOrNext();
        } else if (arg == "--imu-shm") {
            opt.imu_shm = valueOrNext();
        } else if (arg == "--create-mapping") {
            opt.create_mapping = parseBool(valueOrNext());
        } else if (arg == "--can-simulation" || arg == "--can-sim") {
            opt.can_simulation = parseBool(valueOrNext());
        } else if (arg == "--imu-simulation" || arg == "--imu-sim") {
            opt.imu_simulation = parseBool(valueOrNext());
        } else if (arg == "--can-bus-enabled" || arg == "--can-bus") {
            opt.can_bus_enabled = parseBool(valueOrNext());
        } else if (arg == "--control-mode") {
            const std::string mode = lowerCopy(valueOrNext());
            if (mode == "closed_loop_velocity_scalar" || mode == "closed-loop-velocity-scalar" ||
                mode == "velocity_scalar") {
                opt.control_mode = excavator_api::ControlMode::ClosedLoopVelocityScalar;
            } else if (mode == "open_loop_motor_speed" || mode == "open-loop-motor-speed" ||
                       mode == "open_loop") {
                opt.control_mode = excavator_api::ControlMode::OpenLoopMotorSpeed;
            } else {
                throw std::runtime_error("unsupported control mode: " + mode);
            }
        } else if (arg == "--heartbeat-timeout-ms") {
            opt.heartbeat_timeout_ms =
                parseInt(valueOrNext(), "heartbeat-timeout-ms", 1, 60000);
        } else if (arg == "--disable-watchdog") {
            opt.watchdog_enabled = false;
        } else if (arg == "--read-timeout-ms") {
            opt.read_timeout_ms = parseInt(valueOrNext(), "read-timeout-ms", 1, 60000);
        } else if (arg == "--image-width") {
            opt.image_width = parseInt(valueOrNext(), "image-width", 1, 4096);
        } else if (arg == "--image-height") {
            opt.image_height = parseInt(valueOrNext(), "image-height", 1, 4096);
        } else if (arg == "--pid-yaml") {
            opt.pid_yaml = valueOrNext();
        } else if (arg == "--one-shot") {
            opt.one_shot = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    return opt;
}

json responseMessage(const std::string& type,
                     json payload = json::object(),
                     bool ok = true,
                     const std::string& error = "") {
    return json{
        {"version", kProtocolVersion},
        {"type", type},
        {"ok", ok},
        {"error", error},
        {"payload", std::move(payload)},
    };
}

bool sendAll(int fd, const std::string& data) {
    const char* ptr = data.data();
    std::size_t remaining = data.size();
    while (remaining > 0) {
        const ssize_t n = ::send(fd, ptr, remaining, 0);
        if (n <= 0) {
            return false;
        }
        ptr += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

std::string base64Encode(const std::vector<std::uint8_t>& input) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((input.size() + 2U) / 3U) * 4U);
    for (std::size_t i = 0; i < input.size(); i += 3U) {
        const std::uint32_t a = input[i];
        const std::uint32_t b = (i + 1U < input.size()) ? input[i + 1U] : 0U;
        const std::uint32_t c = (i + 2U < input.size()) ? input[i + 2U] : 0U;
        const std::uint32_t triple = (a << 16U) | (b << 8U) | c;
        out.push_back(table[(triple >> 18U) & 0x3FU]);
        out.push_back(table[(triple >> 12U) & 0x3FU]);
        out.push_back((i + 1U < input.size()) ? table[(triple >> 6U) & 0x3FU] : '=');
        out.push_back((i + 2U < input.size()) ? table[triple & 0x3FU] : '=');
    }
    return out;
}

json vectorHeadJson(const excavator_api::Vector8d& v, int n) {
    json out = json::array();
    for (int i = 0; i < n; ++i) {
        out.push_back(v(i));
    }
    return out;
}

json vector8Json(const excavator_api::Vector8d& v) {
    return vectorHeadJson(v, 8);
}

json vector12Json(const excavator_api::Vector12i& v) {
    json out = json::array();
    for (int i = 0; i < 12; ++i) {
        out.push_back(v(i));
    }
    return out;
}

json doubleArrayJson(const std::array<double, 3>& values) {
    return json::array({values[0], values[1], values[2]});
}

json doubleArrayJson(const std::array<double, 4>& values) {
    return json::array({values[0], values[1], values[2], values[3]});
}

json imuHealthJson(const excavator_api::ImuHealth& h, std::uint64_t steady_now_ns) {
    json online = json::array();
    json valid_attitude = json::array();
    json valid_quaternion = json::array();
    json valid_gyro = json::array();
    json valid_accel = json::array();
    json packet_loss_count = json::array();
    json host_rx_age_ms = json::array();
    for (std::size_t i = 0; i < excavator_api::kImuDeviceCount; ++i) {
        online.push_back(static_cast<int>(h.online[i]));
        valid_attitude.push_back(static_cast<int>(h.valid_attitude[i]));
        valid_quaternion.push_back(static_cast<int>(h.valid_quaternion[i]));
        valid_gyro.push_back(static_cast<int>(h.valid_gyro[i]));
        valid_accel.push_back(static_cast<int>(h.valid_accel[i]));
        packet_loss_count.push_back(static_cast<int>(h.packet_loss_count[i]));
        if (h.host_rx_time_ns[i] == 0U || h.host_rx_time_ns[i] > steady_now_ns) {
            host_rx_age_ms.push_back(-1.0);
        } else {
            host_rx_age_ms.push_back(
                static_cast<double>(steady_now_ns - h.host_rx_time_ns[i]) / 1000000.0);
        }
    }
    return json{
        {"online", online},
        {"valid_attitude", valid_attitude},
        {"valid_quaternion", valid_quaternion},
        {"valid_gyro", valid_gyro},
        {"valid_accel", valid_accel},
        {"packet_loss_count", packet_loss_count},
        {"host_rx_age_ms", host_rx_age_ms},
    };
}

json imuDebugJson(const excavator_api::ImuDebug& d, std::uint64_t steady_now_ns) {
    const bool bucket_roll_profile = bucketImu0RollProfileEnabled();
    const std::string joint_rpy_profile =
        lowerCopy(envString("EXCAVATOR_JOINT_RPY_PROFILE", "legacy_diff"));
    const std::string bucket_profile =
        bucket_roll_profile ? envString("EXCAVATOR_BUCKET_IMU0_PROFILE", "roll_ccw90")
                            : "legacy_y";
    const std::string bucket_qpos_source = lowerCopy(envString(
        "EXCAVATOR_BUCKET_QPOS_SOURCE",
        bucket_roll_profile ? "rpy" : "legacy_quaternion"));
    const bool daoyuan_chain =
        joint_rpy_profile == "daoyuan_chain" || bucket_qpos_source == "daoyuan_chain" ||
        bucket_qpos_source == "daoyuan_rpy" || bucket_qpos_source == "chain_rpy";
    const std::string bucket_gyro_axis =
        daoyuan_chain ? "-(imu0-x+imu1-y)"
                      : (bucket_roll_profile ? "imu1-y-minus-imu0-x" : "imu0-y-minus-imu1-y");
    std::string bucket_position_axis =
        daoyuan_chain ? "-(imu0-roll+imu1-pitch)+daoyuan_bucket_offset"
                      : (bucket_roll_profile ? "imu0-roll-minus-imu1-pitch-minus-reference"
                                             : "imu0-imu1-quaternion-y-twist");
    if (!daoyuan_chain &&
        (bucket_qpos_source == "gravity_hinge" || bucket_qpos_source == "gravity" ||
         bucket_qpos_source == "accel_hinge")) {
        bucket_position_axis =
            "gravity-hinge-median21:imu0+X-phase-minus-imu1+Y-phase";
    }
    const std::string stick_gyro_axis =
        daoyuan_chain ? "imu1-y+imu2-y" : "imu1-y-minus-imu2-y";
    const std::string stick_position_axis =
        daoyuan_chain ? "imu1-pitch+imu2-pitch+daoyuan_stick_offset"
                      : "imu1-pitch-minus-imu2-pitch";
    json devices = json::array();
    for (std::size_t i = 0; i < excavator_api::kImuDeviceCount; ++i) {
        const auto& src = d.devices[i];
        double host_rx_age_ms = -1.0;
        if (src.host_rx_time_ns != 0U && src.host_rx_time_ns <= steady_now_ns) {
            host_rx_age_ms =
                static_cast<double>(steady_now_ns - src.host_rx_time_ns) / 1000000.0;
        }
        devices.push_back(json{
            {"index", i},
            {"device_addr", static_cast<int>(src.device_addr)},
            {"online", static_cast<int>(src.online)},
            {"valid_attitude", static_cast<int>(src.valid_attitude)},
            {"valid_quaternion", static_cast<int>(src.valid_quaternion)},
            {"valid_gyro", static_cast<int>(src.valid_gyro)},
            {"valid_accel", static_cast<int>(src.valid_accel)},
            {"packet_loss_count", static_cast<int>(src.packet_loss_count)},
            {"imu_timestamp_ms", src.imu_timestamp_ms},
            {"host_rx_time_ns", src.host_rx_time_ns},
            {"host_rx_age_ms", host_rx_age_ms},
            {"rpy_rad", doubleArrayJson(src.rpy_rad)},
            {"rpy_raw_deg", doubleArrayJson(src.rpy_raw_deg)},
            {"gyro_dps", doubleArrayJson(src.gyro_dps)},
            {"accel_mps2", doubleArrayJson(src.accel_mps2)},
            {"quaternion_wxyz", doubleArrayJson(src.quaternion_wxyz)},
        });
    }
    return json{
        {"joint_rpy_profile", joint_rpy_profile},
        {"bucket_imu0_profile", bucket_profile},
        {"bucket_imu0_reference_rad",
         bucket_roll_profile ? envString("EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD", "0") : ""},
        {"bucket_imu0_sign",
         bucket_roll_profile ? envString("EXCAVATOR_BUCKET_IMU0_SIGN",
                                          envString("EXCAVATOR_BUCKET_IMU0_GYRO_SIGN", "1"))
                             : ""},
        {"daoyuan_stick_policy_offset_rad",
         envString("EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD", "0.19801020488135143")},
        {"swing_policy_offset_rad",
         envString("EXCAVATOR_SWING_POLICY_OFFSET_RAD", "0")},
        {"daoyuan_bucket_policy_offset_rad",
         envString("EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD", "-2.006833804661174")},
        {"bucket_qpos_source", bucket_qpos_source},
        {"bucket_gravity_hinge_reference_rad",
         envString("EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD", "2.0839045979023254")},
        {"bucket_gravity_hinge_policy_offset_rad",
         envString("EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD", "-2.025561263010988")},
        {"bucket_gravity_hinge_median_window",
         envString("EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW", "21")},
        {"devices", devices},
        {"joint_velocity_mapping",
         json{
             {"swing", json{{"device_index", 3}, {"gyro_axis", "-z"}}},
             {"boom", json{{"device_index", 2}, {"gyro_axis", "y"}}},
             {"stick",
              json{{"device_index", 1},
                   {"gyro_axis", stick_gyro_axis},
                   {"position_axis", stick_position_axis}}},
             {"bucket",
             json{{"device_index", 0},
                   {"gyro_axis", bucket_gyro_axis},
                   {"position_profile", bucket_profile},
                   {"position_axis", bucket_position_axis}}},
         }},
    };
}

std::array<double, kActionDim> parseAction4(const json& payload) {
    if (!payload.contains("action") || !payload.at("action").is_array()) {
        throw std::runtime_error("send_action payload missing array field 'action'");
    }
    const auto& action = payload.at("action");
    if (action.size() != static_cast<std::size_t>(kActionDim)) {
        throw std::runtime_error("action must have exactly 4 elements");
    }

    std::array<double, kActionDim> out{};
    for (int i = 0; i < kActionDim; ++i) {
        if (!action.at(static_cast<std::size_t>(i)).is_number()) {
            throw std::runtime_error("action elements must be numeric");
        }
        const double v = action.at(static_cast<std::size_t>(i)).get<double>();
        if (!std::isfinite(v) || v < -1.0 || v > 1.0) {
            throw std::runtime_error("action elements must be finite and in [-1,1]");
        }
        out[static_cast<std::size_t>(i)] = v;
    }
    return out;
}

json imagePayload(int width, int height, std::uint64_t frame_id) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) *
                                    static_cast<std::size_t>(height) * 3U);
    const std::uint8_t frame_r = static_cast<std::uint8_t>((frame_id * 5U) % 255U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t idx = (static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(width) +
                                     static_cast<std::size_t>(x)) *
                                    3U;
            image[idx + 0U] = frame_r;
            image[idx + 1U] = static_cast<std::uint8_t>((x * 255) / std::max(1, width - 1));
            image[idx + 2U] = static_cast<std::uint8_t>((y * 255) / std::max(1, height - 1));
        }
    }
    return json{
        {"encoding", "raw_uint8"},
        {"shape", json::array({height, width, 3})},
        {"data_b64", base64Encode(image)},
    };
}

class BridgeApp {
public:
    explicit BridgeApp(Options options) : options_(std::move(options)) {}

    bool startRuntime() {
        if (options_.imu_simulation) {
            ::setenv("EXCAVATOR_IMU_PLACEHOLDER", "1", 1);
        }
        excavator_api::SessionConfig cfg{};
        cfg.can_if_name = options_.can_if;
        cfg.imu_if_name = options_.imu_if;
        cfg.can_shm_name = options_.can_shm;
        cfg.imu_shm_name = options_.imu_shm;
        cfg.create_mapping = options_.create_mapping;
        cfg.can_simulation = options_.can_simulation;
        cfg.imu_simulation = options_.imu_simulation;
        cfg.can_bus_enabled = options_.can_bus_enabled;

        if (!control_.connect(cfg) || !receive_.connect(cfg) || !control_.start()) {
            std::cerr << "runtime start failed: " << control_.lastError() << "\n";
            return false;
        }
        if (!control_.setControlMode(options_.control_mode)) {
            std::cerr << "failed to set control mode: " << control_.lastError() << "\n";
            return false;
        }
        if (!options_.pid_yaml.empty()) {
            std::vector<std::vector<double>> pid_vectors;
            if (!loadPidVectorsFromYaml(options_.pid_yaml, pid_vectors) ||
                !control_.setPidVectors(pid_vectors)) {
                std::cerr << "failed to load PID YAML: " << options_.pid_yaml
                          << " error=" << control_.lastError() << "\n";
                return false;
            }
            std::cerr << "loaded PID YAML: " << options_.pid_yaml << "\n";
        }
        (void)sendZeroCommand("startup");

        std::cerr << "excavator_real_bridge runtime started"
                  << " can_if=" << options_.can_if
                  << " imu_if=" << options_.imu_if
                  << " can_simulation=" << (options_.can_simulation ? "true" : "false")
                  << " imu_simulation=" << (options_.imu_simulation ? "true" : "false")
                  << " can_bus_enabled=" << (options_.can_bus_enabled ? "true" : "false")
                  << "\n";
        return true;
    }

    int serve() {
        const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "socket failed: " << std::strerror(errno) << "\n";
            return 1;
        }
        const int yes = 1;
        (void)::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(options_.port));
        if (::inet_pton(AF_INET, options_.host.c_str(), &addr.sin_addr) != 1) {
            std::cerr << "invalid IPv4 host: " << options_.host << "\n";
            ::close(server_fd);
            return 1;
        }
        if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            std::cerr << "bind failed: " << std::strerror(errno) << "\n";
            ::close(server_fd);
            return 1;
        }
        if (::listen(server_fd, 8) != 0) {
            std::cerr << "listen failed: " << std::strerror(errno) << "\n";
            ::close(server_fd);
            return 1;
        }
        startStateCacheThread();

        std::cerr << "excavator_real_bridge listening on "
                  << options_.host << ":" << options_.port << "\n";

        std::vector<std::thread> client_threads;
        while (!shutdown_requested_.load()) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(server_fd, &rfds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 100000;
            const int ready = ::select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
            if (ready < 0) {
                if (errno == EINTR) continue;
                std::cerr << "accept select failed: " << std::strerror(errno) << "\n";
                break;
            }
            if (ready == 0) {
                continue;
            }

            sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            const int client_fd =
                ::accept(server_fd, reinterpret_cast<sockaddr*>(&peer), &peer_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                std::cerr << "accept failed: " << std::strerror(errno) << "\n";
                break;
            }
            client_threads.emplace_back([this, client_fd]() {
                std::cerr << "client connected\n";
                handleClient(client_fd);
                ::close(client_fd);
                std::cerr << "client disconnected\n";
            });
            if (options_.one_shot) {
                break;
            }
        }

        shutdown_requested_.store(true);
        for (auto& thread : client_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        ::close(server_fd);
        stopStateCacheThread();
        (void)sendZeroCommand("shutdown");
        (void)control_.close();
        (void)receive_.close();
        return 0;
    }

private:
    void startStateCacheThread() {
        state_thread_running_.store(true);
        state_thread_ = std::thread([this]() { stateCacheLoop(); });
    }

    void stopStateCacheThread() {
        const bool was_running = state_thread_running_.exchange(false);
        state_cv_.notify_all();
        if (was_running && state_thread_.joinable()) {
            state_thread_.join();
        }
    }

    void stateCacheLoop() {
        while (state_thread_running_.load()) {
            excavator_api::Snapshot snap{};
            if (receive_.get(snap, std::chrono::milliseconds(options_.read_timeout_ms))) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    cached_snapshot_ = snap;
                    cached_snapshot_ns_ = nowNs();
                    has_cached_snapshot_ = true;
                    last_state_error_.clear();
                }
                state_cv_.notify_all();
            } else {
                std::lock_guard<std::mutex> lock(state_mutex_);
                last_state_error_ = receive_.lastError();
            }
        }
    }

    json handleMessage(const json& message, bool& close_connection) {
        if (!message.is_object()) {
            return responseMessage("error.response", json::object(), false,
                                   "bridge frame must decode to a JSON object");
        }
        if (message.value("version", -1) != kProtocolVersion) {
            return responseMessage("error.response", json::object(), false,
                                   "unsupported bridge protocol version");
        }
        if (!message.contains("type") || !message.at("type").is_string()) {
            return responseMessage("error.response", json::object(), false,
                                   "bridge frame missing message type");
        }

        const std::string type = message.at("type").get<std::string>();
        const json payload = message.value("payload", json::object());
        try {
            if (type == "send_action.request") {
                return handleSendAction(payload);
            }
            if (type == "send_status.request") {
                return handleSendStatus(payload);
            }
            if (type == "read_state.request") {
                return handleReadState(payload);
            }
            if (type == "reset.request") {
                {
                    std::lock_guard<std::mutex> lock(control_mutex_);
                    (void)sendZeroCommandLocked("reset");
                    last_valid_action_ns_ = 0;
                    watchdog_zeroed_ = true;
                }
                return responseMessage("reset.response", json{{"reset", true}});
            }
            if (type == "close.request") {
                close_connection = true;
                return responseMessage("close.response", json{{"closed", true}});
            }
            if (type == "shutdown.request") {
                close_connection = true;
                shutdown_requested_.store(true);
                return responseMessage("shutdown.response", json{{"shutdown", true}});
            }

            const std::string response_type =
                type.find(".request") != std::string::npos
                    ? type.substr(0, type.size() - std::string(".request").size()) + ".response"
                    : "unknown.response";
            return responseMessage(response_type, json::object(), false,
                                   "unsupported request type: " + type);
        } catch (const std::exception& exc) {
            const std::string response_type =
                type.find(".request") != std::string::npos
                    ? type.substr(0, type.size() - std::string(".request").size()) + ".response"
                    : "error.response";
            return responseMessage(response_type, json::object(), false, exc.what());
        }
    }

    json handleSendStatus(const json& payload) {
        std::uint16_t toggle_mask = 0;
        if (payload.contains("toggle_mask")) {
            toggle_mask = static_cast<std::uint16_t>(payload.at("toggle_mask").get<int>()) & 0x07FFu;
        }
        bool ok = false;
        std::string fault_code;
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            ok = control_.applyStatusToggleMask(toggle_mask);
            fault_code = ok ? "" : control_.lastError();
        }
        return responseMessage(
            "send_status.response",
            json{
                {"ack", ok},
                {"toggle_mask", toggle_mask},
                {"fault_code", fault_code},
            });
    }

    json handleSendAction(const json& payload) {
        const auto action4 = parseAction4(payload);
        excavator_api::SpeedScalarCmd cmd{};
        for (int i = 0; i < kActionDim; ++i) {
            cmd.speed_scalar(i) = action4[static_cast<std::size_t>(i)];
        }
        for (int i = kActionDim; i < kLowerAxisCount; ++i) {
            cmd.speed_scalar(i) = 0.0;
        }

        bool sent = false;
        std::string fault_code;
        const std::uint64_t ts = nowNs();
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            sent = control_.sendCommand(cmd);
            fault_code = sent ? "" : control_.lastError();
            last_valid_action_ns_ = ts;
            watchdog_zeroed_ = false;
        }

        json commanded = json::array();
        for (double v : action4) commanded.push_back(v);
        json raw = vector8Json(cmd.speed_scalar);
        return responseMessage(
            "send_action.response",
            json{
                {"ack", sent},
                {"fault_code", fault_code},
                {"controller_timestamp_ns", ts},
                {"commanded_action", commanded},
                {"raw_low_level_command", raw},
            });
    }

    json handleReadState(const json& payload) {
        const int step_id = payload.value("step_id", 0);
        (void)step_id;
        excavator_api::Snapshot snap{};
        std::uint64_t snapshot_ns = 0;
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            if (!has_cached_snapshot_) {
                state_cv_.wait_for(
                    lock,
                    std::chrono::milliseconds(options_.read_timeout_ms),
                    [this]() { return has_cached_snapshot_ || !state_thread_running_.load(); });
            }
            if (!has_cached_snapshot_) {
                const std::string err = last_state_error_.empty()
                                            ? "no cached state snapshot yet"
                                            : last_state_error_;
                return responseMessage("read_state.response", json::object(), false, err);
            }
            snap = cached_snapshot_;
            snapshot_ns = cached_snapshot_ns_;
        }
        if (snapshot_ns == 0) {
            return responseMessage("read_state.response", json::object(), false,
                                   "cached state snapshot timestamp missing");
        }

        const std::uint64_t ts = nowNs();
        const double snapshot_age_ms =
            static_cast<double>(ts - snapshot_ns) / 1000000.0;
        json qpos = vectorHeadJson(snap.resp.position, kActionDim);
        json qvel = vectorHeadJson(snap.resp.velocity, kActionDim);
        json env_state = json::array();
        for (int i = 0; i < kActionDim; ++i) env_state.push_back(snap.resp.position(i));
        for (int i = 0; i < kActionDim; ++i) env_state.push_back(snap.resp.velocity(i));

        json joint_payload{
            {"qpos", qpos},
            {"qvel", qvel},
            {"status", vector12Json(snap.resp.status)},
            {"motor_rpm", vector8Json(snap.resp.motor_rpm)},
            {"plan_rpm", vector8Json(snap.resp.plan_rpm)},
            {"env_state", env_state},
            {"snapshot_age_ms", snapshot_age_ms},
            {"state_loop_tick", snap.meta.loop_tick},
            {"imu_source", options_.imu_simulation ? "placeholder" : "hardware"},
            {"imu_placeholder", options_.imu_simulation},
            {"imu_health", imuHealthJson(snap.resp.imu_health, steadyNs())},
            {"imu_debug", imuDebugJson(snap.resp.imu_debug, steadyNs())},
        };
        json joint_sample{
            {"timestamp_ns", ts},
            {"source", "excavator_api_snapshot"},
            {"receive_time_ns", ts},
            {"payload", joint_payload},
        };

        json fpv_sample{
            {"timestamp_ns", ts},
            {"source", "bridge_placeholder_fpv"},
            {"receive_time_ns", ts},
            {"payload", imagePayload(options_.image_width, options_.image_height, frame_id_++)},
        };

        return responseMessage(
            "read_state.response",
            json{
                {"joint", joint_sample},
                {"images", json{{"fpv", fpv_sample}}},
            });
    }

    void handleClient(int client_fd) {
        std::string buffer;
        bool close_connection = false;
        while (!close_connection && !shutdown_requested_.load()) {
            checkWatchdog();

            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(client_fd, &rfds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 50000;
            const int ready = ::select(client_fd + 1, &rfds, nullptr, nullptr, &tv);
            if (ready < 0) {
                if (errno == EINTR) continue;
                std::cerr << "select failed: " << std::strerror(errno) << "\n";
                break;
            }
            if (ready == 0) {
                continue;
            }

            char chunk[4096];
            const ssize_t n = ::recv(client_fd, chunk, sizeof(chunk), 0);
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) continue;
                std::cerr << "recv failed: " << std::strerror(errno) << "\n";
                break;
            }
            buffer.append(chunk, static_cast<std::size_t>(n));
            while (true) {
                const std::size_t newline = buffer.find('\n');
                if (newline == std::string::npos) break;
                std::string line = buffer.substr(0, newline);
                buffer.erase(0, newline + 1U);
                if (line.empty()) {
                    continue;
                }

                json response;
                try {
                    response = handleMessage(json::parse(line), close_connection);
                } catch (const std::exception& exc) {
                    response = responseMessage("error.response", json::object(), false,
                                               std::string("invalid bridge JSON frame: ") +
                                                   exc.what());
                }
                if (!sendAll(client_fd, response.dump(-1, ' ', false) + "\n")) {
                    close_connection = true;
                    break;
                }
            }
        }
    }

    bool sendZeroCommand(const char* reason) {
        std::lock_guard<std::mutex> lock(control_mutex_);
        return sendZeroCommandLocked(reason);
    }

    bool sendZeroCommandLocked(const char* reason) {
        excavator_api::SpeedScalarCmd zero{};
        const bool ok = control_.sendCommand(zero);
        if (!ok) {
            std::cerr << "zero command failed during " << reason << ": "
                      << control_.lastError() << "\n";
        }
        return ok;
    }

    void checkWatchdog() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        if (!options_.watchdog_enabled || last_valid_action_ns_ == 0 || watchdog_zeroed_) {
            return;
        }
        const std::uint64_t elapsed_ns = nowNs() - last_valid_action_ns_;
        const std::uint64_t timeout_ns =
            static_cast<std::uint64_t>(options_.heartbeat_timeout_ms) * 1000000ULL;
        if (elapsed_ns <= timeout_ns) {
            return;
        }
        (void)sendZeroCommandLocked("watchdog");
        watchdog_zeroed_ = true;
        std::cerr << "watchdog forced zero command after "
                  << (elapsed_ns / 1000000ULL) << " ms without valid action\n";
    }

    Options options_;
    excavator_api::ExcavatorControl control_{};
    excavator_api::ExcavatorReceive receive_{};
    std::mutex control_mutex_{};
    std::thread state_thread_{};
    std::atomic<bool> state_thread_running_{false};
    std::mutex state_mutex_{};
    std::condition_variable state_cv_{};
    excavator_api::Snapshot cached_snapshot_{};
    std::uint64_t cached_snapshot_ns_{0};
    bool has_cached_snapshot_{false};
    std::string last_state_error_{};
    std::uint64_t last_valid_action_ns_{0};
    bool watchdog_zeroed_{true};
    std::uint64_t frame_id_{0};
    std::atomic<bool> shutdown_requested_{false};
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parseArgs(argc, argv);
        BridgeApp app(std::move(options));
        if (!app.startRuntime()) {
            return 1;
        }
        return app.serve();
    } catch (const std::exception& exc) {
        std::cerr << "excavator_real_bridge: " << exc.what() << "\n";
        return 1;
    }
}
