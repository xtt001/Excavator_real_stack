#include <excavator_api/excavator_control.hpp>
#include <excavator_api/excavator_receive.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
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
    bool one_shot{false};
    excavator_api::ControlMode control_mode{excavator_api::ControlMode::ClosedLoopVelocityScalar};
};

std::uint64_t nowNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock_t::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
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
        << "  --imu-if <canX>                     IMU CAN interface (default can1)\n"
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
        << "  --one-shot                          stop after one client disconnects\n"
        << "  --help                              show this message\n";
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
        if (::listen(server_fd, 1) != 0) {
            std::cerr << "listen failed: " << std::strerror(errno) << "\n";
            ::close(server_fd);
            return 1;
        }

        std::cerr << "excavator_real_bridge listening on "
                  << options_.host << ":" << options_.port << "\n";

        while (!shutdown_requested_) {
            sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            const int client_fd =
                ::accept(server_fd, reinterpret_cast<sockaddr*>(&peer), &peer_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                std::cerr << "accept failed: " << std::strerror(errno) << "\n";
                break;
            }
            std::cerr << "client connected\n";
            handleClient(client_fd);
            ::close(client_fd);
            std::cerr << "client disconnected\n";
            if (options_.one_shot) {
                break;
            }
        }

        ::close(server_fd);
        (void)sendZeroCommand("shutdown");
        (void)control_.close();
        (void)receive_.close();
        return 0;
    }

private:
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
            if (type == "read_state.request") {
                return handleReadState(payload);
            }
            if (type == "reset.request") {
                (void)sendZeroCommand("reset");
                last_valid_action_ns_ = 0;
                watchdog_zeroed_ = true;
                return responseMessage("reset.response", json{{"reset", true}});
            }
            if (type == "close.request") {
                close_connection = true;
                return responseMessage("close.response", json{{"closed", true}});
            }
            if (type == "shutdown.request") {
                close_connection = true;
                shutdown_requested_ = true;
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

    json handleSendAction(const json& payload) {
        const auto action4 = parseAction4(payload);
        excavator_api::SpeedScalarCmd cmd{};
        for (int i = 0; i < kActionDim; ++i) {
            cmd.speed_scalar(i) = action4[static_cast<std::size_t>(i)];
        }
        for (int i = kActionDim; i < kLowerAxisCount; ++i) {
            cmd.speed_scalar(i) = 0.0;
        }

        const bool sent = control_.sendCommand(cmd);
        const std::uint64_t ts = nowNs();
        last_valid_action_ns_ = ts;
        watchdog_zeroed_ = false;

        json commanded = json::array();
        for (double v : action4) commanded.push_back(v);
        json raw = vector8Json(cmd.speed_scalar);
        return responseMessage(
            "send_action.response",
            json{
                {"ack", sent},
                {"fault_code", sent ? "" : control_.lastError()},
                {"controller_timestamp_ns", ts},
                {"commanded_action", commanded},
                {"raw_low_level_command", raw},
            });
    }

    json handleReadState(const json& payload) {
        const int step_id = payload.value("step_id", 0);
        (void)step_id;
        excavator_api::Snapshot snap{};
        if (!receive_.get(snap, std::chrono::milliseconds(options_.read_timeout_ms))) {
            return responseMessage("read_state.response", json::object(), false,
                                   receive_.lastError());
        }

        const std::uint64_t ts = nowNs();
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
        while (!close_connection && !shutdown_requested_) {
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
        excavator_api::SpeedScalarCmd zero{};
        const bool ok = control_.sendCommand(zero);
        if (!ok) {
            std::cerr << "zero command failed during " << reason << ": "
                      << control_.lastError() << "\n";
        }
        return ok;
    }

    void checkWatchdog() {
        if (!options_.watchdog_enabled || last_valid_action_ns_ == 0 || watchdog_zeroed_) {
            return;
        }
        const std::uint64_t elapsed_ns = nowNs() - last_valid_action_ns_;
        const std::uint64_t timeout_ns =
            static_cast<std::uint64_t>(options_.heartbeat_timeout_ms) * 1000000ULL;
        if (elapsed_ns <= timeout_ns) {
            return;
        }
        (void)sendZeroCommand("watchdog");
        watchdog_zeroed_ = true;
        std::cerr << "watchdog forced zero command after "
                  << (elapsed_ns / 1000000ULL) << " ms without valid action\n";
    }

    Options options_;
    excavator_api::ExcavatorControl control_{};
    excavator_api::ExcavatorReceive receive_{};
    std::uint64_t last_valid_action_ns_{0};
    bool watchdog_zeroed_{true};
    std::uint64_t frame_id_{0};
    bool shutdown_requested_{false};
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
