#include "uyvy_preprocess_kernel.hpp"
#include "v4l2_camera.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

std::atomic<bool> g_stop_requested{false};

void signalHandler(int /*signal*/) {
    g_stop_requested.store(true);
}

int64_t systemNowNs() {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

class CameraShmWriter;

struct CameraRequest {
    std::string key;
    std::string device;
};

struct CameraConfig {
    std::string key;
    std::string device;
    std::string serial;
    std::string mount_position;
    int input_width{1920};
    int input_height{1536};
    int output_width{384};
    int output_height{216};
    cv::Mat K;
    cv::Mat D;
    std::string projection{"virtual_rectilinear"};
    double hfov_deg{110.0};
    double yaw_deg{0.0};
    double pitch_down_deg{0.0};
    double roll_deg{0.0};
};

struct Options {
    std::string manifest{"configs/camera_intrinsics/gmsl_h190ta/manifest.json"};
    std::string preprocess_manifest{"configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json"};
    std::vector<CameraRequest> cameras;
    std::string output_json;
    int width{0};
    int height{0};
    int frames{300};
    int warmup{8};
    int buffers{8};
    int timeout_ms{2000};
    int detail_frames{32};
    bool publish_shm{false};
    std::string shm_prefix{"excavator_gmsl_"};
};

struct Mat3 {
    double m[3][3];
};

struct FrameSample {
    uint32_t sequence{0};
    uint32_t flags{0};
    int64_t v4l2_timestamp_ns{0};
    int64_t host_arrival_mono_ns{0};
    int buffer_index{-1};
    int dmabuf_fd{-1};
    double kernel_ms{0.0};
};

struct CameraRuntime {
    explicit CameraRuntime(CameraConfig camera_config) : config(std::move(camera_config)) {}

    CameraConfig config;
    float* map_x_device{nullptr};
    float* map_y_device{nullptr};
    float* output_device{nullptr};
    unsigned char* output_rgb8_device{nullptr};
    cudaStream_t stream{nullptr};
    std::vector<cudaEvent_t> event_start_by_buffer;
    std::vector<cudaEvent_t> event_stop_by_buffer;
    std::vector<cudaEvent_t> event_copy_done_by_buffer;
    std::vector<unsigned char*> host_rgb_by_buffer;
    std::vector<uint8_t> timing_inflight_by_buffer;
    std::vector<FrameSample> timing_sample_by_buffer;
    std::deque<std::size_t> timing_inflight_order;
    CameraShmWriter* shm_writer{nullptr};

    mutable std::mutex mutex;
    std::vector<double> kernel_ms;
    std::vector<int64_t> v4l2_timestamps_ns;
    std::vector<int64_t> host_arrival_ns;
    std::vector<FrameSample> samples;
    uint64_t processed_frames{0};
    uint64_t shm_publish_count{0};
    uint64_t cuda_missing_frames{0};
    uint64_t kernel_failure_count{0};
    std::string error_message;
};

class CameraShmWriter {
public:
    CameraShmWriter() = default;
    CameraShmWriter(std::string name, int width, int height)
        : name_(std::move(name)), width_(width), height_(height) {
        if (width_ <= 0 || height_ <= 0) {
            throw std::runtime_error("invalid shm image size");
        }
        if (imageBytes() > maxImageBytes()) {
            throw std::runtime_error("shm image size exceeds fpv_shm compatibility limit");
        }
        const std::string path = "/dev/shm/" + name_;
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) {
            throw std::runtime_error("open shm " + path + " failed");
        }
        if (ftruncate(fd_, static_cast<off_t>(totalBytes())) != 0) {
            throw std::runtime_error("ftruncate shm " + path + " failed");
        }
        ptr_ = static_cast<unsigned char*>(
            mmap(nullptr, totalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            throw std::runtime_error("mmap shm " + path + " failed");
        }
        writeU32(0, 0x46505631U);
        writeU32(4, 2U);
    }

    CameraShmWriter(const CameraShmWriter&) = delete;
    CameraShmWriter& operator=(const CameraShmWriter&) = delete;

    CameraShmWriter(CameraShmWriter&& other) noexcept {
        moveFrom(other);
    }

    CameraShmWriter& operator=(CameraShmWriter&& other) noexcept {
        if (this != &other) {
            close();
            moveFrom(other);
        }
        return *this;
    }

    ~CameraShmWriter() {
        close();
    }

    void writeRgb(
        const unsigned char* rgb,
        int64_t sync_timestamp_ns,
        int64_t receive_time_ns,
        int64_t v4l2_timestamp_ns,
        uint32_t sequence,
        uint32_t flags) {
        if (ptr_ == nullptr || rgb == nullptr) {
            return;
        }
        const uint32_t nbytes = static_cast<uint32_t>(imageBytes());
        writeU64(8, static_cast<uint64_t>(v4l2_timestamp_ns));
        writeU32(16, flags);
        writeU64(48, static_cast<uint64_t>(sync_timestamp_ns));
        writeU64(56, static_cast<uint64_t>(receive_time_ns));
        writeU32(64, sequence);
        writeU32(68, static_cast<uint32_t>(width_));
        writeU32(72, static_cast<uint32_t>(height_));
        writeU32(76, nbytes);
        std::memcpy(ptr_ + headerBytes(), rgb, imageBytes());
    }

    const std::string& name() const {
        return name_;
    }

private:
    static constexpr std::size_t headerBytes() { return 80U; }
    static constexpr std::size_t maxImageBytes() { return 640U * 480U * 3U; }
    std::size_t imageBytes() const {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 3U;
    }
    std::size_t totalBytes() const {
        return headerBytes() + maxImageBytes();
    }
    void writeU32(std::size_t offset, uint32_t value) {
        std::memcpy(ptr_ + offset, &value, sizeof(value));
    }
    void writeU64(std::size_t offset, uint64_t value) {
        std::memcpy(ptr_ + offset, &value, sizeof(value));
    }
    void close() {
        if (ptr_ != nullptr) {
            munmap(ptr_, totalBytes());
            ptr_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }
    void moveFrom(CameraShmWriter& other) {
        name_ = std::move(other.name_);
        width_ = other.width_;
        height_ = other.height_;
        fd_ = other.fd_;
        ptr_ = other.ptr_;
        other.fd_ = -1;
        other.ptr_ = nullptr;
    }

    std::string name_;
    int width_{0};
    int height_{0};
    int fd_{-1};
    unsigned char* ptr_{nullptr};
};

std::string valueFor(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    return std::string(argv[++index]);
}

CameraRequest parseCameraRequest(const std::string& raw) {
    const std::size_t pos = raw.find('=');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= raw.size()) {
        throw std::runtime_error("--camera must use KEY=/dev/videoN");
    }
    return CameraRequest{raw.substr(0, pos), raw.substr(pos + 1)};
}

void usage() {
    std::cout
        << "Usage: gmsl_realtime_preprocess_probe [options]\n"
        << "  --camera KEY=/dev/videoN        Override camera list/order. Default uses preprocess camera_order.\n"
        << "  --manifest PATH                 Intrinsics manifest JSON.\n"
        << "  --preprocess-manifest PATH      Preprocess manifest JSON.\n"
        << "  --width N --height N            Capture size override, with intrinsics scaled.\n"
        << "  --frames N                      Measured frames per camera. Default: 300.\n"
        << "  --warmup N                      Warmup frames per camera. Default: 8.\n"
        << "  --buffers N                     V4L2 DMABUF buffer count. Default: 8.\n"
        << "  --timeout-ms N                  select() timeout. Default: 2000.\n"
        << "  --detail-frames N               Per-camera frame samples in JSON. Default: 32.\n"
        << "  --publish-shm                   Publish derived RGB8 frames to /dev/shm.\n"
        << "  --shm-prefix PREFIX             SHM name prefix. Default: excavator_gmsl_.\n"
        << "  --output-json PATH              Write JSON report.\n";
}

Options parseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else if (arg == "--camera") {
            opts.cameras.push_back(parseCameraRequest(valueFor(i, argc, argv, arg)));
        } else if (arg == "--manifest") {
            opts.manifest = valueFor(i, argc, argv, arg);
        } else if (arg == "--preprocess-manifest") {
            opts.preprocess_manifest = valueFor(i, argc, argv, arg);
        } else if (arg == "--width") {
            opts.width = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--height") {
            opts.height = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--frames") {
            opts.frames = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--warmup") {
            opts.warmup = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--buffers") {
            opts.buffers = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--timeout-ms") {
            opts.timeout_ms = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--detail-frames") {
            opts.detail_frames = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--publish-shm") {
            opts.publish_shm = true;
        } else if (arg == "--shm-prefix") {
            opts.shm_prefix = valueFor(i, argc, argv, arg);
        } else if (arg == "--output-json") {
            opts.output_json = valueFor(i, argc, argv, arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if ((opts.width > 0) != (opts.height > 0)) {
        throw std::runtime_error("--width and --height must be set together");
    }
    if (opts.width < 0 || opts.height < 0 || opts.frames < 0 || opts.warmup < 0 ||
        opts.buffers < 2 || opts.timeout_ms <= 0 || opts.detail_frames < 0) {
        throw std::runtime_error("invalid numeric option");
    }
    return opts;
}

Json loadJson(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open JSON: " + path);
    }
    Json doc;
    in >> doc;
    return doc;
}

cv::Mat parseK(const Json& camera) {
    cv::Mat K(3, 3, CV_64F);
    const Json& raw = camera.at("K");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            K.at<double>(r, c) = raw.at(r).at(c).get<double>();
        }
    }
    return K;
}

cv::Mat parseD(const Json& camera) {
    cv::Mat D(4, 1, CV_64F);
    const Json& raw = camera.at("D");
    for (int i = 0; i < 4; ++i) {
        D.at<double>(i, 0) = raw.at(i).get<double>();
    }
    return D;
}

double jsonNumberOr(const Json& doc, const std::string& key, double default_value) {
    if (!doc.contains(key) || doc.at(key).is_null()) {
        return default_value;
    }
    return doc.at(key).get<double>();
}

const Json* findCameraJson(const Json& cameras, const std::string& key) {
    for (const auto& camera : cameras) {
        if (camera.at("camera_key").get<std::string>() == key) {
            return &camera;
        }
    }
    return nullptr;
}

std::vector<CameraRequest> defaultCameraRequestsFromPreprocess(const Json& preprocess) {
    std::vector<CameraRequest> requests;
    for (const auto& key_value : preprocess.at("camera_order")) {
        const std::string key = key_value.get<std::string>();
        const Json* camera = findCameraJson(preprocess.at("cameras"), key);
        if (camera == nullptr) {
            throw std::runtime_error("camera_order references missing preprocess camera: " + key);
        }
        requests.push_back(CameraRequest{key, camera->value("device_hint", "/dev/" + key)});
    }
    return requests;
}

CameraConfig scaleCameraForRuntime(const CameraConfig& camera, int width, int height) {
    if (width <= 0 && height <= 0) {
        return camera;
    }
    CameraConfig scaled = camera;
    const double sx = static_cast<double>(width) / static_cast<double>(camera.input_width);
    const double sy = static_cast<double>(height) / static_cast<double>(camera.input_height);
    scaled.input_width = width;
    scaled.input_height = height;
    scaled.K = camera.K.clone();
    scaled.K.at<double>(0, 0) *= sx;
    scaled.K.at<double>(0, 2) *= sx;
    scaled.K.at<double>(1, 1) *= sy;
    scaled.K.at<double>(1, 2) *= sy;
    return scaled;
}

std::vector<CameraConfig> loadCameraConfigs(const Options& opts) {
    const Json intrinsics = loadJson(opts.manifest);
    const Json preprocess = loadJson(opts.preprocess_manifest);
    if (intrinsics.at("distortion_model").get<std::string>() != "opencv_fisheye") {
        throw std::runtime_error("only opencv_fisheye intrinsics are supported");
    }
    const Json& output = preprocess.at("output");
    if (output.value("color_space", "RGB") != "RGB" || output.value("layout", "HWC") != "HWC") {
        throw std::runtime_error("preprocess output must be RGB HWC");
    }

    const std::vector<CameraRequest> requests =
        opts.cameras.empty() ? defaultCameraRequestsFromPreprocess(preprocess) : opts.cameras;
    std::vector<CameraConfig> configs;
    configs.reserve(requests.size());
    for (const CameraRequest& request : requests) {
        const Json* intr = findCameraJson(intrinsics.at("cameras"), request.key);
        const Json* prep = findCameraJson(preprocess.at("cameras"), request.key);
        if (intr == nullptr) {
            throw std::runtime_error("camera key missing from intrinsics manifest: " + request.key);
        }
        if (prep == nullptr) {
            throw std::runtime_error("camera key missing from preprocess manifest: " + request.key);
        }
        const Json& transform = prep->at("transform");
        const std::string projection = transform.at("projection").get<std::string>();
        if (projection != "virtual_rectilinear") {
            throw std::runtime_error("unsupported preprocess projection for " + request.key + ": " + projection);
        }

        CameraConfig cfg;
        cfg.key = request.key;
        cfg.device = request.device;
        cfg.serial = intr->value("serial", "");
        cfg.mount_position = prep->value("mount_position", "");
        cfg.input_width = intrinsics.at("image_width").get<int>();
        cfg.input_height = intrinsics.at("image_height").get<int>();
        cfg.output_width = output.at("width").get<int>();
        cfg.output_height = output.at("height").get<int>();
        cfg.K = parseK(*intr);
        cfg.D = parseD(*intr);
        cfg.projection = projection;
        cfg.hfov_deg = transform.at("hfov_deg").get<double>();
        cfg.yaw_deg = jsonNumberOr(transform, "yaw_deg", 0.0);
        cfg.pitch_down_deg = jsonNumberOr(transform, "pitch_down_deg", 0.0);
        cfg.roll_deg = jsonNumberOr(transform, "roll_deg", 0.0);
        configs.push_back(scaleCameraForRuntime(cfg, opts.width, opts.height));
    }
    return configs;
}

Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                out.m[r][c] += a.m[r][k] * b.m[k][c];
            }
        }
    }
    return out;
}

double degToRad(double degrees) {
    return degrees * CV_PI / 180.0;
}

Mat3 rotationX(double degrees) {
    const double t = degToRad(degrees);
    const double c = std::cos(t);
    const double s = std::sin(t);
    return Mat3{{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}};
}

Mat3 rotationY(double degrees) {
    const double t = degToRad(degrees);
    const double c = std::cos(t);
    const double s = std::sin(t);
    return Mat3{{{c, 0.0, s}, {0.0, 1.0, 0.0}, {-s, 0.0, c}}};
}

Mat3 rotationZ(double degrees) {
    const double t = degToRad(degrees);
    const double c = std::cos(t);
    const double s = std::sin(t);
    return Mat3{{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
}

void buildVirtualRectilinearMaps(const CameraConfig& camera, cv::Mat* map_x, cv::Mat* map_y) {
    if (camera.output_width <= 0 || camera.output_height <= 0) {
        throw std::runtime_error("invalid preprocess output size for " + camera.key);
    }
    if (camera.hfov_deg <= 0.0 || camera.hfov_deg >= 180.0) {
        throw std::runtime_error("invalid hfov_deg for " + camera.key);
    }
    map_x->create(camera.output_height, camera.output_width, CV_32FC1);
    map_y->create(camera.output_height, camera.output_width, CV_32FC1);

    const double fx = camera.K.at<double>(0, 0);
    const double fy = camera.K.at<double>(1, 1);
    const double cx_in = camera.K.at<double>(0, 2);
    const double cy_in = camera.K.at<double>(1, 2);
    const double k1 = camera.D.at<double>(0, 0);
    const double k2 = camera.D.at<double>(1, 0);
    const double k3 = camera.D.at<double>(2, 0);
    const double k4 = camera.D.at<double>(3, 0);

    const double focal = (static_cast<double>(camera.output_width) * 0.5) /
                         std::tan(degToRad(camera.hfov_deg) * 0.5);
    const double cx_out = (static_cast<double>(camera.output_width) - 1.0) * 0.5;
    const double cy_out = (static_cast<double>(camera.output_height) - 1.0) * 0.5;
    const Mat3 r = multiply(
        multiply(rotationZ(camera.roll_deg), rotationY(camera.yaw_deg)),
        rotationX(-camera.pitch_down_deg));

    for (int y = 0; y < camera.output_height; ++y) {
        for (int x = 0; x < camera.output_width; ++x) {
            const double vx = (static_cast<double>(x) - cx_out) / focal;
            const double vy = (static_cast<double>(y) - cy_out) / focal;
            const double vz = 1.0;

            const double sx = r.m[0][0] * vx + r.m[0][1] * vy + r.m[0][2] * vz;
            const double sy = r.m[1][0] * vx + r.m[1][1] * vy + r.m[1][2] * vz;
            const double sz = r.m[2][0] * vx + r.m[2][1] * vy + r.m[2][2] * vz;
            const double xn = sx / sz;
            const double yn = sy / sz;
            const double radius = std::sqrt(xn * xn + yn * yn);

            double scale = 1.0;
            if (radius > 1e-12) {
                const double theta = std::atan(radius);
                const double theta2 = theta * theta;
                const double theta4 = theta2 * theta2;
                const double theta6 = theta4 * theta2;
                const double theta8 = theta4 * theta4;
                const double theta_d =
                    theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
                scale = theta_d / radius;
            }

            map_x->at<float>(y, x) = static_cast<float>(fx * xn * scale + cx_in);
            map_y->at<float>(y, x) = static_cast<float>(fy * yn * scale + cy_in);
        }
    }
}

void checkCuda(cudaError_t status, const char* name) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(status));
    }
}

Json summarize(std::vector<double> values) {
    if (values.empty()) {
        return Json{{"count", 0}, {"mean", nullptr}, {"p50", nullptr}, {"p95", nullptr}, {"p99", nullptr}, {"max", nullptr}};
    }
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    auto percentile = [&](double q) {
        const double pos = (static_cast<double>(values.size()) - 1.0) * q;
        const auto lo = static_cast<std::size_t>(std::floor(pos));
        const auto hi = std::min(lo + 1, values.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    };
    return Json{
        {"count", values.size()},
        {"mean", sum / static_cast<double>(values.size())},
        {"p50", percentile(0.50)},
        {"p95", percentile(0.95)},
        {"p99", percentile(0.99)},
        {"max", values.back()},
    };
}

std::vector<double> timestampDeltasMs(const std::vector<int64_t>& timestamps) {
    std::vector<double> deltas;
    if (timestamps.size() < 2) {
        return deltas;
    }
    deltas.reserve(timestamps.size() - 1);
    for (std::size_t i = 1; i < timestamps.size(); ++i) {
        deltas.push_back(static_cast<double>(timestamps[i] - timestamps[i - 1]) / 1e6);
    }
    return deltas;
}

Json flagsJson(uint32_t flags) {
    return Json{
        {"raw", flags},
        {"error", hasV4L2Error(flags)},
        {"timestamp_clock", timestampClock(flags)},
        {"timestamp_source", timestampSource(flags)},
    };
}

Json statsToJson(const V4L2CameraStats& stats) {
    return Json{
        {"camera_key", stats.camera_key},
        {"device", stats.device},
        {"opened", stats.opened},
        {"finished", stats.finished},
        {"streaming", stats.streaming},
        {"width", stats.width},
        {"height", stats.height},
        {"fourcc", fourccToString(stats.fourcc)},
        {"memory_type", stats.memory_type},
        {"requested_buffers", stats.requested_buffers},
        {"actual_buffers", stats.actual_buffers},
        {"frames_seen", stats.frames_seen},
        {"frames_reported", stats.frames_reported},
        {"warmup_discarded", stats.warmup_discarded},
        {"drop_count", stats.drop_count},
        {"error_flag_count", stats.error_flag_count},
        {"bytes_mismatch_count", stats.bytes_mismatch_count},
        {"sequence_reset_count", stats.sequence_reset_count},
        {"timeout_count", stats.timeout_count},
        {"dmabuf_fd_count", stats.dmabuf_fd_count},
        {"cuda_import_success_count", stats.cuda_import_success_count},
        {"cuda_import_failure_count", stats.cuda_import_failure_count},
        {"deferred_requeue_count", stats.deferred_requeue_count},
        {"deferred_requeue_wait_count", stats.deferred_requeue_wait_count},
        {"deferred_requeue_max_pending", stats.deferred_requeue_max_pending},
        {"first_sequence", stats.first_sequence},
        {"last_sequence", stats.last_sequence},
        {"first_flags", flagsJson(stats.first_flags)},
        {"last_flags", flagsJson(stats.last_flags)},
        {"first_timestamp_ns", stats.first_timestamp_ns},
        {"last_timestamp_ns", stats.last_timestamp_ns},
        {"timestamp_clock", stats.timestamp_clock},
        {"timestamp_source", stats.timestamp_source},
        {"cuda_frame_type", stats.cuda_frame_type},
        {"cuda_pitch", stats.cuda_pitch},
        {"cuda_plane_count", stats.cuda_plane_count},
        {"error_message", stats.error_message},
    };
}

Json sampleToJson(const FrameSample& sample) {
    return Json{
        {"sequence", sample.sequence},
        {"flags", flagsJson(sample.flags)},
        {"v4l2_timestamp_ns", sample.v4l2_timestamp_ns},
        {"host_arrival_mono_ns", sample.host_arrival_mono_ns},
        {"buffer_index", sample.buffer_index},
        {"dmabuf_fd", sample.dmabuf_fd},
        {"kernel_ms", sample.kernel_ms},
    };
}

Json syncReport(const std::vector<std::unique_ptr<CameraRuntime>>& runtimes) {
    std::size_t min_count = std::numeric_limits<std::size_t>::max();
    std::map<std::string, std::vector<int64_t>> timestamps_by_camera;
    for (const auto& runtime : runtimes) {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        timestamps_by_camera[runtime->config.key] = runtime->v4l2_timestamps_ns;
        min_count = std::min(min_count, runtime->v4l2_timestamps_ns.size());
    }
    if (min_count == std::numeric_limits<std::size_t>::max()) {
        min_count = 0;
    }

    std::map<std::string, std::vector<double>> skew_by_camera;
    for (std::size_t i = 0; i < min_count; ++i) {
        int64_t target_ns = std::numeric_limits<int64_t>::min();
        for (const auto& runtime : runtimes) {
            target_ns = std::max(target_ns, timestamps_by_camera.at(runtime->config.key).at(i));
        }
        for (const auto& runtime : runtimes) {
            const int64_t ts = timestamps_by_camera.at(runtime->config.key).at(i);
            skew_by_camera[runtime->config.key].push_back(static_cast<double>(ts - target_ns) / 1e6);
        }
    }

    Json skew = Json::object();
    for (const auto& item : skew_by_camera) {
        skew[item.first] = summarize(item.second);
    }
    return Json{
        {"strategy", "offline nth-frame timestamp grouping; target_t=max(v4l2 timestamps)"},
        {"sample_count", min_count},
        {"skew_ms", skew},
    };
}

void allocateCameraGpuState(CameraRuntime* runtime, int buffer_count, bool publish_shm) {
    cv::Mat map_x;
    cv::Mat map_y;
    buildVirtualRectilinearMaps(runtime->config, &map_x, &map_y);
    const std::size_t map_bytes =
        static_cast<std::size_t>(runtime->config.output_width) *
        static_cast<std::size_t>(runtime->config.output_height) * sizeof(float);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&runtime->map_x_device), map_bytes), "cudaMalloc map_x");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&runtime->map_y_device), map_bytes), "cudaMalloc map_y");
    checkCuda(cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    runtime->event_start_by_buffer.resize(buffer_count, nullptr);
    runtime->event_stop_by_buffer.resize(buffer_count, nullptr);
    runtime->event_copy_done_by_buffer.resize(buffer_count, nullptr);
    runtime->host_rgb_by_buffer.resize(buffer_count, nullptr);
    runtime->timing_inflight_by_buffer.resize(buffer_count, 0);
    runtime->timing_sample_by_buffer.resize(buffer_count);
    for (int i = 0; i < buffer_count; ++i) {
        checkCuda(cudaEventCreate(&runtime->event_start_by_buffer[i]), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&runtime->event_stop_by_buffer[i]), "cudaEventCreate stop");
        if (publish_shm) {
            checkCuda(cudaEventCreate(&runtime->event_copy_done_by_buffer[i]), "cudaEventCreate copy_done");
        }
    }
    if (publish_shm) {
        const std::size_t rgb_bytes =
            static_cast<std::size_t>(runtime->config.output_width) *
            static_cast<std::size_t>(runtime->config.output_height) * 3U;
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&runtime->output_rgb8_device), rgb_bytes),
            "cudaMalloc output_rgb8");
        for (int i = 0; i < buffer_count; ++i) {
            checkCuda(
                cudaMallocHost(reinterpret_cast<void**>(&runtime->host_rgb_by_buffer[i]), rgb_bytes),
                "cudaMallocHost host_rgb");
        }
    }
    checkCuda(cudaMemcpyAsync(runtime->map_x_device, map_x.ptr<float>(), map_bytes, cudaMemcpyHostToDevice, runtime->stream),
              "cudaMemcpyAsync map_x");
    checkCuda(cudaMemcpyAsync(runtime->map_y_device, map_y.ptr<float>(), map_bytes, cudaMemcpyHostToDevice, runtime->stream),
              "cudaMemcpyAsync map_y");
    checkCuda(cudaStreamSynchronize(runtime->stream), "cudaStreamSynchronize maps");
}

void cleanupCameraGpuState(CameraRuntime* runtime) {
    if (runtime == nullptr) {
        return;
    }
    for (unsigned char* host_rgb : runtime->host_rgb_by_buffer) {
        if (host_rgb != nullptr) {
            cudaFreeHost(host_rgb);
        }
    }
    runtime->host_rgb_by_buffer.clear();
    if (runtime->output_rgb8_device != nullptr) {
        cudaFree(runtime->output_rgb8_device);
        runtime->output_rgb8_device = nullptr;
    }
    for (cudaEvent_t event : runtime->event_copy_done_by_buffer) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
    runtime->event_copy_done_by_buffer.clear();
    for (cudaEvent_t event : runtime->event_stop_by_buffer) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
    runtime->event_stop_by_buffer.clear();
    for (cudaEvent_t event : runtime->event_start_by_buffer) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
    runtime->event_start_by_buffer.clear();
    if (runtime->stream != nullptr) {
        cudaStreamDestroy(runtime->stream);
        runtime->stream = nullptr;
    }
    if (runtime->map_y_device != nullptr) {
        cudaFree(runtime->map_y_device);
        runtime->map_y_device = nullptr;
    }
    if (runtime->map_x_device != nullptr) {
        cudaFree(runtime->map_x_device);
        runtime->map_x_device = nullptr;
    }
}

void collectCompletedTimings(CameraRuntime* runtime, bool wait_all, int detail_frames) {
    while (!runtime->timing_inflight_order.empty()) {
        const std::size_t i = runtime->timing_inflight_order.front();
        if (i >= runtime->timing_inflight_by_buffer.size() || runtime->timing_inflight_by_buffer[i] == 0) {
            runtime->timing_inflight_order.pop_front();
            continue;
        }
        cudaEvent_t completion_event = runtime->event_stop_by_buffer[i];
        if (runtime->shm_writer != nullptr && i < runtime->event_copy_done_by_buffer.size() &&
            runtime->event_copy_done_by_buffer[i] != nullptr) {
            completion_event = runtime->event_copy_done_by_buffer[i];
        }
        const cudaError_t status =
            wait_all ? cudaEventSynchronize(completion_event)
                     : cudaEventQuery(completion_event);
        if (!wait_all && status == cudaErrorNotReady) {
            break;
        }
        checkCuda(status, wait_all ? "cudaEventSynchronize timing" : "cudaEventQuery timing");
        float elapsed_ms = 0.0f;
        checkCuda(
            cudaEventElapsedTime(&elapsed_ms, runtime->event_start_by_buffer[i], runtime->event_stop_by_buffer[i]),
            "cudaEventElapsedTime");

        const FrameSample sample = runtime->timing_sample_by_buffer[i];
        runtime->timing_inflight_by_buffer[i] = 0;
        runtime->timing_inflight_order.pop_front();

        bool published = false;
        if (runtime->shm_writer != nullptr && i < runtime->host_rgb_by_buffer.size() &&
            runtime->host_rgb_by_buffer[i] != nullptr) {
            const int64_t now_ns = systemNowNs();
            runtime->shm_writer->writeRgb(
                runtime->host_rgb_by_buffer[i],
                now_ns,
                now_ns,
                sample.v4l2_timestamp_ns,
                sample.sequence,
                sample.flags);
            published = true;
        }

        std::lock_guard<std::mutex> lock(runtime->mutex);
        runtime->processed_frames += 1;
        if (published) {
            runtime->shm_publish_count += 1;
        }
        runtime->kernel_ms.push_back(static_cast<double>(elapsed_ms));
        runtime->v4l2_timestamps_ns.push_back(sample.v4l2_timestamp_ns);
        runtime->host_arrival_ns.push_back(sample.host_arrival_mono_ns);
        if (static_cast<int>(runtime->samples.size()) < detail_frames) {
            FrameSample timed_sample = sample;
            timed_sample.kernel_ms = static_cast<double>(elapsed_ms);
            runtime->samples.push_back(timed_sample);
        }
    }
}

CameraFrameReleaseFence processFrameOnGpu(CameraRuntime* runtime, const CameraFrame& frame, int detail_frames) {
    collectCompletedTimings(runtime, false, detail_frames);
    if (!frame.cuda_imported || frame.cuda_device_ptr == nullptr || frame.cuda_frame_type != "pitch") {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        runtime->cuda_missing_frames += 1;
        return CameraFrameReleaseFence{};
    }

    try {
        if (frame.buffer_index < 0 ||
            static_cast<std::size_t>(frame.buffer_index) >= runtime->event_stop_by_buffer.size()) {
            throw std::runtime_error("invalid frame buffer index for CUDA event fence");
        }
        const std::size_t buffer_index = static_cast<std::size_t>(frame.buffer_index);
        if (runtime->timing_inflight_by_buffer[buffer_index] != 0) {
            collectCompletedTimings(runtime, true, detail_frames);
            if (runtime->timing_inflight_by_buffer[buffer_index] != 0) {
                throw std::runtime_error("buffer reused before previous CUDA timing was collected");
            }
        }

        const int pitch_bytes = frame.cuda_pitch > 0 ? static_cast<int>(frame.cuda_pitch) : frame.width * 2;
        checkCuda(cudaEventRecord(runtime->event_start_by_buffer[buffer_index], runtime->stream), "cudaEventRecord start");
        launchUyvyRemapNormalizeAndRgb8(
            static_cast<const unsigned char*>(frame.cuda_device_ptr),
            frame.width,
            frame.height,
            pitch_bytes,
            runtime->map_x_device,
            runtime->map_y_device,
            runtime->config.output_width,
            runtime->config.output_height,
            runtime->output_device,
            runtime->output_rgb8_device,
            runtime->stream);
        checkCuda(cudaGetLastError(), "launchUyvyRemapNormalizeAndRgb8");
        checkCuda(cudaEventRecord(runtime->event_stop_by_buffer[buffer_index], runtime->stream), "cudaEventRecord stop");
        if (runtime->shm_writer != nullptr) {
            const std::size_t rgb_bytes =
                static_cast<std::size_t>(runtime->config.output_width) *
                static_cast<std::size_t>(runtime->config.output_height) * 3U;
            checkCuda(
                cudaMemcpyAsync(
                    runtime->host_rgb_by_buffer[buffer_index],
                    runtime->output_rgb8_device,
                    rgb_bytes,
                    cudaMemcpyDeviceToHost,
                    runtime->stream),
                "cudaMemcpyAsync output_rgb8");
            checkCuda(
                cudaEventRecord(runtime->event_copy_done_by_buffer[buffer_index], runtime->stream),
                "cudaEventRecord copy_done");
        }
        runtime->timing_sample_by_buffer[buffer_index] = FrameSample{
            frame.sequence,
            frame.flags,
            frame.v4l2_timestamp_ns,
            frame.host_arrival_mono_ns,
            frame.buffer_index,
            frame.dmabuf_fd,
            0.0,
        };
        runtime->timing_inflight_by_buffer[buffer_index] = 1;
        runtime->timing_inflight_order.push_back(buffer_index);
        return CameraFrameReleaseFence{
            CameraFrameReleaseFenceType::CudaEvent,
            reinterpret_cast<void*>(runtime->event_stop_by_buffer[buffer_index]),
        };
    } catch (const std::exception& exc) {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        runtime->kernel_failure_count += 1;
        runtime->error_message = exc.what();
        throw;
    }
}

Json cameraRuntimeToJson(const CameraRuntime& runtime, const V4L2CameraStats& stats, std::size_t camera_index) {
    std::vector<double> kernel_ms;
    std::vector<int64_t> v4l2_timestamps;
    std::vector<int64_t> host_arrivals;
    std::vector<FrameSample> samples;
    uint64_t processed_frames = 0;
    uint64_t shm_publish_count = 0;
    uint64_t cuda_missing_frames = 0;
    uint64_t kernel_failure_count = 0;
    std::string runtime_error;
    {
        std::lock_guard<std::mutex> lock(runtime.mutex);
        kernel_ms = runtime.kernel_ms;
        v4l2_timestamps = runtime.v4l2_timestamps_ns;
        host_arrivals = runtime.host_arrival_ns;
        samples = runtime.samples;
        processed_frames = runtime.processed_frames;
        shm_publish_count = runtime.shm_publish_count;
        cuda_missing_frames = runtime.cuda_missing_frames;
        kernel_failure_count = runtime.kernel_failure_count;
        runtime_error = runtime.error_message;
    }

    Json sample_json = Json::array();
    for (const FrameSample& sample : samples) {
        sample_json.push_back(sampleToJson(sample));
    }

    return Json{
        {"camera_key", runtime.config.key},
        {"device", runtime.config.device},
        {"serial", runtime.config.serial},
        {"mount_position", runtime.config.mount_position},
        {"camera_index", camera_index},
        {"input_width", runtime.config.input_width},
        {"input_height", runtime.config.input_height},
        {"output_width", runtime.config.output_width},
        {"output_height", runtime.config.output_height},
        {"projection", runtime.config.projection},
        {"hfov_deg", runtime.config.hfov_deg},
        {"yaw_deg", runtime.config.yaw_deg},
        {"pitch_down_deg", runtime.config.pitch_down_deg},
        {"roll_deg", runtime.config.roll_deg},
        {"processed_frames", processed_frames},
        {"shm_publish_count", shm_publish_count},
        {"shm_name", runtime.shm_writer != nullptr ? runtime.shm_writer->name() : ""},
        {"cuda_missing_frames", cuda_missing_frames},
        {"kernel_failure_count", kernel_failure_count},
        {"kernel_ms", summarize(kernel_ms)},
        {"timestamp_interval_ms", summarize(timestampDeltasMs(v4l2_timestamps))},
        {"host_arrival_interval_ms", summarize(timestampDeltasMs(host_arrivals))},
        {"output_device_ptr", reinterpret_cast<std::uintptr_t>(runtime.output_device)},
        {"stats", statsToJson(stats)},
        {"runtime_error", runtime_error},
        {"samples", sample_json},
    };
}

}  // namespace

int main(int argc, char** argv) {
    float* output_tensor = nullptr;
    try {
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        const Options opts = parseArgs(argc, argv);
        checkCuda(cudaSetDevice(0), "cudaSetDevice");
        checkCuda(cudaFree(nullptr), "cudaFree(0)");

        const std::vector<CameraConfig> configs = loadCameraConfigs(opts);
        if (configs.empty()) {
            throw std::runtime_error("no cameras configured");
        }
        const int output_width = configs.front().output_width;
        const int output_height = configs.front().output_height;
        for (const CameraConfig& cfg : configs) {
            if (cfg.output_width != output_width || cfg.output_height != output_height) {
                throw std::runtime_error("all cameras must share one output size");
            }
        }

        const std::size_t per_camera_elements =
            static_cast<std::size_t>(output_width) * static_cast<std::size_t>(output_height) * 3U;
        const std::size_t output_elements = per_camera_elements * configs.size();
        const std::size_t output_bytes = output_elements * sizeof(float);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&output_tensor), output_bytes), "cudaMalloc output tensor");
        checkCuda(cudaMemset(output_tensor, 0, output_bytes), "cudaMemset output tensor");

        std::vector<std::unique_ptr<CameraRuntime>> runtimes;
        runtimes.reserve(configs.size());
        for (std::size_t i = 0; i < configs.size(); ++i) {
            auto runtime = std::make_unique<CameraRuntime>(configs.at(i));
            runtime->output_device = output_tensor + i * per_camera_elements;
            allocateCameraGpuState(runtime.get(), opts.buffers, opts.publish_shm);
            runtimes.push_back(std::move(runtime));
        }

        std::map<std::string, CameraShmWriter> shm_writers;
        if (opts.publish_shm) {
            for (auto& runtime : runtimes) {
                const std::string shm_name = opts.shm_prefix + runtime->config.key;
                auto inserted = shm_writers.emplace(
                    runtime->config.key,
                    CameraShmWriter(shm_name, runtime->config.output_width, runtime->config.output_height));
                runtime->shm_writer = &inserted.first->second;
            }
        }

        std::vector<std::unique_ptr<V4L2Camera>> cameras;
        cameras.reserve(runtimes.size());
        for (const auto& runtime : runtimes) {
            V4L2CameraConfig config;
            config.camera_key = runtime->config.key;
            config.device = runtime->config.device;
            config.width = runtime->config.input_width;
            config.height = runtime->config.input_height;
            config.buffer_count = opts.buffers;
            config.warmup_frames = opts.warmup;
            config.capture_frames = opts.frames;
            config.poll_timeout_ms = opts.timeout_ms;
            config.memory_mode = V4L2MemoryMode::Dmabuf;
            config.cuda_import_probe = true;

            CameraRuntime* runtime_ptr = runtime.get();
            cameras.push_back(std::make_unique<V4L2Camera>(
                config,
                [runtime_ptr, detail_frames = opts.detail_frames](const CameraFrame& frame) {
                    return processFrameOnGpu(runtime_ptr, frame, detail_frames);
                }));
        }

        const int64_t start_ns = monotonicNowNs();
        for (auto& camera : cameras) {
            camera->start();
        }
        if (opts.frames == 0) {
            while (!g_stop_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            for (auto& camera : cameras) {
                camera->stop();
            }
        }
        for (auto& camera : cameras) {
            camera->join();
        }
        const int64_t end_ns = monotonicNowNs();
        for (auto& runtime : runtimes) {
            collectCompletedTimings(runtime.get(), true, opts.detail_frames);
        }

        Json report{
            {"version", "gmsl_realtime_preprocess_probe_zero_copy_20260701"},
            {"zero_copy_contract", Json{
                {"capture_memory", "V4L2_MEMORY_DMABUF"},
                {"capture_buffer_allocator", "NvBufSurfaceAllocate"},
                {"gpu_import", "NvBufSurfaceMapEglImage + cuGraphicsEGLRegisterImage"},
                {"input_cpu_frame_copy", false},
                {"intermediate_rgba_or_bgr", false},
                {"output_cpu_download", false},
                {"derived_rgb8_shm_publish", opts.publish_shm},
                {"callback_waits_for_kernel_before_qbuf", false},
                {"qbuf_release_fence", "cuda_event"},
            }},
            {"manifest", opts.manifest},
            {"preprocess_manifest", opts.preprocess_manifest},
            {"frames_per_camera", opts.frames},
            {"warmup", opts.warmup},
            {"buffers", opts.buffers},
            {"timeout_ms", opts.timeout_ms},
            {"publish_shm", opts.publish_shm},
            {"shm_prefix", opts.shm_prefix},
            {"start_host_mono_ns", start_ns},
            {"end_host_mono_ns", end_ns},
            {"elapsed_ms", static_cast<double>(end_ns - start_ns) / 1e6},
            {"output_tensor", Json{
                {"device_ptr", reinterpret_cast<std::uintptr_t>(output_tensor)},
                {"dtype", "float32"},
                {"layout", "NHWC"},
                {"camera_order", Json::array()},
                {"shape", Json::array({configs.size(), output_height, output_width, 3})},
                {"bytes", output_bytes},
            }},
            {"cameras", Json::object()},
        };

        bool ok = true;
        std::vector<double> all_kernel_ms;
        for (std::size_t i = 0; i < runtimes.size(); ++i) {
            const V4L2CameraStats stats = cameras.at(i)->stats();
            Json camera_report = cameraRuntimeToJson(*runtimes.at(i), stats, i);
            report["output_tensor"]["camera_order"].push_back(runtimes.at(i)->config.key);
            report["cameras"][runtimes.at(i)->config.key] = camera_report;

            std::vector<double> camera_kernel_ms;
            {
                std::lock_guard<std::mutex> lock(runtimes.at(i)->mutex);
                camera_kernel_ms = runtimes.at(i)->kernel_ms;
            }
            all_kernel_ms.insert(all_kernel_ms.end(), camera_kernel_ms.begin(), camera_kernel_ms.end());

            const uint64_t processed = camera_report.value("processed_frames", 0ULL);
            const uint64_t cuda_missing = camera_report.value("cuda_missing_frames", 0ULL);
            const uint64_t kernel_failures = camera_report.value("kernel_failure_count", 0ULL);
            if (!stats.error_message.empty() ||
                (opts.frames > 0 && processed < static_cast<uint64_t>(opts.frames)) ||
                cuda_missing > 0 || kernel_failures > 0) {
                ok = false;
            }

            const Json& kernel = camera_report.at("kernel_ms");
            std::cout << runtimes.at(i)->config.key
                      << " device=" << runtimes.at(i)->config.device
                      << " processed=" << processed
                      << " drops=" << stats.drop_count
                      << " errors=" << stats.error_flag_count
                      << " cuda_ok=" << stats.cuda_import_success_count
                      << " cuda_fail=" << stats.cuda_import_failure_count
                      << " missing=" << cuda_missing
                      << " shm_published=" << camera_report.value("shm_publish_count", 0ULL)
                      << " kernel_p50_ms=" << kernel.value("p50", 0.0)
                      << " kernel_p95_ms=" << kernel.value("p95", 0.0)
                      << " kernel_p99_ms=" << kernel.value("p99", 0.0)
                      << " pitch_down_deg=" << runtimes.at(i)->config.pitch_down_deg
                      << "\n";
        }
        report["kernel_ms_all_cameras"] = summarize(all_kernel_ms);
        report["sync"] = syncReport(runtimes);
        report["ok"] = ok;

        if (!opts.output_json.empty()) {
            std::ofstream out(opts.output_json);
            if (!out) {
                throw std::runtime_error("failed to open output JSON: " + opts.output_json);
            }
            out << std::setw(2) << report << "\n";
        }

        for (auto& runtime : runtimes) {
            cleanupCameraGpuState(runtime.get());
        }
        cudaFree(output_tensor);
        output_tensor = nullptr;
        return ok ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        if (output_tensor != nullptr) {
            cudaFree(output_tensor);
        }
        return 1;
    }
}
