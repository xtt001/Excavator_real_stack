#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#if defined(GMSL_HAVE_CUDA_WARPING)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct CameraRequest {
    std::string key;
    std::string device;
    bool raw_only{false};
};

struct CameraConfig {
    std::string key;
    std::string device;
    std::string serial;
    bool raw_only{false};
    bool rotate_180{false};
    cv::Size image_size;
    cv::Mat K;
    cv::Mat D;
};

struct Options {
    std::string manifest{"configs/camera_intrinsics/gmsl_h190ta/manifest.json"};
    std::vector<CameraRequest> cameras;
    std::string output_json;
    int width{0};
    int height{0};
    int frames{300};
    int warmup{30};
    int buffer_count{1};
    double balance{0.0};
    double fov_scale{1.0};
    bool prefer_gpu{true};
    bool capture_only{false};
    bool rotate_from_manifest{true};
};

struct StageSamples {
    std::vector<double> read_ms;
    std::vector<double> color_ms;
    std::vector<double> upload_ms;
    std::vector<double> remap_ms;
    std::vector<double> download_ms;
    std::vector<double> rotate_ms;
    std::vector<double> process_ms;
    std::vector<double> frame_ms;
};

struct CameraResult {
    CameraConfig config;
    StageSamples samples;
    std::string error;
    int frames_seen{0};
    int warmup_frames{0};
    int measured_frames{0};
    int read_failures{0};
    bool used_gpu{false};
};

struct RuntimePipeline {
    bool use_gpu{false};
#if defined(GMSL_HAVE_CUDA_WARPING)
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat gpu_map_x;
    cv::cuda::GpuMat gpu_map_y;
    cv::cuda::GpuMat gpu_rectified;
    cv::cuda::Stream stream;
#endif
};

void usage() {
    std::cout
        << "Usage: gmsl_latency_benchmark [options]\n\n"
        << "Cameras:\n"
        << "  --camera KEY=/dev/videoN      Calibrated camera from manifest; repeat for multi-camera.\n"
        << "  --raw-camera NAME=/dev/videoN Capture-only camera without intrinsics; repeat as needed.\n\n"
        << "Benchmark options:\n"
        << "  --manifest PATH               Intrinsics manifest JSON.\n"
        << "  --frames N                    Measured frame count per camera after warmup. Default: 300.\n"
        << "  --warmup N                    Warmup frames per camera before measuring. Default: 30.\n"
        << "  --width N --height N          Capture size override.\n"
        << "  --buffer-count N              V4L2 capture buffer count hint. Default: 1.\n"
        << "  --balance X                   OpenCV fisheye balance. Default: 0.0.\n"
        << "  --fov-scale X                 OpenCV fisheye fov scale. Default: 1.0.\n"
        << "  --output-json PATH            Write machine-readable latency report.\n"
        << "  --capture-only                Measure dequeue/capture only; no color convert or remap.\n"
        << "  --cpu                         Force CPU remap even if OpenCV CUDA is available.\n"
        << "  --no-rotate                   Ignore manifest rotate_180 orientation.\n"
        << "  --help                        Show this help.\n";
}

std::string valueFor(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    ++index;
    return std::string(argv[index]);
}

CameraRequest parseCameraRequest(const std::string& raw, bool raw_only) {
    const std::size_t pos = raw.find('=');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= raw.size()) {
        throw std::runtime_error("camera mapping must be NAME=/dev/videoN: " + raw);
    }
    return CameraRequest{raw.substr(0, pos), raw.substr(pos + 1), raw_only};
}

Options parseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else if (arg == "--manifest") {
            opts.manifest = valueFor(i, argc, argv, arg);
        } else if (arg == "--camera") {
            opts.cameras.push_back(parseCameraRequest(valueFor(i, argc, argv, arg), false));
        } else if (arg == "--raw-camera") {
            opts.cameras.push_back(parseCameraRequest(valueFor(i, argc, argv, arg), true));
        } else if (arg == "--frames") {
            opts.frames = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--warmup") {
            opts.warmup = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--width") {
            opts.width = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--height") {
            opts.height = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--buffer-count") {
            opts.buffer_count = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--balance") {
            opts.balance = std::stod(valueFor(i, argc, argv, arg));
        } else if (arg == "--fov-scale") {
            opts.fov_scale = std::stod(valueFor(i, argc, argv, arg));
        } else if (arg == "--output-json") {
            opts.output_json = valueFor(i, argc, argv, arg);
        } else if (arg == "--capture-only") {
            opts.capture_only = true;
        } else if (arg == "--cpu") {
            opts.prefer_gpu = false;
        } else if (arg == "--no-rotate") {
            opts.rotate_from_manifest = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.cameras.empty()) {
        throw std::runtime_error("at least one --camera or --raw-camera is required");
    }
    if (opts.frames <= 0) {
        throw std::runtime_error("--frames must be positive");
    }
    if (opts.warmup < 0) {
        throw std::runtime_error("--warmup must be non-negative");
    }
    if ((opts.width > 0) != (opts.height > 0)) {
        throw std::runtime_error("--width and --height must be set together");
    }
    if (opts.width < 0 || opts.height < 0) {
        throw std::runtime_error("--width and --height must be positive");
    }
    if (opts.buffer_count <= 0) {
        throw std::runtime_error("--buffer-count must be positive");
    }
    if (opts.balance < 0.0 || opts.balance > 1.0) {
        throw std::runtime_error("--balance must be in [0, 1]");
    }
    if (opts.fov_scale <= 0.0) {
        throw std::runtime_error("--fov-scale must be positive");
    }
    for (const CameraRequest& request : opts.cameras) {
        if (request.raw_only && !opts.capture_only) {
            throw std::runtime_error("--raw-camera requires --capture-only");
        }
    }
    return opts;
}

Json loadJson(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open manifest: " + path);
    }
    Json doc;
    in >> doc;
    return doc;
}

cv::Mat parseK(const Json& camera) {
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    const auto& raw = camera.at("K");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            K.at<double>(r, c) = raw.at(r).at(c).get<double>();
        }
    }
    return K;
}

cv::Mat parseD(const Json& camera) {
    cv::Mat D(4, 1, CV_64F);
    const auto& raw = camera.at("D");
    for (int i = 0; i < 4; ++i) {
        D.at<double>(i, 0) = raw.at(i).get<double>();
    }
    return D;
}

CameraConfig scaleCameraForRuntime(const CameraConfig& camera, const Options& opts) {
    if (opts.width <= 0 && opts.height <= 0) {
        return camera;
    }
    CameraConfig scaled = camera;
    scaled.image_size = cv::Size(opts.width, opts.height);
    if (!camera.raw_only) {
        const double sx = static_cast<double>(opts.width) / static_cast<double>(camera.image_size.width);
        const double sy = static_cast<double>(opts.height) / static_cast<double>(camera.image_size.height);
        scaled.K = camera.K.clone();
        scaled.K.at<double>(0, 0) *= sx;
        scaled.K.at<double>(0, 2) *= sx;
        scaled.K.at<double>(1, 1) *= sy;
        scaled.K.at<double>(1, 2) *= sy;
    }
    return scaled;
}

CameraConfig loadCalibratedCamera(const Json& manifest, const CameraRequest& request, const Options& opts) {
    if (manifest.at("distortion_model").get<std::string>() != "opencv_fisheye") {
        throw std::runtime_error("only opencv_fisheye distortion_model is supported");
    }
    for (const auto& camera : manifest.at("cameras")) {
        if (camera.at("camera_key").get<std::string>() != request.key) {
            continue;
        }
        CameraConfig cfg;
        cfg.key = request.key;
        cfg.device = request.device;
        cfg.serial = camera.at("serial").get<std::string>();
        cfg.raw_only = false;
        cfg.rotate_180 = opts.rotate_from_manifest && camera.value("orientation", "normal") == "rotate_180";
        cfg.image_size = cv::Size(manifest.at("image_width").get<int>(), manifest.at("image_height").get<int>());
        cfg.K = parseK(camera);
        cfg.D = parseD(camera);
        return scaleCameraForRuntime(cfg, opts);
    }
    throw std::runtime_error("camera key not found in manifest: " + request.key);
}

std::vector<CameraConfig> loadCameraConfigs(const Options& opts) {
    Json manifest;
    bool manifest_loaded = false;
    std::vector<CameraConfig> configs;
    for (const CameraRequest& request : opts.cameras) {
        if (request.raw_only) {
            CameraConfig cfg;
            cfg.key = request.key;
            cfg.device = request.device;
            cfg.serial = "";
            cfg.raw_only = true;
            cfg.rotate_180 = false;
            cfg.image_size = opts.width > 0 ? cv::Size(opts.width, opts.height) : cv::Size();
            configs.push_back(cfg);
            continue;
        }
        if (!manifest_loaded) {
            manifest = loadJson(opts.manifest);
            manifest_loaded = true;
        }
        configs.push_back(loadCalibratedCamera(manifest, request, opts));
    }
    return configs;
}

void buildFisheyeMaps(
    const CameraConfig& camera,
    double balance,
    double fov_scale,
    cv::Mat* map_x,
    cv::Mat* map_y) {
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat new_K;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        camera.K, camera.D, camera.image_size, R, new_K, balance, camera.image_size, fov_scale);
    cv::fisheye::initUndistortRectifyMap(
        camera.K, camera.D, R, new_K, camera.image_size, CV_32FC1, *map_x, *map_y);
}

cv::Mat ensureBgr(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::runtime_error("empty frame");
    }
    cv::Mat bgr;
    if (frame.channels() == 3) {
        bgr = frame;
    } else if (frame.channels() == 2) {
        cv::cvtColor(frame, bgr, cv::COLOR_YUV2BGR_UYVY);
    } else if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else {
        throw std::runtime_error("unsupported frame channel count: " + std::to_string(frame.channels()));
    }
    return bgr;
}

bool cudaRemapAvailable() {
#if defined(GMSL_HAVE_CUDA_WARPING)
    try {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    } catch (const cv::Exception&) {
        return false;
    }
#else
    return false;
#endif
}

void initPipeline(const Options& opts, const cv::Mat& map_x, const cv::Mat& map_y, RuntimePipeline* pipeline) {
    if (pipeline == nullptr) {
        return;
    }
#if defined(GMSL_HAVE_CUDA_WARPING)
    pipeline->use_gpu = opts.prefer_gpu && cudaRemapAvailable();
    if (pipeline->use_gpu) {
        pipeline->gpu_map_x.upload(map_x, pipeline->stream);
        pipeline->gpu_map_y.upload(map_y, pipeline->stream);
        pipeline->stream.waitForCompletion();
    }
#else
    (void)opts;
    (void)map_x;
    (void)map_y;
#endif
}

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void openCapture(const CameraConfig& camera, const Options& opts, cv::VideoCapture* cap) {
    cap->open(camera.device, cv::CAP_V4L2);
    if (!cap->isOpened()) {
        throw std::runtime_error("failed to open capture device: " + camera.device);
    }
    if (opts.width > 0 && opts.height > 0) {
        cap->set(cv::CAP_PROP_FRAME_WIDTH, opts.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, opts.height);
    } else if (!camera.raw_only && camera.image_size.width > 0 && camera.image_size.height > 0) {
        cap->set(cv::CAP_PROP_FRAME_WIDTH, camera.image_size.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, camera.image_size.height);
    }
    cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    cap->set(cv::CAP_PROP_CONVERT_RGB, 0);
    cap->set(cv::CAP_PROP_BUFFERSIZE, opts.buffer_count);
}

void addMeasuredSample(
    StageSamples* samples,
    double read_ms,
    double color_ms,
    double upload_ms,
    double remap_ms,
    double download_ms,
    double rotate_ms,
    double process_ms,
    double frame_ms) {
    samples->read_ms.push_back(read_ms);
    samples->color_ms.push_back(color_ms);
    samples->upload_ms.push_back(upload_ms);
    samples->remap_ms.push_back(remap_ms);
    samples->download_ms.push_back(download_ms);
    samples->rotate_ms.push_back(rotate_ms);
    samples->process_ms.push_back(process_ms);
    samples->frame_ms.push_back(frame_ms);
}

void runCameraBenchmark(const CameraConfig& camera, const Options& opts, CameraResult* result) {
    result->config = camera;
    cv::VideoCapture cap;
    try {
        cv::Mat map_x;
        cv::Mat map_y;
        RuntimePipeline pipeline;
        if (!opts.capture_only) {
            buildFisheyeMaps(camera, opts.balance, opts.fov_scale, &map_x, &map_y);
            initPipeline(opts, map_x, map_y, &pipeline);
            result->used_gpu = pipeline.use_gpu;
        }

        openCapture(camera, opts, &cap);
        const int target_total = opts.warmup + opts.frames;
        for (int i = 0; i < target_total; ++i) {
            const bool measured = i >= opts.warmup;
            const auto frame_start = Clock::now();

            cv::Mat frame;
            const auto read_start = Clock::now();
            const bool ok = cap.read(frame);
            const auto read_end = Clock::now();
            if (!ok || frame.empty()) {
                ++result->read_failures;
                continue;
            }
            ++result->frames_seen;
            if (!measured) {
                ++result->warmup_frames;
            }

            const double read_ms = elapsedMs(read_start, read_end);
            double color_ms = 0.0;
            double upload_ms = 0.0;
            double remap_ms = 0.0;
            double download_ms = 0.0;
            double rotate_ms = 0.0;
            double process_ms = 0.0;

            if (!opts.capture_only) {
                const auto process_start = Clock::now();

                const auto color_start = Clock::now();
                const cv::Mat bgr = ensureBgr(frame);
                const auto color_end = Clock::now();
                color_ms = elapsedMs(color_start, color_end);

#if defined(GMSL_HAVE_CUDA_WARPING)
                if (pipeline.use_gpu) {
                    const auto upload_start = Clock::now();
                    pipeline.gpu_frame.upload(bgr, pipeline.stream);
                    pipeline.stream.waitForCompletion();
                    const auto upload_end = Clock::now();
                    upload_ms = elapsedMs(upload_start, upload_end);

                    const auto remap_start = Clock::now();
                    cv::cuda::remap(
                        pipeline.gpu_frame,
                        pipeline.gpu_rectified,
                        pipeline.gpu_map_x,
                        pipeline.gpu_map_y,
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(),
                        pipeline.stream);
                    pipeline.stream.waitForCompletion();
                    const auto remap_end = Clock::now();
                    remap_ms = elapsedMs(remap_start, remap_end);

                    cv::Mat rectified;
                    const auto download_start = Clock::now();
                    pipeline.gpu_rectified.download(rectified, pipeline.stream);
                    pipeline.stream.waitForCompletion();
                    const auto download_end = Clock::now();
                    download_ms = elapsedMs(download_start, download_end);

                    if (camera.rotate_180) {
                        const auto rotate_start = Clock::now();
                        cv::rotate(rectified, rectified, cv::ROTATE_180);
                        const auto rotate_end = Clock::now();
                        rotate_ms = elapsedMs(rotate_start, rotate_end);
                    }
                } else
#endif
                {
                    cv::Mat rectified;
                    const auto remap_start = Clock::now();
                    cv::remap(bgr, rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
                    const auto remap_end = Clock::now();
                    remap_ms = elapsedMs(remap_start, remap_end);

                    if (camera.rotate_180) {
                        const auto rotate_start = Clock::now();
                        cv::rotate(rectified, rectified, cv::ROTATE_180);
                        const auto rotate_end = Clock::now();
                        rotate_ms = elapsedMs(rotate_start, rotate_end);
                    }
                }
                const auto process_end = Clock::now();
                process_ms = elapsedMs(process_start, process_end);
            }

            const auto frame_end = Clock::now();
            if (measured) {
                ++result->measured_frames;
                addMeasuredSample(
                    &result->samples,
                    read_ms,
                    color_ms,
                    upload_ms,
                    remap_ms,
                    download_ms,
                    rotate_ms,
                    process_ms,
                    elapsedMs(frame_start, frame_end));
            }
        }
    } catch (const std::exception& exc) {
        result->error = exc.what();
    }
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) {
        return values.at(lo);
    }
    const double weight = pos - static_cast<double>(lo);
    return values.at(lo) * (1.0 - weight) + values.at(hi) * weight;
}

Json summarize(const std::vector<double>& values) {
    Json out;
    out["count"] = values.size();
    if (values.empty()) {
        out["mean"] = 0.0;
        out["p50"] = 0.0;
        out["p95"] = 0.0;
        out["p99"] = 0.0;
        out["max"] = 0.0;
        return out;
    }
    double sum = 0.0;
    double max_value = values.front();
    for (double value : values) {
        sum += value;
        max_value = std::max(max_value, value);
    }
    out["mean"] = sum / static_cast<double>(values.size());
    out["p50"] = percentile(values, 0.50);
    out["p95"] = percentile(values, 0.95);
    out["p99"] = percentile(values, 0.99);
    out["max"] = max_value;
    return out;
}

Json samplesToJson(const StageSamples& samples) {
    return Json{
        {"read_ms", summarize(samples.read_ms)},
        {"color_ms", summarize(samples.color_ms)},
        {"upload_ms", summarize(samples.upload_ms)},
        {"remap_ms", summarize(samples.remap_ms)},
        {"download_ms", summarize(samples.download_ms)},
        {"rotate_ms", summarize(samples.rotate_ms)},
        {"process_ms", summarize(samples.process_ms)},
        {"frame_ms", summarize(samples.frame_ms)},
    };
}

Json resultToJson(const CameraResult& result) {
    Json out;
    out["camera_key"] = result.config.key;
    out["device"] = result.config.device;
    out["serial"] = result.config.serial;
    out["raw_only"] = result.config.raw_only;
    out["rotate_180"] = result.config.rotate_180;
    out["used_gpu"] = result.used_gpu;
    out["frames_seen"] = result.frames_seen;
    out["warmup_frames"] = result.warmup_frames;
    out["measured_frames"] = result.measured_frames;
    out["read_failures"] = result.read_failures;
    out["error"] = result.error;
    out["metrics"] = samplesToJson(result.samples);
    return out;
}

Json reportToJson(const Options& opts, const std::vector<CameraResult>& results) {
    Json report;
    report["version"] = "gmsl_latency_benchmark_20260630";
    report["latency_boundary"] =
        "cap.read returned buffer to preprocessing output ready; display, recording, policy forward, and control are excluded";
    report["manifest"] = opts.manifest;
    report["frames"] = opts.frames;
    report["warmup"] = opts.warmup;
    report["capture_only"] = opts.capture_only;
    report["prefer_gpu"] = opts.prefer_gpu;
    report["width"] = opts.width;
    report["height"] = opts.height;
    report["balance"] = opts.balance;
    report["fov_scale"] = opts.fov_scale;
    report["cameras"] = Json::array();
    for (const CameraResult& result : results) {
        report["cameras"].push_back(resultToJson(result));
    }
    return report;
}

void printSummary(const std::vector<CameraResult>& results) {
    for (const CameraResult& result : results) {
        const Json process = summarize(result.samples.process_ms);
        const Json frame = summarize(result.samples.frame_ms);
        std::cout << result.config.key << " device=" << result.config.device
                  << " measured=" << result.measured_frames
                  << " mode=" << (result.used_gpu ? "gpu" : "cpu_or_capture_only");
        if (!result.error.empty()) {
            std::cout << " error=\"" << result.error << "\"";
        }
        std::cout << " process_ms p50/p95/p99/max=" << std::fixed << std::setprecision(3)
                  << process.at("p50").get<double>() << "/" << process.at("p95").get<double>() << "/"
                  << process.at("p99").get<double>() << "/" << process.at("max").get<double>()
                  << " frame_ms p50/p95/p99/max=" << frame.at("p50").get<double>() << "/"
                  << frame.at("p95").get<double>() << "/" << frame.at("p99").get<double>() << "/"
                  << frame.at("max").get<double>() << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);
        const std::vector<CameraConfig> configs = loadCameraConfigs(opts);

        std::vector<CameraResult> results(configs.size());
        std::vector<std::thread> workers;
        workers.reserve(configs.size());
        for (std::size_t i = 0; i < configs.size(); ++i) {
            workers.emplace_back(runCameraBenchmark, configs.at(i), opts, &results.at(i));
        }
        for (std::thread& worker : workers) {
            worker.join();
        }

        printSummary(results);
        if (!opts.output_json.empty()) {
            std::ofstream out(opts.output_json);
            if (!out) {
                throw std::runtime_error("failed to open output json: " + opts.output_json);
            }
            out << reportToJson(opts, results).dump(2) << "\n";
        }

        for (const CameraResult& result : results) {
            if (!result.error.empty()) {
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 2;
    }
}
