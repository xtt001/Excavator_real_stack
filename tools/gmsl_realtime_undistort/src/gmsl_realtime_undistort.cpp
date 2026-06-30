#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#if defined(GMSL_HAVE_CUDA_WARPING)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CameraConfig {
    std::string camera_key;
    std::string serial;
    std::string device_hint;
    std::string orientation;
    cv::Size image_size;
    cv::Mat K;
    cv::Mat D;
    bool rotate_180{false};
};

struct Options {
    std::string manifest{"configs/camera_intrinsics/gmsl_h190ta/manifest.json"};
    std::string camera_key{"video6"};
    std::string device;
    std::string input_image;
    std::string output_image;
    std::string output_video;
    int width{0};
    int height{0};
    int frames{0};
    double balance{0.0};
    double fov_scale{1.0};
    double output_fps{30.0};
    bool display{false};
    bool prefer_gpu{true};
    bool rotate_from_manifest{true};
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
        << "Usage: gmsl_realtime_undistort [options]\n\n"
        << "Required mode:\n"
        << "  --input-image PATH            Process one image.\n"
        << "  --device /dev/videoN          Process a live V4L2 camera. Defaults to manifest device.\n\n"
        << "Camera/options:\n"
        << "  --manifest PATH               Intrinsics manifest JSON.\n"
        << "  --camera KEY                  Camera key, e.g. video6 or video7.\n"
        << "  --output-image PATH           Output image for --input-image mode.\n"
        << "  --output-video PATH           Optional processed video output for live mode.\n"
        << "  --display                     Show processed frames.\n"
        << "  --frames N                    Stop after N frames in live mode; 0 means until q/Esc.\n"
        << "  --width N --height N          Capture size override.\n"
        << "  --balance X                   OpenCV fisheye balance, 0 crops edges, 1 keeps more FOV.\n"
        << "  --fov-scale X                 OpenCV fisheye fov scale.\n"
        << "  --cpu                         Force CPU remap.\n"
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
            opts.camera_key = valueFor(i, argc, argv, arg);
        } else if (arg == "--device") {
            opts.device = valueFor(i, argc, argv, arg);
        } else if (arg == "--input-image") {
            opts.input_image = valueFor(i, argc, argv, arg);
        } else if (arg == "--output-image") {
            opts.output_image = valueFor(i, argc, argv, arg);
        } else if (arg == "--output-video") {
            opts.output_video = valueFor(i, argc, argv, arg);
        } else if (arg == "--frames") {
            opts.frames = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--width") {
            opts.width = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--height") {
            opts.height = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--balance") {
            opts.balance = std::stod(valueFor(i, argc, argv, arg));
        } else if (arg == "--fov-scale") {
            opts.fov_scale = std::stod(valueFor(i, argc, argv, arg));
        } else if (arg == "--output-fps") {
            opts.output_fps = std::stod(valueFor(i, argc, argv, arg));
        } else if (arg == "--display") {
            opts.display = true;
        } else if (arg == "--cpu") {
            opts.prefer_gpu = false;
        } else if (arg == "--no-rotate") {
            opts.rotate_from_manifest = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.balance < 0.0 || opts.balance > 1.0) {
        throw std::runtime_error("--balance must be in [0, 1]");
    }
    if (opts.fov_scale <= 0.0) {
        throw std::runtime_error("--fov-scale must be positive");
    }
    if ((opts.width > 0) != (opts.height > 0)) {
        throw std::runtime_error("--width and --height must be set together");
    }
    if (opts.width < 0 || opts.height < 0) {
        throw std::runtime_error("--width and --height must be positive");
    }
    return opts;
}

nlohmann::json loadJson(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open manifest: " + path);
    }
    nlohmann::json doc;
    in >> doc;
    return doc;
}

cv::Mat parseK(const nlohmann::json& camera) {
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    const auto& raw = camera.at("K");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            K.at<double>(r, c) = raw.at(r).at(c).get<double>();
        }
    }
    return K;
}

cv::Mat parseD(const nlohmann::json& camera) {
    cv::Mat D(4, 1, CV_64F);
    const auto& raw = camera.at("D");
    for (int i = 0; i < 4; ++i) {
        D.at<double>(i, 0) = raw.at(i).get<double>();
    }
    return D;
}

CameraConfig loadCameraConfig(const Options& opts) {
    const auto doc = loadJson(opts.manifest);
    if (doc.at("distortion_model").get<std::string>() != "opencv_fisheye") {
        throw std::runtime_error("only opencv_fisheye distortion_model is supported");
    }
    for (const auto& camera : doc.at("cameras")) {
        if (camera.at("camera_key").get<std::string>() != opts.camera_key) {
            continue;
        }
        CameraConfig cfg;
        cfg.camera_key = camera.at("camera_key").get<std::string>();
        cfg.serial = camera.at("serial").get<std::string>();
        cfg.device_hint = camera.value("device_hint", "");
        cfg.orientation = camera.value("orientation", "normal");
        cfg.image_size = cv::Size(doc.at("image_width").get<int>(), doc.at("image_height").get<int>());
        cfg.K = parseK(camera);
        cfg.D = parseD(camera);
        cfg.rotate_180 = opts.rotate_from_manifest && cfg.orientation == "rotate_180";
        return cfg;
    }
    throw std::runtime_error("camera key not found in manifest: " + opts.camera_key);
}

CameraConfig scaleCameraForRuntime(const CameraConfig& camera, const Options& opts) {
    if (opts.width <= 0 && opts.height <= 0) {
        return camera;
    }
    CameraConfig scaled = camera;
    const double sx = static_cast<double>(opts.width) / static_cast<double>(camera.image_size.width);
    const double sy = static_cast<double>(opts.height) / static_cast<double>(camera.image_size.height);
    scaled.image_size = cv::Size(opts.width, opts.height);
    scaled.K = camera.K.clone();
    scaled.K.at<double>(0, 0) *= sx;
    scaled.K.at<double>(0, 2) *= sx;
    scaled.K.at<double>(1, 1) *= sy;
    scaled.K.at<double>(1, 2) *= sy;
    return scaled;
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

cv::Mat processCpu(const cv::Mat& frame, const cv::Mat& map_x, const cv::Mat& map_y, bool rotate_180) {
    const cv::Mat bgr = ensureBgr(frame);
    cv::Mat rectified;
    cv::remap(bgr, rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    if (rotate_180) {
        cv::rotate(rectified, rectified, cv::ROTATE_180);
    }
    return rectified;
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

cv::Mat processGpuOrCpu(
    const cv::Mat& frame,
    const cv::Mat& map_x,
    const cv::Mat& map_y,
    bool rotate_180,
    RuntimePipeline* pipeline) {
#if defined(GMSL_HAVE_CUDA_WARPING)
    if (pipeline != nullptr && pipeline->use_gpu) {
        const cv::Mat bgr = ensureBgr(frame);
        pipeline->gpu_frame.upload(bgr, pipeline->stream);
        cv::cuda::remap(
            pipeline->gpu_frame,
            pipeline->gpu_rectified,
            pipeline->gpu_map_x,
            pipeline->gpu_map_y,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(),
            pipeline->stream);
        cv::Mat rectified;
        pipeline->gpu_rectified.download(rectified, pipeline->stream);
        pipeline->stream.waitForCompletion();
        if (rotate_180) {
            cv::rotate(rectified, rectified, cv::ROTATE_180);
        }
        return rectified;
    }
#endif
    return processCpu(frame, map_x, map_y, rotate_180);
}

void initRuntimePipeline(
    const Options& opts,
    const cv::Mat& map_x,
    const cv::Mat& map_y,
    RuntimePipeline* pipeline) {
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

void processImageMode(const Options& opts, const CameraConfig& camera, const cv::Mat& map_x, const cv::Mat& map_y) {
    if (opts.output_image.empty()) {
        throw std::runtime_error("--output-image is required with --input-image");
    }
    const cv::Mat frame = cv::imread(opts.input_image, cv::IMREAD_UNCHANGED);
    if (frame.empty()) {
        throw std::runtime_error("failed to read input image: " + opts.input_image);
    }
    RuntimePipeline pipeline;
    initRuntimePipeline(opts, map_x, map_y, &pipeline);
    const cv::Mat output = processGpuOrCpu(frame, map_x, map_y, camera.rotate_180, &pipeline);
    if (!cv::imwrite(opts.output_image, output)) {
        throw std::runtime_error("failed to write output image: " + opts.output_image);
    }
    std::cout << "processed image camera=" << camera.camera_key << " serial=" << camera.serial
              << " mode=" << (pipeline.use_gpu ? "gpu" : "cpu") << " output=" << opts.output_image << "\n";
}

void openCapture(const Options& opts, const CameraConfig& camera, cv::VideoCapture* cap) {
    const std::string device = opts.device.empty() ? camera.device_hint : opts.device;
    if (device.empty()) {
        throw std::runtime_error("--device is required when manifest has no device_hint");
    }
    cap->open(device, cv::CAP_V4L2);
    if (!cap->isOpened()) {
        throw std::runtime_error("failed to open capture device: " + device);
    }
    const int width = opts.width > 0 ? opts.width : camera.image_size.width;
    const int height = opts.height > 0 ? opts.height : camera.image_size.height;
    cap->set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    cap->set(cv::CAP_PROP_CONVERT_RGB, 1);
}

void processLiveMode(const Options& opts, const CameraConfig& camera, const cv::Mat& map_x, const cv::Mat& map_y) {
    cv::VideoCapture cap;
    openCapture(opts, camera, &cap);

    RuntimePipeline pipeline;
    initRuntimePipeline(opts, map_x, map_y, &pipeline);
    std::cout << "live camera=" << camera.camera_key << " serial=" << camera.serial
              << " mode=" << (pipeline.use_gpu ? "gpu" : "cpu")
              << " rotate_180=" << (camera.rotate_180 ? "true" : "false") << "\n";

    cv::VideoWriter writer;
    bool writer_open = false;
    int count = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to read frame");
        }
        cv::Mat output = processGpuOrCpu(frame, map_x, map_y, camera.rotate_180, &pipeline);
        ++count;

        if (!opts.output_video.empty() && !writer_open) {
            writer.open(
                opts.output_video,
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                opts.output_fps,
                output.size(),
                true);
            if (!writer.isOpened()) {
                throw std::runtime_error("failed to open output video: " + opts.output_video);
            }
            writer_open = true;
        }
        if (writer_open) {
            writer.write(output);
        }
        if (opts.display) {
            cv::imshow("gmsl_realtime_undistort", output);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 27 || key == 'q') {
                break;
            }
        }
        if (opts.frames > 0 && count >= opts.frames) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        const double report_s = std::chrono::duration<double>(now - last_report).count();
        if (report_s >= 2.0) {
            const double total_s = std::chrono::duration<double>(now - t0).count();
            const double fps = static_cast<double>(count) / std::max(total_s, 1e-9);
            std::cout << "frames=" << count << " avg_fps=" << std::fixed << std::setprecision(2)
                      << fps << "\n";
            last_report = now;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);
        const CameraConfig camera = scaleCameraForRuntime(loadCameraConfig(opts), opts);
        cv::Mat map_x;
        cv::Mat map_y;
        buildFisheyeMaps(camera, opts.balance, opts.fov_scale, &map_x, &map_y);

        if (!opts.input_image.empty()) {
            processImageMode(opts, camera, map_x, map_y);
        } else {
            processLiveMode(opts, camera, map_x, map_y);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        std::cerr << "run with --help for usage\n";
        return 2;
    }
}
