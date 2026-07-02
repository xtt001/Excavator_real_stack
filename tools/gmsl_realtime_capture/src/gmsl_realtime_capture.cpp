#include "timestamp_ring.hpp"
#include "v4l2_camera.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
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
#include <vector>

namespace {

using Json = nlohmann::json;

struct CameraSpec {
    std::string key;
    std::string device;
};

struct Options {
    int width{1920};
    int height{1536};
    int frames{3000};
    int warmup{0};
    int buffers{4};
    int ring_size{8};
    int timeout_ms{2000};
    int detail_frames{-1};
    int sync_sample_limit{240};
    bool frame_details{true};
    V4L2MemoryMode memory_mode{V4L2MemoryMode::Mmap};
    bool cuda_import_probe{false};
    std::string output_json;
    std::vector<CameraSpec> cameras{
        {"video4", "/dev/video4"},
        {"video5", "/dev/video5"},
        {"video6", "/dev/video6"},
        {"video7", "/dev/video7"},
    };
};

struct FrameRecord {
    CameraFrame frame;
    std::optional<double> inter_frame_delta_ms;
};

struct CaptureStore {
    explicit CaptureStore(int ring_capacity) : ring(ring_capacity) {}
    TimestampRing<CameraFrame> ring;
    std::mutex mutex;
    std::vector<FrameRecord> frames;
};

std::string valueFor(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    return argv[++i];
}

CameraSpec parseCameraSpec(const std::string& value) {
    const std::size_t pos = value.find('=');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= value.size()) {
        throw std::runtime_error("--camera must use KEY=/dev/videoN");
    }
    return CameraSpec{value.substr(0, pos), value.substr(pos + 1)};
}

V4L2MemoryMode parseMemoryMode(const std::string& value) {
    if (value == "mmap") {
        return V4L2MemoryMode::Mmap;
    }
    if (value == "dmabuf") {
        return V4L2MemoryMode::Dmabuf;
    }
    throw std::runtime_error("--memory must be mmap or dmabuf");
}

std::string memoryModeName(V4L2MemoryMode mode) {
    return mode == V4L2MemoryMode::Dmabuf ? "dmabuf" : "mmap";
}

Options parseArgs(int argc, char** argv) {
    Options opts;
    bool camera_overridden = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--camera") {
            if (!camera_overridden) {
                opts.cameras.clear();
                camera_overridden = true;
            }
            opts.cameras.push_back(parseCameraSpec(valueFor(i, argc, argv, arg)));
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
        } else if (arg == "--ring-size") {
            opts.ring_size = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--timeout-ms") {
            opts.timeout_ms = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--detail-frames") {
            opts.detail_frames = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--sync-sample-limit") {
            opts.sync_sample_limit = std::stoi(valueFor(i, argc, argv, arg));
        } else if (arg == "--memory") {
            opts.memory_mode = parseMemoryMode(valueFor(i, argc, argv, arg));
        } else if (arg == "--cuda-import-probe") {
            opts.cuda_import_probe = true;
        } else if (arg == "--output-json") {
            opts.output_json = valueFor(i, argc, argv, arg);
        } else if (arg == "--no-frame-details") {
            opts.frame_details = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: gmsl_realtime_capture [options]\n"
                << "  --camera KEY=/dev/videoN       Add a camera. Default: video4..video7.\n"
                << "  --width N --height N           Capture size. Default: 1920x1536.\n"
                << "  --frames N                     Reported frames per camera. Default: 3000.\n"
                << "  --warmup N                     Dequeue and discard N startup frames. Default: 0.\n"
                << "  --buffers N                    V4L2 MMAP buffer count. Default: 4.\n"
                << "  --ring-size N                  Per-camera latest ring size. Default: 8.\n"
                << "  --timeout-ms N                 select() timeout. Default: 2000.\n"
                << "  --detail-frames N              Metadata frames per camera in JSON; -1 keeps all.\n"
                << "  --sync-sample-limit N          Max sync sample rows in JSON. Default: 240.\n"
                << "  --memory mmap|dmabuf           Capture memory mode. Default: mmap.\n"
                << "  --cuda-import-probe            In dmabuf mode, import each buffer to CUDA/EGL.\n"
                << "  --no-frame-details             Keep only summaries and sync stats.\n"
                << "  --output-json PATH             Write JSON report.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.cameras.empty()) {
        throw std::runtime_error("at least one --camera is required");
    }
    if (opts.width <= 0 || opts.height <= 0 || opts.frames <= 0 || opts.warmup < 0 ||
        opts.buffers < 2 || opts.ring_size < 1 || opts.timeout_ms <= 0 ||
        opts.detail_frames < -1 || opts.sync_sample_limit < 0) {
        throw std::runtime_error("invalid numeric option");
    }
    return opts;
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

Json flagsJson(uint32_t flags) {
    return Json{
        {"raw", flags},
        {"error", hasV4L2Error(flags)},
        {"timestamp_clock", timestampClock(flags)},
        {"timestamp_source", timestampSource(flags)},
    };
}

Json frameToJson(const FrameRecord& record) {
    const CameraFrame& frame = record.frame;
    Json out{
        {"camera_key", frame.camera_key},
        {"device", frame.device},
        {"memory_type", frame.memory_type},
        {"sequence", frame.sequence},
        {"flags", flagsJson(frame.flags)},
        {"v4l2_timestamp_ns", frame.v4l2_timestamp_ns},
        {"timestamp_clock", timestampClock(frame.flags)},
        {"timestamp_source", timestampSource(frame.flags)},
        {"host_arrival_mono_ns", frame.host_arrival_mono_ns},
        {"width", frame.width},
        {"height", frame.height},
        {"fourcc", fourccToString(frame.fourcc)},
        {"buffer_index", frame.buffer_index},
        {"dmabuf_fd", frame.dmabuf_fd},
        {"surface_pitch", frame.surface_pitch},
        {"surface_plane_count", frame.surface_plane_count},
        {"surface_data_size", frame.surface_data_size},
        {"cuda_imported", frame.cuda_imported},
        {"cuda_device_ptr", reinterpret_cast<std::uintptr_t>(frame.cuda_device_ptr)},
        {"cuda_pitch", frame.cuda_pitch},
        {"cuda_plane_count", frame.cuda_plane_count},
        {"cuda_frame_type", frame.cuda_frame_type},
        {"bytesused", frame.bytesused},
        {"data_ptr_valid_only_during_callback", true},
    };
    if (record.inter_frame_delta_ms.has_value()) {
        out["inter_frame_delta_ms"] = record.inter_frame_delta_ms.value();
    } else {
        out["inter_frame_delta_ms"] = nullptr;
    }
    return out;
}

std::vector<double> timestampDeltas(const std::vector<FrameRecord>& frames, bool host) {
    std::vector<double> deltas;
    if (frames.size() < 2) {
        return deltas;
    }
    deltas.reserve(frames.size() - 1);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const int64_t current = host ? frames[i].frame.host_arrival_mono_ns : frames[i].frame.v4l2_timestamp_ns;
        const int64_t previous = host ? frames[i - 1].frame.host_arrival_mono_ns : frames[i - 1].frame.v4l2_timestamp_ns;
        deltas.push_back(static_cast<double>(current - previous) / 1e6);
    }
    return deltas;
}

std::optional<CameraFrame> nearestFrameAtOrBeforeTimestamp(const std::vector<FrameRecord>& records, int64_t target_ns) {
    if (records.empty()) {
        return std::nullopt;
    }
    auto best = records.end();
    int64_t best_diff = std::numeric_limits<int64_t>::max();
    for (auto it = records.begin(); it != records.end(); ++it) {
        if (it->frame.v4l2_timestamp_ns > target_ns) {
            continue;
        }
        const int64_t diff = target_ns - it->frame.v4l2_timestamp_ns;
        if (diff < best_diff) {
            best_diff = diff;
            best = it;
        }
    }
    if (best == records.end()) {
        best = std::min_element(
            records.begin(),
            records.end(),
            [&](const FrameRecord& lhs, const FrameRecord& rhs) {
                return std::llabs(lhs.frame.v4l2_timestamp_ns - target_ns) <
                       std::llabs(rhs.frame.v4l2_timestamp_ns - target_ns);
            });
    }
    return best->frame;
}

Json syncReport(const std::vector<CameraSpec>& specs,
                const std::map<std::string, std::vector<FrameRecord>>& frames_by_camera,
                int sample_limit) {
    std::size_t min_count = std::numeric_limits<std::size_t>::max();
    for (const CameraSpec& spec : specs) {
        const auto found = frames_by_camera.find(spec.key);
        const std::size_t count = found == frames_by_camera.end() ? 0 : found->second.size();
        min_count = std::min(min_count, count);
    }
    if (min_count == std::numeric_limits<std::size_t>::max()) {
        min_count = 0;
    }

    std::map<std::string, std::vector<double>> skew_by_camera;
    Json samples = Json::array();
    for (std::size_t i = 0; i < min_count; ++i) {
        int64_t target_ns = std::numeric_limits<int64_t>::min();
        for (const CameraSpec& spec : specs) {
            const auto& records = frames_by_camera.at(spec.key);
            target_ns = std::max(target_ns, records[i].frame.v4l2_timestamp_ns);
        }

        Json sample{
            {"index", i},
            {"target_timestamp_ns", target_ns},
            {"strategy", "target=max_nth_camera_timestamp; choose nearest available frame at or before target"},
            {"cameras", Json::object()},
        };
        for (const CameraSpec& spec : specs) {
            const auto& records = frames_by_camera.at(spec.key);
            const auto nearest = nearestFrameAtOrBeforeTimestamp(records, target_ns);
            if (!nearest.has_value()) {
                continue;
            }
            const double skew_ms = static_cast<double>(nearest->v4l2_timestamp_ns - target_ns) / 1e6;
            skew_by_camera[spec.key].push_back(skew_ms);
            if (static_cast<int>(samples.size()) < sample_limit) {
                sample["cameras"][spec.key] = Json{
                    {"sequence", nearest->sequence},
                    {"v4l2_timestamp_ns", nearest->v4l2_timestamp_ns},
                    {"cross_camera_skew_ms", skew_ms},
                    {"flags", flagsJson(nearest->flags)},
                };
            }
        }
        if (static_cast<int>(samples.size()) < sample_limit) {
            samples.push_back(sample);
        }
    }

    Json skew = Json::object();
    for (const auto& item : skew_by_camera) {
        skew[item.first] = summarize(item.second);
    }
    return Json{
        {"strategy", "offline timestamp grouping using target_t=max(nth timestamps), nearest available frame at or before target per camera"},
        {"sample_count", min_count},
        {"skew_ms", skew},
        {"samples", samples},
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);

        std::map<std::string, std::unique_ptr<CaptureStore>> stores;
        std::map<std::string, int64_t> last_timestamp_by_camera;
        std::mutex last_timestamp_mutex;

        for (const CameraSpec& spec : opts.cameras) {
            stores.emplace(spec.key, std::make_unique<CaptureStore>(opts.ring_size));
        }

        std::vector<std::unique_ptr<V4L2Camera>> cameras;
        cameras.reserve(opts.cameras.size());
        for (const CameraSpec& spec : opts.cameras) {
            V4L2CameraConfig config;
            config.camera_key = spec.key;
            config.device = spec.device;
            config.width = opts.width;
            config.height = opts.height;
            config.buffer_count = opts.buffers;
        config.warmup_frames = opts.warmup;
        config.capture_frames = opts.frames;
        config.poll_timeout_ms = opts.timeout_ms;
        config.memory_mode = opts.memory_mode;
        config.cuda_import_probe = opts.cuda_import_probe;

            CaptureStore* store = stores.at(spec.key).get();
            auto callback = [store, &last_timestamp_by_camera, &last_timestamp_mutex, detail_frames = opts.detail_frames](const CameraFrame& frame) {
                store->ring.push(frame);
                std::lock_guard<std::mutex> lock(store->mutex);
                std::optional<double> delta_ms;
                if (!store->frames.empty()) {
                    delta_ms = static_cast<double>(frame.v4l2_timestamp_ns - store->frames.back().frame.v4l2_timestamp_ns) / 1e6;
                }
                if (detail_frames < 0 || static_cast<int>(store->frames.size()) < detail_frames) {
                    store->frames.push_back(FrameRecord{frame, delta_ms});
                }
                std::lock_guard<std::mutex> last_lock(last_timestamp_mutex);
                last_timestamp_by_camera[frame.camera_key] = frame.v4l2_timestamp_ns;
                return CameraFrameReleaseFence{};
            };
            cameras.push_back(std::make_unique<V4L2Camera>(config, callback));
        }

        const int64_t start_ns = monotonicNowNs();
        for (auto& camera : cameras) {
            camera->start();
        }
        for (auto& camera : cameras) {
            camera->join();
        }
        const int64_t end_ns = monotonicNowNs();

        Json report{
            {"version", "gmsl_realtime_capture_phase1_20260701"},
            {"config", Json{
                {"width", opts.width},
                {"height", opts.height},
                {"pixelformat", "UYVY"},
                {"frames_per_camera", opts.frames},
                {"warmup", opts.warmup},
                {"buffers", opts.buffers},
                {"ring_size", opts.ring_size},
                {"timeout_ms", opts.timeout_ms},
                {"detail_frames", opts.detail_frames},
                {"frame_details", opts.frame_details},
                {"memory", memoryModeName(opts.memory_mode)},
                {"cuda_import_probe", opts.cuda_import_probe},
            }},
            {"start_host_mono_ns", start_ns},
            {"end_host_mono_ns", end_ns},
            {"elapsed_ms", static_cast<double>(end_ns - start_ns) / 1e6},
            {"cameras", Json::object()},
        };

        std::map<std::string, std::vector<FrameRecord>> frames_by_camera;
        bool ok = true;
        for (std::size_t i = 0; i < opts.cameras.size(); ++i) {
            const CameraSpec& spec = opts.cameras[i];
            const V4L2CameraStats stats = cameras[i]->stats();
            Json camera_report = statsToJson(stats);
            CaptureStore& store = *stores.at(spec.key);
            std::vector<FrameRecord> frames;
            {
                std::lock_guard<std::mutex> lock(store.mutex);
                frames = store.frames;
            }
            frames_by_camera[spec.key] = frames;
            camera_report["timestamp_interval_ms"] = summarize(timestampDeltas(frames, false));
            camera_report["host_arrival_interval_ms"] = summarize(timestampDeltas(frames, true));
            camera_report["ring_snapshot_size"] = store.ring.snapshot().size();
            if (opts.frame_details) {
                Json detail = Json::array();
                for (const FrameRecord& frame : frames) {
                    detail.push_back(frameToJson(frame));
                }
                camera_report["frames"] = detail;
            }
            report["cameras"][spec.key] = camera_report;
            if (!stats.error_message.empty() || stats.frames_reported < static_cast<uint64_t>(opts.frames)) {
                ok = false;
            }
        }

        report["sync"] = syncReport(opts.cameras, frames_by_camera, opts.sync_sample_limit);
        report["ok"] = ok;

        for (const CameraSpec& spec : opts.cameras) {
            const Json& camera = report["cameras"][spec.key];
            const Json& interval = camera["timestamp_interval_ms"];
            std::cout << spec.key
                      << " device=" << spec.device
                      << " memory=" << camera.value("memory_type", "")
                      << " frames=" << camera.value("frames_reported", 0ULL)
                      << " drops=" << camera.value("drop_count", 0ULL)
                      << " errors=" << camera.value("error_flag_count", 0ULL)
                      << " cuda_ok=" << camera.value("cuda_import_success_count", 0ULL)
                      << " cuda_fail=" << camera.value("cuda_import_failure_count", 0ULL)
                      << " bytes_mismatch=" << camera.value("bytes_mismatch_count", 0ULL)
                      << " ts_source=" << camera.value("timestamp_source", "")
                      << " ts_p50_ms=" << interval.value("p50", 0.0)
                      << " ts_p95_ms=" << interval.value("p95", 0.0)
                      << "\n";
        }

        if (!opts.output_json.empty()) {
            std::ofstream out(opts.output_json);
            if (!out) {
                throw std::runtime_error("failed to open output json: " + opts.output_json);
            }
            out << std::setw(2) << report << "\n";
        }

        return ok ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
