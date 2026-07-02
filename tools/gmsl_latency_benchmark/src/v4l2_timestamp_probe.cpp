#include <linux/videodev2.h>
#include <nlohmann/json.hpp>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

struct Options {
    std::string device{"/dev/video0"};
    std::string output_json;
    int width{1920};
    int height{1536};
    int frames{120};
    int warmup{4};
    int buffers{4};
};

struct Buffer {
    void* start{nullptr};
    size_t length{0};
};

struct FrameSample {
    uint32_t index{0};
    uint32_t sequence{0};
    uint32_t flags{0};
    uint32_t bytesused{0};
    int64_t timestamp_ns{0};
    int64_t host_mono_ns{0};
};

std::string valueFor(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    return argv[++i];
}

Options parseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--device") {
            opts.device = valueFor(i, argc, argv, arg);
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
        } else if (arg == "--output-json") {
            opts.output_json = valueFor(i, argc, argv, arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: v4l2_timestamp_probe --device /dev/videoN [options]\n"
                << "  --width N --height N        Capture format. Default: 1920x1536.\n"
                << "  --frames N                  Measured frames after warmup. Default: 120.\n"
                << "  --warmup N                  Warmup frames. Default: 4.\n"
                << "  --buffers N                 MMAP buffer count. Default: 4.\n"
                << "  --output-json PATH          Write JSON report.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.width <= 0 || opts.height <= 0 || opts.frames <= 0 || opts.warmup < 0 || opts.buffers <= 0) {
        throw std::runtime_error("invalid numeric option");
    }
    return opts;
}

void xioctl(int fd, unsigned long request, void* arg, const std::string& name) {
    int rc;
    do {
        rc = ioctl(fd, request, arg);
    } while (rc == -1 && errno == EINTR);
    if (rc == -1) {
        throw std::runtime_error(name + " failed: " + std::strerror(errno));
    }
}

int64_t tvToNs(const timeval& tv) {
    return static_cast<int64_t>(tv.tv_sec) * 1000000000LL + static_cast<int64_t>(tv.tv_usec) * 1000LL;
}

int64_t monoNowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

std::vector<double> deltasMs(const std::vector<FrameSample>& samples, bool host) {
    std::vector<double> out;
    if (samples.size() < 2) {
        return out;
    }
    out.reserve(samples.size() - 1);
    for (size_t i = 1; i < samples.size(); ++i) {
        const int64_t cur = host ? samples[i].host_mono_ns : samples[i].timestamp_ns;
        const int64_t prev = host ? samples[i - 1].host_mono_ns : samples[i - 1].timestamp_ns;
        out.push_back(static_cast<double>(cur - prev) / 1e6);
    }
    return out;
}

Json summarize(std::vector<double> values) {
    if (values.empty()) {
        return Json{{"count", 0}, {"mean", nullptr}, {"p50", nullptr}, {"p95", nullptr}, {"max", nullptr}};
    }
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    auto pct = [&](double q) {
        const double pos = (static_cast<double>(values.size()) - 1.0) * q;
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = std::min(lo + 1, values.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    };
    return Json{
        {"count", values.size()},
        {"mean", sum / static_cast<double>(values.size())},
        {"p50", pct(0.50)},
        {"p95", pct(0.95)},
        {"max", values.back()},
    };
}

std::string timestampSource(uint32_t flags) {
    switch (flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) {
        case V4L2_BUF_FLAG_TSTAMP_SRC_EOF:
            return "eof";
        case V4L2_BUF_FLAG_TSTAMP_SRC_SOE:
            return "soe";
        default:
            return "unknown";
    }
}

std::string timestampClock(uint32_t flags) {
    switch (flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) {
        case V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
            return "monotonic";
        case V4L2_BUF_FLAG_TIMESTAMP_COPY:
            return "copy";
        default:
            return "unknown";
    }
}

Json flagsJson(uint32_t flags) {
    return Json{
        {"raw", flags},
        {"error", static_cast<bool>(flags & V4L2_BUF_FLAG_ERROR)},
        {"timestamp_clock", timestampClock(flags)},
        {"timestamp_source", timestampSource(flags)},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);
        const int fd = open(opts.device.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd < 0) {
            throw std::runtime_error("failed to open " + opts.device + ": " + std::strerror(errno));
        }

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = opts.width;
        fmt.fmt.pix.height = opts.height;
        fmt.fmt.pix.pixelformat = v4l2_fourcc('U', 'Y', 'V', 'Y');
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        xioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT");

        v4l2_requestbuffers req{};
        req.count = opts.buffers;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(fd, VIDIOC_REQBUFS, &req, "VIDIOC_REQBUFS");
        if (req.count < 2) {
            throw std::runtime_error("driver returned fewer than two buffers");
        }

        std::vector<Buffer> buffers(req.count);
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(fd, VIDIOC_QUERYBUF, &buf, "VIDIOC_QUERYBUF");
            buffers[i].length = buf.length;
            buffers[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
            if (buffers[i].start == MAP_FAILED) {
                throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
            }
            xioctl(fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF");
        }

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");

        std::vector<FrameSample> samples;
        const int target = opts.warmup + opts.frames;
        for (int i = 0; i < target;) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            timeval tv{};
            tv.tv_sec = 2;
            const int sel = select(fd + 1, &fds, nullptr, nullptr, &tv);
            if (sel == -1 && errno == EINTR) {
                continue;
            }
            if (sel <= 0) {
                throw std::runtime_error("select timeout or failure");
            }

            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            xioctl(fd, VIDIOC_DQBUF, &buf, "VIDIOC_DQBUF");
            const int64_t host_ns = monoNowNs();
            if (i >= opts.warmup) {
                samples.push_back(FrameSample{
                    buf.index,
                    buf.sequence,
                    buf.flags,
                    buf.bytesused,
                    tvToNs(buf.timestamp),
                    host_ns,
                });
            }
            xioctl(fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF");
            ++i;
        }

        xioctl(fd, VIDIOC_STREAMOFF, &type, "VIDIOC_STREAMOFF");
        for (const Buffer& buffer : buffers) {
            if (buffer.start && buffer.start != MAP_FAILED) {
                munmap(buffer.start, buffer.length);
            }
        }
        close(fd);

        Json frames = Json::array();
        for (const FrameSample& sample : samples) {
            frames.push_back(Json{
                {"index", sample.index},
                {"sequence", sample.sequence},
                {"flags", flagsJson(sample.flags)},
                {"bytesused", sample.bytesused},
                {"timestamp_ns", sample.timestamp_ns},
                {"host_mono_ns", sample.host_mono_ns},
                {"host_minus_timestamp_ms", static_cast<double>(sample.host_mono_ns - sample.timestamp_ns) / 1e6},
            });
        }

        Json report{
            {"version", "v4l2_timestamp_probe_20260701"},
            {"device", opts.device},
            {"width", fmt.fmt.pix.width},
            {"height", fmt.fmt.pix.height},
            {"pixelformat", "UYVY"},
            {"frames", samples.size()},
            {"warmup", opts.warmup},
            {"buffers", req.count},
            {"timestamp_interval_ms", summarize(deltasMs(samples, false))},
            {"host_interval_ms", summarize(deltasMs(samples, true))},
            {"first_flags", samples.empty() ? Json{} : flagsJson(samples.front().flags)},
            {"last_flags", samples.empty() ? Json{} : flagsJson(samples.back().flags)},
            {"frames_detail", frames},
        };

        std::cout << opts.device
                  << " frames=" << samples.size()
                  << " clock=" << report["first_flags"].value("timestamp_clock", "")
                  << " source=" << report["first_flags"].value("timestamp_source", "")
                  << " error_flag=" << report["first_flags"].value("error", false)
                  << " ts_delta_p50=" << report["timestamp_interval_ms"].value("p50", 0.0)
                  << " ts_delta_p95=" << report["timestamp_interval_ms"].value("p95", 0.0)
                  << "\n";

        if (!opts.output_json.empty()) {
            std::ofstream out(opts.output_json);
            if (!out) {
                throw std::runtime_error("failed to open output json: " + opts.output_json);
            }
            out << std::setw(2) << report << "\n";
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
