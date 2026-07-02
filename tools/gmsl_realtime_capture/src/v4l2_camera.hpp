#pragma once

#include <linux/videodev2.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum class V4L2MemoryMode {
    Mmap,
    Dmabuf,
};

struct CameraFrame {
    std::string camera_key;
    std::string device;
    std::string memory_type;
    uint32_t sequence{0};
    uint32_t flags{0};
    int64_t v4l2_timestamp_ns{0};
    int64_t host_arrival_mono_ns{0};
    int width{0};
    int height{0};
    uint32_t fourcc{0};
    int buffer_index{-1};
    void* data{nullptr};
    int dmabuf_fd{-1};
    uint32_t surface_pitch{0};
    uint32_t surface_plane_count{0};
    size_t surface_data_size{0};
    bool cuda_imported{false};
    void* cuda_device_ptr{nullptr};
    uint32_t cuda_pitch{0};
    uint32_t cuda_plane_count{0};
    std::string cuda_frame_type;
    size_t bytesused{0};
};

enum class CameraFrameReleaseFenceType {
    NoFence,
    CudaEvent,
};

struct CameraFrameReleaseFence {
    CameraFrameReleaseFenceType type{CameraFrameReleaseFenceType::NoFence};
    void* handle{nullptr};
};

struct V4L2CameraConfig {
    std::string camera_key;
    std::string device;
    int width{1920};
    int height{1536};
    uint32_t fourcc{v4l2_fourcc('U', 'Y', 'V', 'Y')};
    int buffer_count{4};
    int warmup_frames{0};
    int capture_frames{3000};
    int poll_timeout_ms{2000};
    V4L2MemoryMode memory_mode{V4L2MemoryMode::Mmap};
    bool cuda_import_probe{false};
};

struct V4L2CameraStats {
    std::string camera_key;
    std::string device;
    int width{0};
    int height{0};
    uint32_t fourcc{0};
    int requested_buffers{0};
    int actual_buffers{0};
    std::string memory_type{"mmap"};
    uint64_t frames_seen{0};
    uint64_t frames_reported{0};
    uint64_t warmup_discarded{0};
    uint64_t error_flag_count{0};
    uint64_t bytes_mismatch_count{0};
    uint64_t drop_count{0};
    uint64_t sequence_reset_count{0};
    uint64_t timeout_count{0};
    uint64_t dmabuf_fd_count{0};
    uint64_t cuda_import_success_count{0};
    uint64_t cuda_import_failure_count{0};
    uint64_t deferred_requeue_count{0};
    uint64_t deferred_requeue_wait_count{0};
    uint64_t deferred_requeue_max_pending{0};
    uint32_t first_sequence{0};
    uint32_t last_sequence{0};
    uint32_t first_flags{0};
    uint32_t last_flags{0};
    int64_t first_timestamp_ns{0};
    int64_t last_timestamp_ns{0};
    std::string timestamp_clock{"unknown"};
    std::string timestamp_source{"unknown"};
    std::string cuda_frame_type{"none"};
    uint32_t cuda_pitch{0};
    uint32_t cuda_plane_count{0};
    std::string error_message;
    bool opened{false};
    bool streaming{false};
    bool finished{false};
};

class V4L2Camera {
public:
    using FrameCallback = std::function<CameraFrameReleaseFence(const CameraFrame&)>;

    V4L2Camera(V4L2CameraConfig config, FrameCallback callback);
    ~V4L2Camera();

    V4L2Camera(const V4L2Camera&) = delete;
    V4L2Camera& operator=(const V4L2Camera&) = delete;

    void start();
    void stop();
    void join();

    V4L2CameraStats stats() const;

private:
    struct MappedBuffer {
        void* start{nullptr};
        size_t length{0};
        int dmabuf_fd{-1};
        void* nvbuf_surface{nullptr};
        void* egl_image{nullptr};
        void* cuda_resource{nullptr};
        void* cuda_device_ptr{nullptr};
        uint32_t cuda_pitch{0};
        uint32_t cuda_plane_count{0};
        std::string cuda_frame_type;
        uint32_t surface_pitch{0};
        uint32_t surface_plane_count{0};
        size_t surface_data_size{0};
        bool egl_mapped{false};
        bool cuda_registered{false};
        bool cuda_import_valid{false};
    };

    void run();
    void openAndConfigure();
    void requestMmapBuffers(int requested_count);
    void requestDmabufBuffers(int requested_count);
    void startStreaming();
    void stopStreaming();
    void cleanup();
    CameraFrameReleaseFence handleBuffer(const v4l2_buffer& buffer);
    void prepareCudaImport(MappedBuffer& buffer);
    bool refreshCudaFrame(MappedBuffer& buffer);
    void queueBuffer(v4l2_buffer& buffer);
    void requeueReadyPending(bool wait_for_one);
    void drainPendingRequeues();
    bool releaseFenceReady(const CameraFrameReleaseFence& fence, bool wait);

    struct PendingRequeue {
        v4l2_buffer buffer{};
        CameraFrameReleaseFence fence{};
    };

    V4L2CameraConfig config_;
    FrameCallback callback_;
    mutable std::mutex stats_mutex_;
    V4L2CameraStats stats_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    int fd_{-1};
    std::vector<MappedBuffer> buffers_;
    std::deque<PendingRequeue> pending_requeues_;
    int queued_buffer_count_{0};
    bool have_last_sequence_{false};
    uint32_t last_sequence_seen_{0};
};

int64_t monotonicNowNs();
int64_t timevalToNs(const timeval& tv);
std::string fourccToString(uint32_t fourcc);
std::string timestampClock(uint32_t flags);
std::string timestampSource(uint32_t flags);
bool hasV4L2Error(uint32_t flags);
