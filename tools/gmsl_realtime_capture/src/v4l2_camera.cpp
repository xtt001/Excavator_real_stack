#include "v4l2_camera.hpp"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <mutex>
#include <stdexcept>

#ifdef GMSL_HAVE_JETSON_ZERO_COPY
#include <EGL/egl.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <nvbufsurface.h>
#endif

namespace {

void xioctl(int fd, unsigned long request, void* arg, const char* name) {
    int rc;
    do {
        rc = ioctl(fd, request, arg);
    } while (rc == -1 && errno == EINTR);
    if (rc == -1) {
        throw std::runtime_error(std::string(name) + " failed: " + std::strerror(errno));
    }
}

size_t expectedBytes(const V4L2CameraConfig& config) {
    if (config.fourcc == v4l2_fourcc('U', 'Y', 'V', 'Y')) {
        return static_cast<size_t>(config.width) * static_cast<size_t>(config.height) * 2U;
    }
    return 0;
}

v4l2_memory toV4L2Memory(V4L2MemoryMode mode) {
    return mode == V4L2MemoryMode::Dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
}

std::string memoryModeName(V4L2MemoryMode mode) {
    return mode == V4L2MemoryMode::Dmabuf ? "dmabuf" : "mmap";
}

#ifdef GMSL_HAVE_JETSON_ZERO_COPY
NvBufSurfaceColorFormat toNvBufColorFormat(uint32_t fourcc) {
    if (fourcc == v4l2_fourcc('U', 'Y', 'V', 'Y')) {
        return NVBUF_COLOR_FORMAT_UYVY;
    }
    throw std::runtime_error("DMABUF zero-copy currently supports UYVY only");
}

std::string cudaFrameTypeName(CUeglFrameType type) {
    switch (type) {
        case CU_EGL_FRAME_TYPE_ARRAY:
            return "array";
        case CU_EGL_FRAME_TYPE_PITCH:
            return "pitch";
        default:
            return "unknown";
    }
}

void checkCuda(CUresult result, const char* name) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* error_name = nullptr;
    cuGetErrorName(result, &error_name);
    throw std::runtime_error(std::string(name) + " failed: " + (error_name ? error_name : "unknown"));
}
#endif

}  // namespace

V4L2Camera::V4L2Camera(V4L2CameraConfig config, FrameCallback callback)
    : config_(std::move(config)), callback_(std::move(callback)) {
    stats_.camera_key = config_.camera_key;
    stats_.device = config_.device;
    stats_.width = config_.width;
    stats_.height = config_.height;
    stats_.fourcc = config_.fourcc;
    stats_.requested_buffers = config_.buffer_count;
    stats_.memory_type = memoryModeName(config_.memory_mode);
}

V4L2Camera::~V4L2Camera() {
    stop();
    join();
}

void V4L2Camera::start() {
    thread_ = std::thread(&V4L2Camera::run, this);
}

void V4L2Camera::stop() {
    stop_requested_.store(true);
}

void V4L2Camera::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

V4L2CameraStats V4L2Camera::stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void V4L2Camera::run() {
    try {
        openAndConfigure();
        startStreaming();

        while (!stop_requested_.load()) {
            requeueReadyPending(false);
            if (queued_buffer_count_ == 0 && !pending_requeues_.empty()) {
                requeueReadyPending(true);
            }
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (config_.capture_frames > 0 &&
                    stats_.frames_reported >= static_cast<uint64_t>(config_.capture_frames)) {
                    break;
                }
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);
            timeval tv{};
            tv.tv_sec = config_.poll_timeout_ms / 1000;
            tv.tv_usec = (config_.poll_timeout_ms % 1000) * 1000;

            const int selected = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (selected == -1 && errno == EINTR) {
                continue;
            }
            if (selected <= 0) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.timeout_count += 1;
                throw std::runtime_error("select timeout or failure");
            }

            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = toV4L2Memory(config_.memory_mode);
            xioctl(fd_, VIDIOC_DQBUF, &buffer, "VIDIOC_DQBUF");
            queued_buffer_count_ = std::max(0, queued_buffer_count_ - 1);
            CameraFrameReleaseFence fence = handleBuffer(buffer);
            if (fence.type == CameraFrameReleaseFenceType::NoFence || fence.handle == nullptr) {
                queueBuffer(buffer);
            } else {
                pending_requeues_.push_back(PendingRequeue{buffer, fence});
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.deferred_requeue_count += 1;
                stats_.deferred_requeue_max_pending = std::max(
                    stats_.deferred_requeue_max_pending,
                    static_cast<uint64_t>(pending_requeues_.size()));
            }
        }
        drainPendingRequeues();
    } catch (const std::exception& exc) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.error_message = exc.what();
    }

    cleanup();
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.finished = true;
    }
}

void V4L2Camera::openAndConfigure() {
    fd_ = open(config_.device.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        throw std::runtime_error("open " + config_.device + " failed: " + std::strerror(errno));
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.opened = true;
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = config_.width;
    format.fmt.pix.height = config_.height;
    format.fmt.pix.pixelformat = config_.fourcc;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    xioctl(fd_, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");

    config_.width = static_cast<int>(format.fmt.pix.width);
    config_.height = static_cast<int>(format.fmt.pix.height);
    config_.fourcc = format.fmt.pix.pixelformat;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.width = config_.width;
        stats_.height = config_.height;
        stats_.fourcc = config_.fourcc;
    }

    if (config_.memory_mode == V4L2MemoryMode::Dmabuf) {
        requestDmabufBuffers(config_.buffer_count);
    } else {
        requestMmapBuffers(config_.buffer_count);
    }
}

void V4L2Camera::requestMmapBuffers(int requested_count) {
    v4l2_requestbuffers request{};
    request.count = static_cast<uint32_t>(requested_count);
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    xioctl(fd_, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
    if (request.count < 2) {
        throw std::runtime_error("driver returned fewer than two mmap buffers");
    }

    buffers_.resize(request.count);
    for (uint32_t i = 0; i < request.count; ++i) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;
        xioctl(fd_, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
        buffers_[i].length = buffer.length;
        buffers_[i].start = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            buffers_[i].start = nullptr;
            throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
        }
        queueBuffer(buffer);
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.actual_buffers = static_cast<int>(request.count);
    }
}

void V4L2Camera::requestDmabufBuffers(int requested_count) {
#ifndef GMSL_HAVE_JETSON_ZERO_COPY
    (void)requested_count;
    throw std::runtime_error("this build does not include Jetson zero-copy support");
#else
    if (config_.fourcc != v4l2_fourcc('U', 'Y', 'V', 'Y')) {
        throw std::runtime_error("DMABUF mode currently supports UYVY only");
    }
    if (config_.cuda_import_probe) {
        const cudaError_t runtime_status = cudaFree(nullptr);
        if (runtime_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(0) failed: ") + cudaGetErrorString(runtime_status));
        }
        checkCuda(cuInit(0), "cuInit");
    }

    v4l2_requestbuffers request{};
    request.count = static_cast<uint32_t>(requested_count);
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_DMABUF;
    xioctl(fd_, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
    if (request.count < 2) {
        throw std::runtime_error("driver returned fewer than two dmabuf buffers");
    }

    buffers_.resize(request.count);
    for (uint32_t i = 0; i < request.count; ++i) {
        NvBufSurfaceAllocateParams params{};
        params.params.width = static_cast<uint32_t>(config_.width);
        params.params.height = static_cast<uint32_t>(config_.height);
        params.params.layout = NVBUF_LAYOUT_PITCH;
        params.params.colorFormat = toNvBufColorFormat(config_.fourcc);
        params.params.memType = NVBUF_MEM_SURFACE_ARRAY;
        params.memtag = NvBufSurfaceTag_CAMERA;

        NvBufSurface* surface = nullptr;
        if (NvBufSurfaceAllocate(&surface, 1, &params) != 0 || surface == nullptr) {
            throw std::runtime_error("NvBufSurfaceAllocate failed");
        }
        surface->numFilled = 1;

        MappedBuffer& mapped = buffers_[i];
        mapped.nvbuf_surface = surface;
        mapped.dmabuf_fd = static_cast<int>(surface->surfaceList[0].bufferDesc);
        mapped.surface_pitch = surface->surfaceList[0].pitch;
        mapped.surface_plane_count = surface->surfaceList[0].planeParams.num_planes;
        mapped.surface_data_size = surface->surfaceList[0].dataSize;
        mapped.length = surface->surfaceList[0].dataSize;

        if (config_.cuda_import_probe) {
            prepareCudaImport(mapped);
        }

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_DMABUF;
        buffer.index = i;
        xioctl(fd_, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
        mapped.length = buffer.length;
        buffer.m.fd = mapped.dmabuf_fd;
        queueBuffer(buffer);
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.actual_buffers = static_cast<int>(request.count);
        stats_.dmabuf_fd_count = request.count;
    }
#endif
}

void V4L2Camera::startStreaming() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.streaming = true;
}

void V4L2Camera::stopStreaming() {
    if (fd_ < 0) {
        return;
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.streaming = false;
}

void V4L2Camera::cleanup() {
    stopStreaming();
    for (MappedBuffer& buffer : buffers_) {
        if (buffer.cuda_registered) {
#ifdef GMSL_HAVE_JETSON_ZERO_COPY
            cuGraphicsUnregisterResource(reinterpret_cast<CUgraphicsResource>(buffer.cuda_resource));
#endif
            buffer.cuda_resource = nullptr;
            buffer.cuda_registered = false;
        }
        if (buffer.egl_mapped) {
#ifdef GMSL_HAVE_JETSON_ZERO_COPY
            NvBufSurfaceUnMapEglImage(static_cast<NvBufSurface*>(buffer.nvbuf_surface), 0);
#endif
            buffer.egl_image = nullptr;
            buffer.egl_mapped = false;
        }
        if (buffer.nvbuf_surface != nullptr) {
#ifdef GMSL_HAVE_JETSON_ZERO_COPY
            NvBufSurfaceDestroy(static_cast<NvBufSurface*>(buffer.nvbuf_surface));
#endif
            buffer.nvbuf_surface = nullptr;
            buffer.dmabuf_fd = -1;
        }
        if (buffer.start != nullptr) {
            munmap(buffer.start, buffer.length);
            buffer.start = nullptr;
            buffer.length = 0;
        }
    }
    buffers_.clear();
    pending_requeues_.clear();
    queued_buffer_count_ = 0;
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void V4L2Camera::prepareCudaImport(MappedBuffer& buffer) {
#ifndef GMSL_HAVE_JETSON_ZERO_COPY
    (void)buffer;
    throw std::runtime_error("this build does not include Jetson zero-copy support");
#else
    auto* surface = static_cast<NvBufSurface*>(buffer.nvbuf_surface);
    if (surface == nullptr) {
        throw std::runtime_error("cannot CUDA-import a null NvBufSurface");
    }
    if (!buffer.egl_mapped) {
        if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
            throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        }
        buffer.egl_image = surface->surfaceList[0].mappedAddr.eglImage;
        buffer.egl_mapped = true;
    }
    if (buffer.egl_image == nullptr) {
        throw std::runtime_error("NvBufSurfaceMapEglImage returned null EGLImage");
    }
    if (!buffer.cuda_registered) {
        CUgraphicsResource resource = nullptr;
        checkCuda(cuGraphicsEGLRegisterImage(
                      &resource,
                      reinterpret_cast<EGLImageKHR>(buffer.egl_image),
                      CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY),
                  "cuGraphicsEGLRegisterImage");
        buffer.cuda_resource = resource;
        buffer.cuda_registered = true;
    }
    if (!refreshCudaFrame(buffer)) {
        throw std::runtime_error("initial CUDA EGL frame import failed");
    }
#endif
}

bool V4L2Camera::refreshCudaFrame(MappedBuffer& buffer) {
#ifndef GMSL_HAVE_JETSON_ZERO_COPY
    (void)buffer;
    return false;
#else
    if (!buffer.cuda_registered) {
        return false;
    }
    CUeglFrame egl_frame{};
    const CUresult status = cuGraphicsResourceGetMappedEglFrame(
        &egl_frame,
        reinterpret_cast<CUgraphicsResource>(buffer.cuda_resource),
        0,
        0);
    if (status != CUDA_SUCCESS) {
        return false;
    }
    buffer.cuda_frame_type = cudaFrameTypeName(egl_frame.frameType);
    buffer.cuda_pitch = egl_frame.pitch;
    buffer.cuda_plane_count = egl_frame.planeCount;
    buffer.cuda_device_ptr = egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH ? egl_frame.frame.pPitch[0] : nullptr;
    buffer.cuda_import_valid = buffer.cuda_device_ptr != nullptr;
    return buffer.cuda_import_valid;
#endif
}

CameraFrameReleaseFence V4L2Camera::handleBuffer(const v4l2_buffer& buffer) {
    const int64_t host_ns = monotonicNowNs();
    const int64_t timestamp_ns = timevalToNs(buffer.timestamp);
    const uint64_t frames_seen_before = stats().frames_seen;
    const bool should_report = frames_seen_before >= static_cast<uint64_t>(config_.warmup_frames);
    const size_t expected = expectedBytes(config_);
    MappedBuffer* mapped_buffer = nullptr;
    if (buffer.index < buffers_.size()) {
        mapped_buffer = &buffers_[buffer.index];
    }
    bool cuda_imported = false;
    if (mapped_buffer != nullptr && config_.memory_mode == V4L2MemoryMode::Dmabuf && config_.cuda_import_probe) {
        cuda_imported = refreshCudaFrame(*mapped_buffer);
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.frames_seen += 1;
        if (!have_last_sequence_) {
            stats_.first_sequence = buffer.sequence;
            stats_.first_flags = buffer.flags;
            stats_.first_timestamp_ns = timestamp_ns;
            stats_.timestamp_clock = timestampClock(buffer.flags);
            stats_.timestamp_source = timestampSource(buffer.flags);
            have_last_sequence_ = true;
        } else if (buffer.sequence > last_sequence_seen_ + 1U) {
            stats_.drop_count += static_cast<uint64_t>(buffer.sequence - last_sequence_seen_ - 1U);
        } else if (buffer.sequence <= last_sequence_seen_) {
            stats_.sequence_reset_count += 1;
        }
        last_sequence_seen_ = buffer.sequence;
        stats_.last_sequence = buffer.sequence;
        stats_.last_flags = buffer.flags;
        stats_.last_timestamp_ns = timestamp_ns;
        if (hasV4L2Error(buffer.flags) && should_report) {
            stats_.error_flag_count += 1;
        }
        if (expected != 0 && buffer.bytesused != expected && should_report) {
            stats_.bytes_mismatch_count += 1;
        }
        if (!should_report) {
            stats_.warmup_discarded += 1;
        } else {
            stats_.frames_reported += 1;
            if (config_.memory_mode == V4L2MemoryMode::Dmabuf && config_.cuda_import_probe) {
                if (cuda_imported) {
                    stats_.cuda_import_success_count += 1;
                    if (mapped_buffer != nullptr) {
                        stats_.cuda_frame_type = mapped_buffer->cuda_frame_type;
                        stats_.cuda_pitch = mapped_buffer->cuda_pitch;
                        stats_.cuda_plane_count = mapped_buffer->cuda_plane_count;
                    }
                } else {
                    stats_.cuda_import_failure_count += 1;
                }
            }
        }
    }

    if (!should_report || !callback_) {
        return CameraFrameReleaseFence{};
    }
    void* data = mapped_buffer != nullptr ? mapped_buffer->start : nullptr;
    int dmabuf_fd = mapped_buffer != nullptr ? mapped_buffer->dmabuf_fd : -1;
    uint32_t surface_pitch = mapped_buffer != nullptr ? mapped_buffer->surface_pitch : 0;
    uint32_t surface_plane_count = mapped_buffer != nullptr ? mapped_buffer->surface_plane_count : 0;
    size_t surface_data_size = mapped_buffer != nullptr ? mapped_buffer->surface_data_size : 0;
    void* cuda_device_ptr = mapped_buffer != nullptr ? mapped_buffer->cuda_device_ptr : nullptr;
    uint32_t cuda_pitch = mapped_buffer != nullptr ? mapped_buffer->cuda_pitch : 0;
    uint32_t cuda_plane_count = mapped_buffer != nullptr ? mapped_buffer->cuda_plane_count : 0;
    std::string cuda_frame_type = mapped_buffer != nullptr ? mapped_buffer->cuda_frame_type : "";
    CameraFrame frame{
        config_.camera_key,
        config_.device,
        memoryModeName(config_.memory_mode),
        buffer.sequence,
        buffer.flags,
        timestamp_ns,
        host_ns,
        config_.width,
        config_.height,
        config_.fourcc,
        static_cast<int>(buffer.index),
        data,
        dmabuf_fd,
        surface_pitch,
        surface_plane_count,
        surface_data_size,
        cuda_imported,
        cuda_device_ptr,
        cuda_pitch,
        cuda_plane_count,
        cuda_frame_type,
        buffer.bytesused,
    };
    return callback_(frame);
}

void V4L2Camera::queueBuffer(v4l2_buffer& buffer) {
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = toV4L2Memory(config_.memory_mode);
    if (config_.memory_mode == V4L2MemoryMode::Dmabuf && buffer.index < buffers_.size()) {
        buffer.m.fd = buffers_[buffer.index].dmabuf_fd;
    }
    xioctl(fd_, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
    queued_buffer_count_ += 1;
}

void V4L2Camera::requeueReadyPending(bool wait_for_one) {
    bool waited = false;
    while (!pending_requeues_.empty()) {
        PendingRequeue& pending = pending_requeues_.front();
        const bool wait = wait_for_one && !waited;
        if (!releaseFenceReady(pending.fence, wait)) {
            break;
        }
        if (wait) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.deferred_requeue_wait_count += 1;
        }
        queueBuffer(pending.buffer);
        pending_requeues_.pop_front();
        if (wait_for_one) {
            waited = true;
            break;
        }
    }
}

void V4L2Camera::drainPendingRequeues() {
    while (!pending_requeues_.empty()) {
        requeueReadyPending(true);
    }
}

bool V4L2Camera::releaseFenceReady(const CameraFrameReleaseFence& fence, bool wait) {
    if (fence.type == CameraFrameReleaseFenceType::NoFence || fence.handle == nullptr) {
        return true;
    }
    if (fence.type != CameraFrameReleaseFenceType::CudaEvent) {
        throw std::runtime_error("unsupported frame release fence type");
    }
#ifndef GMSL_HAVE_JETSON_ZERO_COPY
    (void)wait;
    throw std::runtime_error("CUDA release fences require Jetson zero-copy support");
#else
    auto event = reinterpret_cast<cudaEvent_t>(fence.handle);
    const cudaError_t status = wait ? cudaEventSynchronize(event) : cudaEventQuery(event);
    if (status == cudaSuccess) {
        return true;
    }
    if (!wait && status == cudaErrorNotReady) {
        return false;
    }
    throw std::runtime_error(std::string(wait ? "cudaEventSynchronize" : "cudaEventQuery") +
                             " failed: " + cudaGetErrorString(status));
#endif
}

int64_t monotonicNowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

int64_t timevalToNs(const timeval& tv) {
    return static_cast<int64_t>(tv.tv_sec) * 1000000000LL + static_cast<int64_t>(tv.tv_usec) * 1000LL;
}

std::string fourccToString(uint32_t fourcc) {
    std::string out(4, '\0');
    out[0] = static_cast<char>(fourcc & 0xFF);
    out[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    out[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    out[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    return out;
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

bool hasV4L2Error(uint32_t flags) {
    return (flags & V4L2_BUF_FLAG_ERROR) != 0;
}
