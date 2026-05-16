#include "fpv_frame_store.hpp"

#include <pthread.h>

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace excavator_fpv {
namespace {

constexpr std::uint32_t kMagic = 0x46505631U;  // "FPV1"
constexpr std::uint32_t kVersion = 1U;

struct ShmHeader {
    std::uint32_t magic{kMagic};
    std::uint32_t version{kVersion};
    pthread_mutex_t mutex{};
    std::uint64_t timestamp_ns{0};
    std::uint64_t receive_time_ns{0};
    std::uint32_t sequence{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t data_bytes{0};
    std::uint8_t data[kFpvMaxBytes]{};
};

std::size_t shmTotalSize() { return sizeof(ShmHeader); }

std::string shmPath(const std::string& name) {
    if (name.empty()) {
        throw std::invalid_argument("shm name must not be empty");
    }
    return name.front() == '/' ? name : "/" + name;
}

std::uint64_t steadyNowNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

void initMutex(pthread_mutex_t* mutex) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}

void mapShm(const std::string& shm_name, bool create, void** mapped, std::size_t* size, int* fd) {
    const std::string path = shmPath(shm_name);
    const int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
    *fd = shm_open(path.c_str(), flags, 0666);
    if (*fd < 0) {
        throw std::runtime_error("shm_open failed for " + path);
    }
    *size = shmTotalSize();
    if (create) {
        if (ftruncate(*fd, static_cast<off_t>(*size)) != 0) {
            throw std::runtime_error("ftruncate failed for " + path);
        }
    }
    *mapped = mmap(nullptr, *size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if (*mapped == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + path);
    }
    if (create) {
        auto* hdr = static_cast<ShmHeader*>(*mapped);
        std::memset(hdr, 0, *size);
        hdr->magic = kMagic;
        hdr->version = kVersion;
        initMutex(&hdr->mutex);
    }
}

ShmHeader* header(void* mapped) { return static_cast<ShmHeader*>(mapped); }

}  // namespace

FpvFrameStoreWriter::FpvFrameStoreWriter(std::string shm_name) : shm_name_(std::move(shm_name)) {
    mapShm(shm_name_, true, &mapped_, &mapped_size_, &shm_fd_);
}

FpvFrameStoreWriter::~FpvFrameStoreWriter() {
    if (mapped_ != nullptr && mapped_ != MAP_FAILED) {
        munmap(mapped_, mapped_size_);
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
    }
}

bool FpvFrameStoreWriter::writeRgb(
    const std::uint8_t* rgb,
    int width,
    int height,
    std::uint64_t timestamp_ns,
    std::uint64_t receive_time_ns) {
    if (rgb == nullptr || width <= 0 || height <= 0 || width > kFpvMaxWidth ||
        height > kFpvMaxHeight) {
        return false;
    }
    const std::size_t bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    if (bytes > kFpvMaxBytes) {
        return false;
    }

    ShmHeader* hdr = header(mapped_);
    pthread_mutex_lock(&hdr->mutex);
    hdr->timestamp_ns = timestamp_ns;
    hdr->receive_time_ns = receive_time_ns;
    hdr->sequence += 1U;
    hdr->width = static_cast<std::uint32_t>(width);
    hdr->height = static_cast<std::uint32_t>(height);
    hdr->data_bytes = static_cast<std::uint32_t>(bytes);
    std::memcpy(hdr->data, rgb, bytes);
    pthread_mutex_unlock(&hdr->mutex);
    return true;
}

FpvFrameStoreReader::FpvFrameStoreReader(std::string shm_name) : shm_name_(std::move(shm_name)) {
    mapShm(shm_name_, false, &mapped_, &mapped_size_, &shm_fd_);
}

FpvFrameStoreReader::~FpvFrameStoreReader() {
    if (mapped_ != nullptr && mapped_ != MAP_FAILED) {
        munmap(mapped_, mapped_size_);
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
    }
}

bool FpvFrameStoreReader::readLatest(FpvFrameView* out, std::vector<std::uint8_t>* rgb_copy) const {
    if (out == nullptr || rgb_copy == nullptr || mapped_ == nullptr) {
        return false;
    }
    const ShmHeader* hdr = header(mapped_);
    if (hdr->magic != kMagic || hdr->version != kVersion || hdr->data_bytes == 0U) {
        return false;
    }
    const int width = static_cast<int>(hdr->width);
    const int height = static_cast<int>(hdr->height);
    if (width <= 0 || height <= 0) {
        return false;
    }

    pthread_mutex_lock(const_cast<pthread_mutex_t*>(&hdr->mutex));
    rgb_copy->assign(hdr->data, hdr->data + hdr->data_bytes);
    out->timestamp_ns = hdr->timestamp_ns;
    out->receive_time_ns = hdr->receive_time_ns;
    out->sequence = hdr->sequence;
    out->width = width;
    out->height = height;
    out->rgb = rgb_copy->data();
    out->rgb_size = rgb_copy->size();
    pthread_mutex_unlock(const_cast<pthread_mutex_t*>(&hdr->mutex));
    return true;
}

bool FpvFrameStoreReader::isFresh(std::uint64_t now_ns, int max_age_ms) const {
    if (mapped_ == nullptr || max_age_ms <= 0) {
        return false;
    }
    const ShmHeader* hdr = header(mapped_);
    if (hdr->magic != kMagic || hdr->receive_time_ns == 0U) {
        return false;
    }
    const std::uint64_t age_ns =
        now_ns > hdr->receive_time_ns ? now_ns - hdr->receive_time_ns : 0U;
    return age_ns <= static_cast<std::uint64_t>(max_age_ms) * 1000000ULL;
}

}  // namespace excavator_fpv
