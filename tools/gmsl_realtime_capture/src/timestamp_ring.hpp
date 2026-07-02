#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

template <typename T>
class TimestampRing {
public:
    explicit TimestampRing(std::size_t capacity) : capacity_(capacity) {}

    void push(const T& value) {
        if (capacity_ == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.size() < capacity_) {
            items_.push_back(value);
            next_ = items_.size() % capacity_;
            return;
        }
        items_[next_] = value;
        next_ = (next_ + 1) % capacity_;
    }

    std::optional<T> latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.empty()) {
            return std::nullopt;
        }
        if (items_.size() < capacity_) {
            return items_.back();
        }
        const std::size_t index = (next_ + capacity_ - 1) % capacity_;
        return items_[index];
    }

    std::optional<T> nearest(int64_t timestamp_ns) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.empty()) {
            return std::nullopt;
        }
        const auto best = std::min_element(
            items_.begin(),
            items_.end(),
            [&](const T& lhs, const T& rhs) {
                const auto lhs_diff = absDiff(lhs.v4l2_timestamp_ns, timestamp_ns);
                const auto rhs_diff = absDiff(rhs.v4l2_timestamp_ns, timestamp_ns);
                return lhs_diff < rhs_diff;
            });
        return *best;
    }

    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.empty() || items_.size() < capacity_) {
            return items_;
        }
        std::vector<T> out;
        out.reserve(items_.size());
        for (std::size_t i = 0; i < items_.size(); ++i) {
            out.push_back(items_[(next_ + i) % items_.size()]);
        }
        return out;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

private:
    static int64_t absDiff(int64_t a, int64_t b) {
        return a >= b ? a - b : b - a;
    }

    std::size_t capacity_{0};
    std::size_t next_{0};
    mutable std::mutex mutex_;
    std::vector<T> items_;
};
