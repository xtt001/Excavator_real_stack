#include <cv_bridge/cv_bridge.h>
#include <fpv_frame_store.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace {

std::uint64_t stampToNs(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<std::uint64_t>(stamp.sec) * 1000000000ULL +
           static_cast<std::uint64_t>(stamp.nanosec);
}

std::uint64_t systemNowNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

}  // namespace

class FpvImageSubscriberNode : public rclcpp::Node {
public:
    FpvImageSubscriberNode()
        : Node("fpv_image_subscriber"),
          writer_(declare_parameter<std::string>("shm_name", "excavator_fpv_v1")) {
        const std::string topic =
            declare_parameter<std::string>("compressed_topic", "/camera/color/image_raw/compressed");
        const std::string qos_profile = declare_parameter<std::string>("qos_profile", "sensor_data");

        rclcpp::QoS qos = rclcpp::SensorDataQoS();
        if (qos_profile == "default") {
            qos = rclcpp::QoS(rclcpp::KeepLast(10));
        }

        sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            topic,
            qos,
            std::bind(&FpvImageSubscriberNode::onImage, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "FPV subscriber on %s -> shm %s", topic.c_str(), writer_.shmName().c_str());
    }

private:
    void onImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        if (msg == nullptr) {
            return;
        }
        cv_bridge::CvImagePtr decoded;
        try {
            decoded = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& exc) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge decode failed: %s", exc.what());
            return;
        }

        cv::Mat rgb;
        cv::cvtColor(decoded->image, rgb, cv::COLOR_BGR2RGB);
        if (rgb.empty()) {
            return;
        }

        const std::uint64_t timestamp_ns = stampToNs(msg->header.stamp);
        const std::uint64_t receive_time_ns = systemNowNs();
        if (!writer_.writeRgb(
                rgb.data,
                rgb.cols,
                rgb.rows,
                timestamp_ns,
                receive_time_ns)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "FPV frame rejected (size %dx%d)", rgb.cols, rgb.rows);
        }
    }

    excavator_fpv::FpvFrameStoreWriter writer_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FpvImageSubscriberNode>());
    rclcpp::shutdown();
    return 0;
}
