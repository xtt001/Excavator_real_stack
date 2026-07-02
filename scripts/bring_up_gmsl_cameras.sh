#!/usr/bin/env bash
set -euo pipefail

DRIVER_DIR="${GMSL_DRIVER_DIR:-/home/mundane/SG8A_AGON_G2Y_A1_AGX_Orin_YUV_JP6.2_L4TR36.4.3}"

MODEL="${GMSL_CAMERA_MODEL:-sg3s_isx031}"
VIDEO_DEVICES_RAW="${GMSL_VIDEO_DEVICES-4 5 6 7}"
VIDEO_DEVICES=""
TRIG_MODE="${GMSL_TRIG_MODE:-2}"
TRIG_PIN="${GMSL_TRIG_PIN:-0x00020007}"
PIXEL_FORMAT="${GMSL_PIXEL_FORMAT:-UYVY}"

red_print() {
    echo -e "\e[1;31m$1\e[0m"
}

green_print() {
    echo -e "\e[1;32m$1\e[0m"
}

run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

write_sysfs() {
    local value="$1"
    local path="$2"

    run_as_root sh -c "printf '%s\n' '$value' > '$path'"
}

normalize_video_devices() {
    local raw="$1"
    local normalized=()

    if [[ "${raw}" =~ ^[[:space:]]*$ ]]; then
        red_print "GMSL_VIDEO_DEVICES must include at least one video device id from 0 to 7"
        exit 2
    fi

    for dev in ${raw}; do
        if [[ ! "${dev}" =~ ^[0-7]$ ]]; then
            red_print "Invalid GMSL_VIDEO_DEVICES entry '${dev}': must be an integer from 0 to 7"
            exit 2
        fi
        normalized+=("${dev}")
    done

    VIDEO_DEVICES="${normalized[*]}"
}

normalize_video_devices "${VIDEO_DEVICES_RAW}"

case "${MODEL}" in
    sg8_ox08bc|SG8-OX08BC|SG8-OX08BC-5300-GMSL2-Hxxx)
        WIDTH="${GMSL_WIDTH:-3840}"
        HEIGHT="${GMSL_HEIGHT:-2160}"
        SENSOR_MODE="${GMSL_SENSOR_MODE:-4}"
        ENABLE_3G_0="${GMSL_ENABLE_3G_0:-0,0,0,0}"
        ENABLE_3G_1="${GMSL_ENABLE_3G_1:-0,0,0,0}"
        ;;
    sg8s_ar0820|SG8S-AR0820C|SG8S-AR0820C-5300-G2A-Hxxx)
        WIDTH="${GMSL_WIDTH:-3840}"
        HEIGHT="${GMSL_HEIGHT:-2160}"
        SENSOR_MODE="${GMSL_SENSOR_MODE:-4}"
        ENABLE_3G_0="${GMSL_ENABLE_3G_0:-0,0,0,0}"
        ENABLE_3G_1="${GMSL_ENABLE_3G_1:-0,0,0,0}"
        ;;
    sg8_isx028c|SG8-ISX028C|SG8-ISX028C-G2G-Hxxx)
        WIDTH="${GMSL_WIDTH:-3840}"
        HEIGHT="${GMSL_HEIGHT:-2160}"
        SENSOR_MODE="${GMSL_SENSOR_MODE:-5}"
        ENABLE_3G_0="${GMSL_ENABLE_3G_0:-0,0,0,0}"
        ENABLE_3G_1="${GMSL_ENABLE_3G_1:-0,0,0,0}"
        ;;
    sg3s_isx031|SG3S-ISX031C-GMSL2-Hxxx)
        WIDTH="${GMSL_WIDTH:-1920}"
        HEIGHT="${GMSL_HEIGHT:-1536}"
        SENSOR_MODE="${GMSL_SENSOR_MODE:-2}"
        ENABLE_3G_0="${GMSL_ENABLE_3G_0:-0,0,0,0}"
        ENABLE_3G_1="${GMSL_ENABLE_3G_1:-0,0,0,0}"
        ;;
    sg3s_isx031f|SG3S-ISX031C-GMSL2F-Hxxx)
        WIDTH="${GMSL_WIDTH:-1920}"
        HEIGHT="${GMSL_HEIGHT:-1536}"
        SENSOR_MODE="${GMSL_SENSOR_MODE:-2}"
        ENABLE_3G_0="${GMSL_ENABLE_3G_0:-1,1,1,1}"
        ENABLE_3G_1="${GMSL_ENABLE_3G_1:-1,1,1,1}"
        ;;
    *)
        red_print "Unsupported GMSL_CAMERA_MODEL='${MODEL}'"
        red_print "Supported: sg8_ox08bc, sg8s_ar0820, sg8_isx028c, sg3s_isx031, sg3s_isx031f"
        exit 2
        ;;
esac

green_print "GMSL driver dir: ${DRIVER_DIR}"
green_print "GMSL model: ${MODEL}"
green_print "Link speed enable_3G_0=${ENABLE_3G_0} enable_3G_1=${ENABLE_3G_1}"
green_print "Video format: ${WIDTH}x${HEIGHT}, pixelformat=${PIXEL_FORMAT}, sensor_mode=${SENSOR_MODE}, trig_mode=${TRIG_MODE}, trig_pin=${TRIG_PIN}"
green_print "Video devices: ${VIDEO_DEVICES}"

if [[ "${GMSL_PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
    exit 0
fi

if [[ ! -d "${DRIVER_DIR}" ]]; then
    red_print "GMSL driver directory not found: ${DRIVER_DIR}"
    red_print "Set GMSL_DRIVER_DIR to the SENSING SG8A driver package directory."
    exit 2
fi

cd "${DRIVER_DIR}"

for required in ko/max96712.ko ko/sgx-yuv-gmsl2.ko ko/pwm-gpio.ko boost_clock.sh; do
    if [[ ! -e "${required}" ]]; then
        red_print "Missing ${DRIVER_DIR}/${required}"
        exit 2
    fi
done

green_print "Loading camera kernel modules..."
run_as_root rmmod sgx_yuv_gmsl2 2>/dev/null || true
run_as_root rmmod max96712 2>/dev/null || true
run_as_root insmod ko/max96712.ko
run_as_root insmod ko/sgx-yuv-gmsl2.ko "enable_3G_0=${ENABLE_3G_0}" "enable_3G_1=${ENABLE_3G_1}"

green_print "Loading PWM module..."
if ! lsmod | grep -q '^pwm_gpio[[:space:]]'; then
    run_as_root insmod ko/pwm-gpio.ko 2>/dev/null || true
fi

green_print "Configuring PWM trigger..."
if [[ ! -d /sys/class/pwm/pwmchip5/pwm0 ]]; then
    write_sysfs 0 /sys/class/pwm/pwmchip5/export
fi
write_sysfs 33333333 /sys/class/pwm/pwmchip5/pwm0/period
write_sysfs 30000000 /sys/class/pwm/pwmchip5/pwm0/duty_cycle
write_sysfs 1 /sys/class/pwm/pwmchip5/pwm0/enable

green_print "Configuring video devices..."
for dev in ${VIDEO_DEVICES}; do
    if [[ ! -e "/dev/video${dev}" ]]; then
        red_print "/dev/video${dev} is missing, skip"
        continue
    fi
    v4l2-ctl \
        --set-fmt-video="width=${WIDTH},height=${HEIGHT},pixelformat=${PIXEL_FORMAT}" \
        --set-ctrl "bypass_mode=0,sensor_mode=${SENSOR_MODE},trig_mode=${TRIG_MODE},trig_pin=${TRIG_PIN}," \
        -d "/dev/video${dev}"
done

green_print "Boosting clock..."
run_as_root sh ./boost_clock.sh

green_print "Detecting I2C devices..."
run_as_root i2cdetect -y -r 9 || true
run_as_root i2cdetect -y -r 10 || true

green_print "Camera bring-up complete!"
