#pragma once

#include <cuda_runtime.h>

void launchUyvyRemapNormalize(
    const unsigned char* uyvy,
    int input_width,
    int input_height,
    int input_pitch_bytes,
    const float* map_x,
    const float* map_y,
    int output_width,
    int output_height,
    float* output_hwc_rgb,
    cudaStream_t stream);

void launchUyvyRemapNormalizeAndRgb8(
    const unsigned char* uyvy,
    int input_width,
    int input_height,
    int input_pitch_bytes,
    const float* map_x,
    const float* map_y,
    int output_width,
    int output_height,
    float* output_hwc_rgb,
    unsigned char* output_hwc_rgb8,
    cudaStream_t stream);
