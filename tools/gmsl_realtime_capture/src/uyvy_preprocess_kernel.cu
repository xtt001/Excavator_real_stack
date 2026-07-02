#include "uyvy_preprocess_kernel.hpp"

#include <cuda_runtime.h>

namespace {

__device__ float clamp255(float v) {
    return fminf(255.0f, fmaxf(0.0f, v));
}

__device__ void sampleUyvyPixel(
    const unsigned char* uyvy,
    int width,
    int height,
    int pitch,
    int x,
    int y,
    float* yy,
    float* uu,
    float* vv) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    const unsigned char* row = uyvy + y * pitch;
    const int pair = (x >> 1) * 4;
    *uu = static_cast<float>(row[pair + 0]);
    *vv = static_cast<float>(row[pair + 2]);
    *yy = static_cast<float>(row[pair + ((x & 1) ? 3 : 1)]);
}

__global__ void uyvyRemapNormalizeKernel(
    const unsigned char* uyvy,
    int input_width,
    int input_height,
    int input_pitch_bytes,
    const float* map_x,
    const float* map_y,
    int output_width,
    int output_height,
    float* output_hwc_rgb,
    unsigned char* output_hwc_rgb8) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= output_width || oy >= output_height) {
        return;
    }

    const int out_index = oy * output_width + ox;
    const float sx = map_x[out_index];
    const float sy = map_y[out_index];
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (sx >= 0.0f && sy >= 0.0f && sx < static_cast<float>(input_width - 1) &&
        sy < static_cast<float>(input_height - 1)) {
        const int x0 = static_cast<int>(floorf(sx));
        const int y0 = static_cast<int>(floorf(sy));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float ax = sx - static_cast<float>(x0);
        const float ay = sy - static_cast<float>(y0);

        float y00, u00, v00;
        float y10, u10, v10;
        float y01, u01, v01;
        float y11, u11, v11;
        sampleUyvyPixel(uyvy, input_width, input_height, input_pitch_bytes, x0, y0, &y00, &u00, &v00);
        sampleUyvyPixel(uyvy, input_width, input_height, input_pitch_bytes, x1, y0, &y10, &u10, &v10);
        sampleUyvyPixel(uyvy, input_width, input_height, input_pitch_bytes, x0, y1, &y01, &u01, &v01);
        sampleUyvyPixel(uyvy, input_width, input_height, input_pitch_bytes, x1, y1, &y11, &u11, &v11);

        const float w00 = (1.0f - ax) * (1.0f - ay);
        const float w10 = ax * (1.0f - ay);
        const float w01 = (1.0f - ax) * ay;
        const float w11 = ax * ay;
        const float yy = y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11;
        const float uu = u00 * w00 + u10 * w10 + u01 * w01 + u11 * w11 - 128.0f;
        const float vv = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11 - 128.0f;

        r = clamp255(yy + 1.402f * vv) * (1.0f / 255.0f);
        g = clamp255(yy - 0.344136f * uu - 0.714136f * vv) * (1.0f / 255.0f);
        b = clamp255(yy + 1.772f * uu) * (1.0f / 255.0f);
    }

    const int base = out_index * 3;
    if (output_hwc_rgb != nullptr) {
        output_hwc_rgb[base + 0] = r;
        output_hwc_rgb[base + 1] = g;
        output_hwc_rgb[base + 2] = b;
    }
    if (output_hwc_rgb8 != nullptr) {
        output_hwc_rgb8[base + 0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
        output_hwc_rgb8[base + 1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
        output_hwc_rgb8[base + 2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
    }
}

}  // namespace

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
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);
    uyvyRemapNormalizeKernel<<<grid, block, 0, stream>>>(
        uyvy,
        input_width,
        input_height,
        input_pitch_bytes,
        map_x,
        map_y,
        output_width,
        output_height,
        output_hwc_rgb,
        nullptr);
}

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
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);
    uyvyRemapNormalizeKernel<<<grid, block, 0, stream>>>(
        uyvy,
        input_width,
        input_height,
        input_pitch_bytes,
        map_x,
        map_y,
        output_width,
        output_height,
        output_hwc_rgb,
        output_hwc_rgb8);
}
