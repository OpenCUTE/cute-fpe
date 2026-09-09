#pragma once

#ifndef NVIDIA_ON
#define NVIDIA_ON 1
#endif

#include <cstdlib>
#include <cstring>

int32_t mymac(void *a, void *b, void *c, int size, int dtype, bool float_type);
int32_t mymac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype);

#if NVIDIA_ON

#include "mma_kernel.h"  // 包含runMma2fp32函数声明

static inline bool nv_use_native_128b_mma_input() {
    return getenv("TOPOLOGY_128_INPUT") != NULL || getenv("FP4_TOPOLOGY_128") != NULL;
}

template <typename T>
static inline void rebuild_native_128b_payload(const T* src, int physical_k, int group_count, T* dst) {
    int logical_k = physical_k / 2;
    int physical_group = physical_k / group_count;
    int logical_group = logical_k / group_count;
    for (int g = 0; g < group_count; ++g) {
        memcpy(dst + g * logical_group,
               src + g * physical_group,
               logical_group * sizeof(T));
    }
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvtf32mac(int32_t* a, int32_t* b, int32_t *c, int size, int dtype) {
    // 固定矩阵维度：M=16行, N=8列；TOPOLOGY_128_INPUT时改走原生K=4的TF32指令。
    const int M = 16, N = 8;
    const bool native_128b = nv_use_native_128b_mma_input();
    const int K = native_128b ? 4 : 8;
    const int group_count = 2;

    // 1. 初始化指针为nullptr，避免野指针
    cutlass::tfloat32_t *h_A = nullptr, *h_B = nullptr;
    float *h_C32 = nullptr;
    cutlass::tfloat32_t *d_A = nullptr, *d_B = nullptr;
    float *d_C32 = nullptr;

    // 2. 分配主机内存（增加内存分配失败检查）
    h_A = (cutlass::tfloat32_t*)malloc(M*K*sizeof(cutlass::tfloat32_t));
    h_B = (cutlass::tfloat32_t*)malloc(K*N*sizeof(cutlass::tfloat32_t));
    h_C32 = (float*)malloc(M*N*sizeof(float));

    cudaMalloc(&d_A, M*K*sizeof(cutlass::tfloat32_t));
    cudaMalloc(&d_B, K*N*sizeof(cutlass::tfloat32_t));
    cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i] = cutlass::tfloat32_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i] = cutlass::tfloat32_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    if (native_128b) {
        int32_t a_native[4] = {0};
        int32_t b_native[4] = {0};
        rebuild_native_128b_payload(a, size, group_count, a_native);
        rebuild_native_128b_payload(b, size, group_count, b_native);
        for(int i=0; i<valid_size; i++) {
            *((uint32_t*)&h_A[i]) = (uint32_t)a_native[i];
        }
        for(int i=0; i<valid_size; i++) {
            *((uint32_t*)&h_B[i*N]) = (uint32_t)b_native[i];
        }
    } else {
        // a填充到h_A第一行（h_A[0] ~ h_A[7]）
        for(int i=0; i<valid_size; i++) {
            *((uint32_t*)&h_A[i]) = (uint32_t)a[i]; // 保留原始bit位
        }
        // b填充到h_B第一列（h_B[0], h_B[8], h_B[16]...h_B[56]）
        for(int i=0; i<valid_size; i++) {
            *((uint32_t*)&h_B[i*N]) = (uint32_t)b[i]; // 保留原始bit位
        }
    }

    // 6. 初始化h_C32（可选：用c的第一个值初始化）
    if (c != nullptr) {
        *((uint32_t*)&h_C32[0]) = (uint32_t)c[0];
    } else {
        *((uint32_t*)&h_C32[0]) = 0x2f5896f0; // 默认初始值
    }

    // 7. 执行CUDA TF32 MAC计算
    runMma2fp32<cutlass::tfloat32_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        dtype, 0
    );

    // 8. 获取结果的原始bit（转换为int32_t返回）
    int32_t result_bits = 0; // 改为int32_t类型
    memcpy(&result_bits, &h_C32[0], sizeof(int32_t)); // bit位完全拷贝

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp16mac(int16_t* a, int16_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8;
    const bool native_128b = nv_use_native_128b_mma_input();
    const int K = native_128b ? 8 : 16;
    const int group_count = 2;

    //分配内存空间
    cutlass::half_t *h_A   = (cutlass::half_t*)malloc(M*K*sizeof(cutlass::half_t));
    cutlass::half_t *h_B   = (cutlass::half_t*)malloc(K*N*sizeof(cutlass::half_t));
    cutlass::half_t *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float           *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::half_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::half_t));
    cutlass::half_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::half_t));
    cutlass::half_t *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float           *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i] = cutlass::half_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    if (native_128b) {
        int16_t a_native[8] = {0};
        int16_t b_native[8] = {0};
        rebuild_native_128b_payload(a, size, group_count, a_native);
        rebuild_native_128b_payload(b, size, group_count, b_native);
        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_A[i], &a_native[i], sizeof(uint16_t));
        }

        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_B[i * N], &b_native[i], sizeof(uint16_t));
        }
    } else {
        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_A[i], &a[i], sizeof(uint16_t));
        }

        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_B[i * N], &b[i], sizeof(uint16_t));
        }
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::half_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::half_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}


// 返回值改为int32_t，保持原始bit位不变
int32_t nvbf16mac(int16_t* a, int16_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8;
    const bool native_128b = nv_use_native_128b_mma_input();
    const int K = native_128b ? 8 : 16;
    const int group_count = 2;

    //分配内存空间
    cutlass::bfloat16_t *h_A   = (cutlass::bfloat16_t*)malloc(M*K*sizeof(cutlass::bfloat16_t));
    cutlass::bfloat16_t *h_B   = (cutlass::bfloat16_t*)malloc(K*N*sizeof(cutlass::bfloat16_t));
    float               *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::bfloat16_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::bfloat16_t));
    cutlass::bfloat16_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::bfloat16_t));
    float               *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i] = cutlass::bfloat16_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i] = cutlass::bfloat16_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    if (native_128b) {
        int16_t a_native[8] = {0};
        int16_t b_native[8] = {0};
        rebuild_native_128b_payload(a, size, group_count, a_native);
        rebuild_native_128b_payload(b, size, group_count, b_native);
        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_A[i], &a_native[i], sizeof(uint16_t));
        }

        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_B[i * N], &b_native[i], sizeof(uint16_t));
        }
    } else {
        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_A[i], &a[i], sizeof(uint16_t));
        }

        for(int i = 0; i < valid_size; i++) {
            memcpy(&h_B[i * N], &b[i], sizeof(uint16_t));
        }
    }

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32<cutlass::bfloat16_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    int32_t result_bits = 0; // 改为int32_t类型
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp8e4m3mac(int8_t* a, int8_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e4m3_t *h_A   = (cutlass::float_e4m3_t*)malloc(M*K*sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t *h_B   = (cutlass::float_e4m3_t*)malloc(K*N*sizeof(cutlass::float_e4m3_t));
    cutlass::half_t       *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e4m3_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e4m3_t));
    cutlass::half_t       *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e4m3_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e4m3_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::float_e4m3_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::float_e4m3_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp8e5m2mac(int8_t* a, int8_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e5m2_t *h_A   = (cutlass::float_e5m2_t*)malloc(M*K*sizeof(cutlass::float_e5m2_t));
    cutlass::float_e5m2_t *h_B   = (cutlass::float_e5m2_t*)malloc(K*N*sizeof(cutlass::float_e5m2_t));
    cutlass::half_t       *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e5m2_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e5m2_t));
    cutlass::float_e5m2_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e5m2_t));
    cutlass::half_t       *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e5m2_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e5m2_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::float_e5m2_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::float_e5m2_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp6e2m3mac(int8_t* a, int8_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e2m3_t *h_A   = (cutlass::float_e2m3_t*)malloc(M*K*sizeof(cutlass::float_e2m3_t));
    cutlass::float_e2m3_t *h_B   = (cutlass::float_e2m3_t*)malloc(K*N*sizeof(cutlass::float_e2m3_t));
    cutlass::half_t       *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e2m3_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e2m3_t));
    cutlass::float_e2m3_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e2m3_t));
    cutlass::half_t       *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e2m3_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e2m3_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::float_e2m3_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::float_e2m3_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp6e3m2mac(int8_t* a, int8_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e3m2_t *h_A   = (cutlass::float_e3m2_t*)malloc(M*K*sizeof(cutlass::float_e3m2_t));
    cutlass::float_e3m2_t *h_B   = (cutlass::float_e3m2_t*)malloc(K*N*sizeof(cutlass::float_e3m2_t));
    cutlass::half_t       *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e3m2_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e3m2_t));
    cutlass::float_e3m2_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e3m2_t));
    cutlass::half_t       *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e3m2_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e3m2_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    //填充
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::float_e3m2_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::float_e3m2_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvfp4e2m1mac(int8_t* a, int8_t* b, int32_t *c, int size, int dtype, bool float_type = 0) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e2m1_t *h_A   = (cutlass::float_e2m1_t*)malloc(M*K*sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *h_B   = (cutlass::float_e2m1_t*)malloc(K*N*sizeof(cutlass::float_e2m1_t));
    cutlass::half_t       *h_C16 = (cutlass::half_t*)malloc(M*N*sizeof(cutlass::half_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e2m1_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e2m1_t));
    cutlass::half_t       *d_C16; cudaMalloc(&d_C16, M*N*sizeof(cutlass::half_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e2m1_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e2m1_t(0.0f);
    for(int i=0; i<M*N; i++) h_C16[i] = cutlass::half_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A（第一行）
    for(int i = 0; i < valid_size; i++) {
        uint8_t tmp = ((uint8_t)a[i]) >> 2;  // 逻辑右移2位
        memcpy(&h_A[i], &tmp, sizeof(uint8_t));
    }

    // 填充 B（第一列）
    for(int i = 0; i < valid_size; i++) {
        uint8_t tmp = ((uint8_t)b[i]) >> 2;  // 逻辑右移2位
        memcpy(&h_B[i * N], &tmp, sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    if (!float_type) {
        // ---------------- FP32 路径 ----------------
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32<cutlass::float_e2m1_t>(
            M, N, K,
            h_A, h_B, h_C32,
            d_A, d_B, d_C32,
            dtype, 0
        );

        // 8. 获取结果bit（32bit）
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    }
    else {
        // ---------------- FP16 路径 ----------------
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16<cutlass::float_e2m1_t>(
            M, N, K,
            h_A, h_B, h_C16,
            d_A, d_B, d_C16,
            dtype, 0
        );

        // 8. 获取结果bit（16bit → 放入int32_t低位）
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));

        result_bits = (int32_t)result16;  // 高16位为0
    }

    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}


// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp8e4m3mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e4m3_t *h_A   = (cutlass::float_e4m3_t*)malloc(M*K*sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t *h_B   = (cutlass::float_e4m3_t*)malloc(K*N*sizeof(cutlass::float_e4m3_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e4m3_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e4m3_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e4m3_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e4m3_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale<cutlass::float_e4m3_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    
    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp8e5m2mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e5m2_t *h_A   = (cutlass::float_e5m2_t*)malloc(M*K*sizeof(cutlass::float_e5m2_t));
    cutlass::float_e5m2_t *h_B   = (cutlass::float_e5m2_t*)malloc(K*N*sizeof(cutlass::float_e5m2_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e5m2_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e5m2_t));
    cutlass::float_e5m2_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e5m2_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e5m2_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e5m2_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale<cutlass::float_e5m2_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    
    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp6e2m3mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e2m3_t *h_A   = (cutlass::float_e2m3_t*)malloc(M*K*sizeof(cutlass::float_e2m3_t));
    cutlass::float_e2m3_t *h_B   = (cutlass::float_e2m3_t*)malloc(K*N*sizeof(cutlass::float_e2m3_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e2m3_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e2m3_t));
    cutlass::float_e2m3_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e2m3_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e2m3_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e2m3_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale<cutlass::float_e2m3_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    
    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp6e3m2mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e3m2_t *h_A   = (cutlass::float_e3m2_t*)malloc(M*K*sizeof(cutlass::float_e3m2_t));
    cutlass::float_e3m2_t *h_B   = (cutlass::float_e3m2_t*)malloc(K*N*sizeof(cutlass::float_e3m2_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e3m2_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e3m2_t));
    cutlass::float_e3m2_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e3m2_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e3m2_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e3m2_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale<cutlass::float_e3m2_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    
    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp4e2m1mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 32;

    //分配内存空间
    cutlass::float_e2m1_t *h_A   = (cutlass::float_e2m1_t*)malloc(M*K*sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *h_B   = (cutlass::float_e2m1_t*)malloc(K*N*sizeof(cutlass::float_e2m1_t));
    float                 *h_C32 = (float*)malloc(M*N*sizeof(float));

    cutlass::float_e2m1_t *d_A;   cudaMalloc(&d_A, M*K*sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *d_B;   cudaMalloc(&d_B, K*N*sizeof(cutlass::float_e2m1_t));
    float                 *d_C32; cudaMalloc(&d_C32, M*N*sizeof(float));

    // 4. 初始化内存为0（避免脏数据）
    for(int i=0; i<M*K; i++) h_A[i]   = cutlass::float_e2m1_t(0.0f);
    for(int i=0; i<K*N; i++) h_B[i]   = cutlass::float_e2m1_t(0.0f);
    for(int i=0; i<M*N; i++) h_C32[i] = 0.0f;

    // 5. 核心：将a/b的原始bit数据填充到h_A第一行、h_B第一列
    int valid_size = (size > K) ? K : size; // 限制不超过K=8，避免越界
    // 填充 A
    for(int i = 0; i < valid_size; i++) {
        uint8_t tmp = ((uint8_t)a[i]) >> 2;  // 逻辑右移2位
        memcpy(&h_A[i], &tmp, sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        uint8_t tmp = ((uint8_t)b[i]) >> 2;  // 逻辑右移2位
        memcpy(&h_B[i * N], &tmp, sizeof(uint8_t));
    }

    // 根据float_type来填充不同的C
    int32_t result_bits = 0; // 最终返回值（统一用int32_t装）

    // ---------------- FP32 路径 ----------------
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale<cutlass::float_e2m1_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0
    );

    // 8. 获取结果bit（32bit）
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    
    // 9. 释放所有内存（避免泄漏）
    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);

    // 10. 返回int32_t类型的原始bit值
    return result_bits;
}

// 每个输入字节包含两个4bit(e2m1)，这里展开成kernel需要的逐元素布局。
static inline void unpack_fp4e2m1_to_kernel_layout(
    const int8_t* src, int valid_size, cutlass::float_e2m1_t* dst, int stride) {
    for (int i = 0; i < valid_size; ++i) {
        uint8_t val = (uint8_t)src[i];
        uint8_t low = val & 0x0F;
        uint8_t high = (val >> 4) & 0x0F;
        memcpy(&dst[(2 * i) * stride], &low, sizeof(uint8_t));
        memcpy(&dst[(2 * i + 1) * stride], &high, sizeof(uint8_t));
    }
}

// dtype9/10共用主流程：仅通过拓扑参数控制scale解释方式。
static inline int32_t nvmxf4_common_mac_impl(
    int8_t* a, int8_t* a_scale,
    int8_t* b, int8_t* b_scale,
    int32_t* c, int size, int dtype,
    int mxf4_topology_128, int blocksize16) {
    const int M = 16, N = 8, K = 64;

    cutlass::float_e2m1_t *h_A = (cutlass::float_e2m1_t*)malloc(M * K * sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *h_B = (cutlass::float_e2m1_t*)malloc(K * N * sizeof(cutlass::float_e2m1_t));
    float *h_C32 = (float*)malloc(M * N * sizeof(float));

    cutlass::float_e2m1_t *d_A; cudaMalloc(&d_A, M * K * sizeof(cutlass::float_e2m1_t));
    cutlass::float_e2m1_t *d_B; cudaMalloc(&d_B, K * N * sizeof(cutlass::float_e2m1_t));
    float *d_C32; cudaMalloc(&d_C32, M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i) h_A[i] = cutlass::float_e2m1_t(0.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = cutlass::float_e2m1_t(0.0f);
    for (int i = 0; i < M * N; ++i) h_C32[i] = 0.0f;

    int valid_size = (size > (K / 2)) ? (K / 2) : size;
    unpack_fp4e2m1_to_kernel_layout(a, valid_size, h_A, 1);
    unpack_fp4e2m1_to_kernel_layout(b, valid_size, h_B, N);

    int32_t result_bits = 0;
    memcpy(&h_C32[0], &c[0], sizeof(uint32_t));
    runMma2fp32_scale<cutlass::float_e2m1_t>(
        M, N, K,
        h_A, h_B, h_C32,
        d_A, d_B, d_C32,
        a_scale, b_scale,
        dtype, 0, mxf4_topology_128, blocksize16);
    memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));

    free(h_A); free(h_B); free(h_C32);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32);
    return result_bits;
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp4mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    return nvmxf4_common_mac_impl(
        a, a_scale, b, b_scale, c, size, dtype,
        nv_use_native_128b_mma_input() ? 1 : 0,
        mma_mxf4_blocksize16_enabled() ? 1 : 0);
}

// 返回值改为int32_t，保持原始bit位不变
int32_t nvnvfp4mac_4X(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    return nvmxf4_common_mac_impl(
        a, a_scale, b, b_scale, c, size, dtype,
        nv_use_native_128b_mma_input() ? 1 : 0,
        mma_mxf4_blocksize16_enabled() ? 1 : 0);
}

static inline void mma_shape_params(const MmaDescriptor& desc, int* M, int* N, int* K) {
    *N = 8;
    switch (desc.shape) {
        case MmaShape::M8N8K16:
            *M = 8; *K = 16; break;
        case MmaShape::M8N8K32:
            *M = 8; *K = 32; break;
        case MmaShape::M16N8K8:
            *M = 16; *K = 8; break;
        case MmaShape::M16N8K16:
            *M = 16; *K = 16; break;
        case MmaShape::M16N8K32:
            *M = 16; *K = 32; break;
        case MmaShape::M16N8K64:
            *M = 16; *K = 64; break;
        default:
            *M = 16; *K = 0; break;
    }
}

static inline uint8_t normalize_raw_byte_for_kernel(uint8_t raw, MmaOperandKind kind) {
    return (kind == MmaOperandKind::E2M1) ? (raw >> 2) : raw;
}

static inline int32_t nv_f8f6f4_mixed_mac(int8_t* a, int8_t* b, int32_t* c, int dtype, bool float_type) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc) return 0;
    int M = 0, N = 0, K = 0;
    mma_shape_params(*desc, &M, &N, &K);

    uint8_t* h_A = (uint8_t*)malloc(M * K);
    uint8_t* h_B = (uint8_t*)malloc(K * N);
    uint8_t* d_A = nullptr;
    uint8_t* d_B = nullptr;
    float* h_C32 = nullptr;
    float* d_C32 = nullptr;
    cutlass::half_t* h_C16 = nullptr;
    cutlass::half_t* d_C16 = nullptr;
    cudaMalloc(&d_A, M * K);
    cudaMalloc(&d_B, K * N);

    memset(h_A, 0, M * K);
    memset(h_B, 0, K * N);

    for (int i = 0; i < K; ++i) {
        h_A[i] = normalize_raw_byte_for_kernel((uint8_t)a[i], desc->a_kind);
        h_B[i * N] = normalize_raw_byte_for_kernel((uint8_t)b[i], desc->b_kind);
    }
    int32_t result_bits = 0;
    if (!float_type) {
        h_C32 = (float*)malloc(M * N * sizeof(float));
        cudaMalloc(&d_C32, M * N * sizeof(float));
        memset(h_C32, 0, M * N * sizeof(float));
        memcpy(&h_C32[0], &c[0], sizeof(uint32_t));

        runMma2fp32_mixed_f8f6f4(M, N, K, h_A, h_B, h_C32, d_A, d_B, d_C32,
                                 (int)desc->a_kind, (int)desc->b_kind, 0);
        memcpy(&result_bits, &h_C32[0], sizeof(uint32_t));
    } else {
        h_C16 = (cutlass::half_t*)malloc(M * N * sizeof(cutlass::half_t));
        cudaMalloc(&d_C16, M * N * sizeof(cutlass::half_t));
        memset(h_C16, 0, M * N * sizeof(cutlass::half_t));
        uint16_t low16 = (uint16_t)(c[0] & 0xFFFF);
        memcpy(&h_C16[0], &low16, sizeof(uint16_t));

        runMma2fp16_mixed_f8f6f4(M, N, K, h_A, h_B, h_C16, d_A, d_B, d_C16,
                                 (int)desc->a_kind, (int)desc->b_kind, 0);
        uint16_t result16 = 0;
        memcpy(&result16, &h_C16[0], sizeof(uint16_t));
        result_bits = (int32_t)result16;
    }

    free(h_A); free(h_B);
    free(h_C32); free(h_C16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C32); cudaFree(d_C16);
    return result_bits;
}

static inline int32_t nvmxf8f6f4_mixed_scale_mac(
    int8_t* a,
    int8_t* a_scale,
    int8_t* b,
    int8_t* b_scale,
    int32_t* c,
    int dtype)
{
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc) return 0;

    int M = 0, N = 0, K = 0;
    mma_shape_params(*desc, &M, &N, &K);

    uint8_t* h_A = (uint8_t*)malloc(M * K);
    uint8_t* h_B = (uint8_t*)malloc(K * N);
    float* h_C = (float*)malloc(M * N * sizeof(float));
    uint8_t* d_A = nullptr;
    uint8_t* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, M * K);
    cudaMalloc(&d_B, K * N);
    cudaMalloc(&d_C, M * N * sizeof(float));

    memset(h_A, 0, M * K);
    memset(h_B, 0, K * N);
    memset(h_C, 0, M * N * sizeof(float));

    for (int i = 0; i < K; ++i) {
        h_A[i] = normalize_raw_byte_for_kernel((uint8_t)a[i], desc->a_kind);
        h_B[i * N] = normalize_raw_byte_for_kernel((uint8_t)b[i], desc->b_kind);
    }
    memcpy(&h_C[0], &c[0], sizeof(uint32_t));

    runMma2fp32_scale_mixed_f8f6f4(M, N, K,
                                   h_A, h_B, h_C,
                                   d_A, d_B, d_C,
                                   a_scale, b_scale,
                                   (int)desc->a_kind, (int)desc->b_kind, 0);

    int32_t result_bits = 0;
    memcpy(&result_bits, &h_C[0], sizeof(uint32_t));
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return result_bits;
}

static inline int32_t nv_integer_mac_tile(
    const int8_t* a,
    const int8_t* b,
    int32_t c_value,
    const MmaDescriptor& hw_desc,
    int input_k)
{
    int M = 0, N = 0, K = 0;
    mma_shape_params(hw_desc, &M, &N, &K);
    if (K <= 0) return c_value;

    int copy_k = input_k < K ? input_k : K;
    if (copy_k < 0) copy_k = 0;

    int8_t* h_A = (int8_t*)malloc(M * K);
    int8_t* h_B = (int8_t*)malloc(K * N);
    int32_t* h_C = (int32_t*)malloc(M * N * sizeof(int32_t));
    int8_t* d_A = nullptr;
    int8_t* d_B = nullptr;
    int32_t* d_C = nullptr;
    if (!h_A || !h_B || !h_C) {
        free(h_A); free(h_B); free(h_C);
        return c_value;
    }
    cudaMalloc(&d_A, M * K);
    cudaMalloc(&d_B, K * N);
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    memset(h_A, 0, M * K);
    memset(h_B, 0, K * N);
    memset(h_C, 0, M * N * sizeof(int32_t));
    for (int i = 0; i < copy_k; ++i) {
        h_A[i] = a[i];
        h_B[i * N] = b[i];
    }
    h_C[0] = c_value;

    runMma2s32_integer(M, N, K, h_A, h_B, h_C, d_A, d_B, d_C,
                       (int)hw_desc.shape, (int)hw_desc.a_kind, (int)hw_desc.b_kind,
                       hw_desc.satfinite, 0);

    int32_t result_bits = h_C[0];
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return result_bits;
}

static inline int32_t nv_integer_mac(int8_t* a, int8_t* b, int32_t* c, int size, int dtype) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc || !c) return 0;

    // A custom RTL int4 K is a scalar dot product, not a single fixed MMA
    // tile.  Execute native K=32/K=64 instructions and feed each tile's D[0]
    // back as the next tile's C[0].  The final partial tile is zero padded.
    if (mma_is_rtl_int4_dtype(dtype)) {
        int32_t acc = c[0];
        int offset = 0;
        while (offset < size) {
            int remaining = size - offset;
            int tile_input_k = remaining > 64 ? 64 : remaining;
            MmaDescriptor tile_desc = *desc;
            tile_desc.shape = tile_input_k <= 32 ? MmaShape::M8N8K32 : MmaShape::M16N8K64;
            acc = nv_integer_mac_tile(a + offset, b + offset, acc, tile_desc, tile_input_k);
            offset += tile_input_k;
        }
        return acc;
    }

    MmaDescriptor hw_desc = *desc;
    if (desc->shape == MmaShape::M16N8K16 &&
        (desc->a_kind == MmaOperandKind::S8 || desc->a_kind == MmaOperandKind::U8) &&
        (desc->b_kind == MmaOperandKind::S8 || desc->b_kind == MmaOperandKind::U8)) {
        hw_desc.shape = MmaShape::M8N8K16;
    } else if (desc->shape == MmaShape::M16N8K32 &&
               (desc->a_kind == MmaOperandKind::S4 || desc->a_kind == MmaOperandKind::U4) &&
               (desc->b_kind == MmaOperandKind::S4 || desc->b_kind == MmaOperandKind::U4)) {
        hw_desc.shape = MmaShape::M8N8K32;
    }
    return nv_integer_mac_tile(a, b, c[0], hw_desc, mma_shape_k(hw_desc.shape));
}

static inline bool nv_descriptor_uses_integer_mma(const MmaDescriptor& desc) {
    if (desc.category == MmaCategory::Integer) {
        return true;
    }
    if (desc.category != MmaCategory::Legacy) {
        return false;
    }
    return (desc.a_kind == MmaOperandKind::S8 || desc.a_kind == MmaOperandKind::U8 ||
            desc.a_kind == MmaOperandKind::S4 || desc.a_kind == MmaOperandKind::U4) &&
           (desc.b_kind == MmaOperandKind::S8 || desc.b_kind == MmaOperandKind::U8 ||
            desc.b_kind == MmaOperandKind::S4 || desc.b_kind == MmaOperandKind::U4);
}

// 统一调用runMma2fp32
int32_t nvmac(void *a, void *b, void *c, int size, int dtype, bool float_type = 0) {
  const MmaDescriptor* desc = mma_get_descriptor(dtype);
  if (desc && desc->category == MmaCategory::MixedFloat) {
      return nv_f8f6f4_mixed_mac((int8_t*)a, (int8_t*)b, (int32_t*)c, dtype, float_type);
  } else if (desc && nv_descriptor_uses_integer_mma(*desc)) {
      return nv_integer_mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype);
  } else if (dtype == 1) { // fp16
      return nvfp16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 2) { // bf16
      return nvbf16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, dtype);
  } else if (dtype == 3) { // tf32
      return nvtf32mac((int32_t*)a, (int32_t*)b, (int32_t*)c, size, dtype);
  }
  return 0; // 错误处理
}

// 统一调用runMma2fp32_scale
int32_t nvmac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (desc && desc->category == MmaCategory::MixedFloat && desc->uses_scale) {
        return nvmxf8f6f4_mixed_scale_mac((int8_t*)a, (int8_t*)a_scale,
                                          (int8_t*)b, (int8_t*)b_scale,
                                          (int32_t*)c, dtype);
    } else if (dtype == 9) { // mxf4nvf4::nvfp4
        return nvnvfp4mac_4X((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } 
    else if (dtype == 10) { // mxf4nvf4::mxfp4
        return nvmxfp4mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    }
    return 0; // 错误处理
}

#else

inline int32_t nvmac(void *a, void *b, void *c, int size, int dtype, bool float_type = 0) {
    return mymac(a, b, c, size, dtype, float_type);
}

inline int32_t nvmac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype) {
    return mymac_scale(a, a_scale, b, b_scale, c, size, dtype);
}

#endif
