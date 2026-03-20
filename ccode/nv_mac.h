#include "mma_kernel.h"  // 包含runMma2fp32函数声明
#include <cstring>            // memcpy依赖

// 返回值改为int32_t，保持原始bit位不变
int32_t nvtf32mac(int32_t* a, int32_t* b, int32_t *c, int size, int dtype) {
    // 固定矩阵维度：M=16行, N=8列, K=8（A的列数/B的行数）
    const int M = 16, N = 8, K = 8;

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
    // a填充到h_A第一行（h_A[0] ~ h_A[7]）
    for(int i=0; i<valid_size; i++) {
        *((uint32_t*)&h_A[i]) = (uint32_t)a[i]; // 保留原始bit位
    }
    // b填充到h_B第一列（h_B[0], h_B[8], h_B[16]...h_B[56]）
    for(int i=0; i<valid_size; i++) {
        *((uint32_t*)&h_B[i*N]) = (uint32_t)b[i]; // 保留原始bit位
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
    const int M = 16, N = 8, K = 16;

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
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint16_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint16_t));
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
    const int M = 16, N = 8, K = 16;

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
    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_A[i], &a[i], sizeof(uint16_t));
    }

    for(int i = 0; i < valid_size; i++) {
        memcpy(&h_B[i * N], &b[i], sizeof(uint16_t));
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

// 返回值改为int32_t，保持原始bit位不变
int32_t nvmxfp4mac_2X(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 64;

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

    // 5. 现在的a[i],b[i]为两个e2m1的拼起来，我现在需要把低四位先传入h_A[i]然后再将高4位传入h_A[i+1]。h_B也以此类推
    int valid_size = (size > (K/2)) ? (K/2) : size; // 限制不超过K=8，避免越界
    // 填充 A
    // 每个 a[i] 拆成两个
    for(int i = 0; i < valid_size; i++) {
        uint8_t val = (uint8_t)a[i];

        uint8_t low  = val & 0x0F;        // 低4位
        uint8_t high = (val >> 4) & 0x0F; // 高4位

        memcpy(&h_A[2 * i], &low, sizeof(uint8_t));
        memcpy(&h_A[2 * i + 1], &high, sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        uint8_t val = (uint8_t)b[i];

        uint8_t low  = val & 0x0F;
        uint8_t high = (val >> 4) & 0x0F;

        memcpy(&h_B[(2 * i) * N], &low, sizeof(uint8_t));
        memcpy(&h_B[(2 * i + 1) * N], &high, sizeof(uint8_t));
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


// 返回值改为int32_t，保持原始bit位不变
int32_t nvnvfp4mac_4X(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size, int dtype) {
    const int M = 16, N = 8, K = 64;

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

    // 5. 现在的a[i],b[i]为两个e2m1的拼起来，我现在需要把低四位先传入h_A[i]然后再将高4位传入h_A[i+1]。h_B也以此类推
    int valid_size = (size > (K/2)) ? (K/2) : size; // 限制不超过K=8，避免越界
    // 填充 A
    // 每个 a[i] 拆成两个
    for(int i = 0; i < valid_size; i++) {
        uint8_t val = (uint8_t)a[i];

        uint8_t low  = val & 0x0F;        // 低4位
        uint8_t high = (val >> 4) & 0x0F; // 高4位

        memcpy(&h_A[2 * i], &low, sizeof(uint8_t));
        memcpy(&h_A[2 * i + 1], &high, sizeof(uint8_t));
    }

    for(int i = 0; i < valid_size; i++) {
        uint8_t val = (uint8_t)b[i];

        uint8_t low  = val & 0x0F;
        uint8_t high = (val >> 4) & 0x0F;

        memcpy(&h_B[(2 * i) * N], &low, sizeof(uint8_t));
        memcpy(&h_B[(2 * i + 1) * N], &high, sizeof(uint8_t));
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


int32_t nvmac(void *a, void *b, void *c, int size, int dtype, bool float_type = 0) {
  if (dtype == 1) { // fp16
      return nvfp16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 2) { // bf16
      return nvbf16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, dtype);
  } else if (dtype == 3) { // tf32
      return nvtf32mac((int32_t*)a, (int32_t*)b, (int32_t*)c, size, dtype);
  } else if (dtype == 13) { //FP8E4M3
      return nvfp8e4m3mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 14) { //FP8E5M2
      return nvfp8e5m2mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 15) { //FP6E2M3
      return nvfp6e2m3mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 16) { //FP6E2M3
      return nvfp6e3m2mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype, float_type);
  } else if (dtype == 18) { //FP4E2M1
      return nvfp4e2m1mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, dtype, float_type);
  }
  return 0; // 错误处理
}


int32_t nvmac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype) {
    if (dtype == 7) { //MXFP8E4M3(带scale)
        return nvmxfp8e4m3mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } else if (dtype == 8) { //MXFP8E5M2(带scale)
        return nvmxfp8e5m2mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    }
    else if (dtype == 9) { // nvfp4
        return nvnvfp4mac_4X((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } 
    else if (dtype == 10) { // mxfp4
        return nvmxfp4mac_2X((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } 
    else if (dtype == 11) { // mxfp6e2m3
        return nvmxfp6e2m3mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } 
    else if (dtype == 12) { // mxfp6e3m2
        return nvmxfp6e3m2mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    } 
    else if (dtype == 17) { // fp4e2m1
        return nvmxfp4e2m1mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size, dtype);
    }
    return 0; // 错误处理
}
