#ifndef MMA_TF32_KERNEL_H_
#define MMA_TF32_KERNEL_H_

// 必要的CUDA和Cutlass头文件（封装时需包含依赖）
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cute/numeric/numeric_types.hpp>
#include <vector>
#include <cmath>

// 1. 模板结构体封装（保留__device__函数）

//A矩阵形状
template <typename T> struct Afrag_16x64 {
  static constexpr size_t ne = 32; // num of elements per thread

  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * ((l / 8) % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return (tid % 4)*8 + (l % 8) + 32 * (l / 16);
  }
};

template <typename T> struct Afrag_16x32 {
  //static表示，ne属于该Afrag_16x32类型的所有对象，而不会根据不同对象而改变其值
  //constexpr：表示在编译的时候为一个常量；size_t则表示其为无符号整数类型
  static constexpr size_t ne = 16; // num of elements per thread

  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * ((l / 4) % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return (tid % 4)*4 + (l % 4) + 16 * (l / 8);
  }
};

template <typename T> struct Afrag_16x16 {
  static constexpr size_t ne = 8; // num of elements per thread

  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * ((l / 2) % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + (l % 2) + 8 * (l / 4);
  }
};

template <typename T> 
struct Afrag_16x8 {
  static constexpr size_t ne = 4; // num of elements per thread
  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * (l % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return tid % 4 + 4 * (l / 2);
  }
};

//B矩阵形状
template <typename T> struct Bfrag_64x8 {
  static constexpr size_t ne = 16;
  T x[ne] = {};
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) * 8 + (l % 8) + 32 * (l / 8);
  }

  static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

template <typename T> struct Bfrag_32x8 {
  static constexpr size_t ne = 8;
  T x[ne] = {};
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) * 4 + (l % 4) + 16 * (l / 4);
  }

  static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

template <typename T> 
struct Bfrag_8x8 {
  static constexpr size_t ne = 2;
  T x[ne] = {};
  
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) + 4 * l;
  }

  static __device__ size_t get_col(int tid, int l) { 
    return tid >> 2; 
  }
};

template <typename T> struct Bfrag_16x8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) * 2 + (l % 2) + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

template <typename T> 
struct CFrag_16x8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    return (tid >> 2) + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + (l % 2);
  }
};

__device__ __forceinline__
uint32_t pack8_fp4_to_u32(const cutlass::float_e2m1_t *src)
{
    uint32_t v = 0;

    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t raw = p[i] & 0xF;     // 取4bit

        v |= (uint32_t(raw) << (i * 4));
    }

    return v;
}

__host__ __device__ __forceinline__ uint16_t pack_ue8m0x2(uint8_t sf0, uint8_t sf1) {
  // sf0 -> low 8 bits, sf1 -> high 8 bits
  return static_cast<uint16_t>(sf0) | (static_cast<uint16_t>(sf1) << 8);
}

__host__ __device__ __forceinline__ uint32_t pack_ue4m3x4(
      cutlass::float_ue4m3_t a,
      cutlass::float_ue4m3_t b,
      cutlass::float_ue4m3_t c,
      cutlass::float_ue4m3_t d)
  {
      // Use UE4M3 raw storage bytes, not numeric integer conversion.
      return uint32_t(a.raw()) |
            (uint32_t(b.raw()) << 8) |
            (uint32_t(c.raw()) << 16) |
            (uint32_t(d.raw()) << 24);
  }

__device__ __forceinline__
uint32_t pack_e2m3_to_u32(const cutlass::float_e2m3_t *src)
{
    uint32_t v = 0;

    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint8_t raw = p[i] & 0x3F;     // 取6bit

        v |= (uint32_t(raw) << (i * 8));
    }

    return v;
}

__device__ __forceinline__
uint32_t pack_e3m2_to_u32(const cutlass::float_e3m2_t *src)
{
    uint32_t v = 0;

    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint8_t raw = p[i] & 0x3F;     // 取6bit

        v |= (uint32_t(raw) << (i * 8));
    }

    return v;
}


//将2个fp16的元素打包到一个32寄存器中
__device__ __forceinline__
uint32_t pack4_fp16_to_u32(const cutlass::half_t *src)
{
    uint32_t v = 0;

    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint16_t raw = p[i] & 0xFFFF;     // 取16bit
        v |= (uint32_t(raw) << (i * 16));
    }

    return v;
}

//将一个32位的寄存器，解包成两个fp16元素
__device__ __forceinline__
void unpack4_u32_to_fp16(uint32_t v, cutlass::half_t* dst)
{
    union { uint32_t u; __half2 h2; } tmp;
    tmp.u = v;
    dst[0] = __low2half(tmp.h2);
    dst[1] = __high2half(tmp.h2);
}

__device__ __forceinline__
uint32_t pack4_fp4_to_u32(const cutlass::float_e2m1_t *src)
{
    uint32_t v = 0;

    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint8_t raw = p[i] & 0xF;     // 取4bit
        uint8_t mid = raw << 2;      // 放到bit[5:2]，前后补0
        // uint8_t mid = (raw << 2) | 0b00000001;      // 验证e2m1是否共用e2m3的数据通路

        v |= (uint32_t(mid) << (i * 8));
    }

    return v;
}

// 3. 核函数实现（模板函数需在头文件中实现，否则链接报错）
template <typename T>
__global__ void mma2fp32(const T *A, const T *B, float *C, int M, int N, int K, int dtype) {

  switch(dtype){
    //FP16 and BF16
    case 1:
    case 2: {
      //Each thread has a copy of this tile
      Afrag_16x16<T> a_tile;
      Bfrag_16x8<T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;

      if(dtype == 1){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else if(dtype == 2){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else {
        printf("dtype is error, dtype is %d",dtype);
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);

        C[row * N + col] = c_tile.x[i];
      }

      break;
    }
    //TF32
    case 3: {
      //Each thread has a copy of this tile
      Afrag_16x8<T> a_tile;
      Bfrag_8x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
          c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;

      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                  "{%0, %1, %2, %3}, "
                  "{%4, %5, %6, %7}, "
                  "{%8, %9}, "
                  "{%0, %1, %2, %3};\n"
                  : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                    "+f"(c_tile.x[3])
                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                    "r"(b_regs[0]), "r"(b_regs[1]));

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP8E4M3 or FP8E5M2
    case 13: 
    case 14: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;

      if(dtype == 13){
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else if(dtype == 14){
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else {
        printf("dtype is error, dtype is %d",dtype);
      }
      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP6E2M3 or FP6E3M2
    case 15:
    case 16: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      if constexpr (std::is_same_v<T, cutlass::float_e2m3_t>) {
        uint32_t a_reg0 = pack_e2m3_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e2m3_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e2m3_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e2m3_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e2m3_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e2m3_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
              "+f"(c_tile.x[3])
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1));
      }

      if constexpr (std::is_same_v<T, cutlass::float_e3m2_t>) {
        uint32_t a_reg0 = pack_e3m2_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e3m2_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e3m2_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e3m2_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e3m2_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e3m2_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
              "+f"(c_tile.x[3])
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);

        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP4M2M1
    case 18: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        uint32_t a_reg0 = pack4_fp4_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack4_fp4_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack4_fp4_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack4_fp4_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack4_fp4_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack4_fp4_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
              "+f"(c_tile.x[3])
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
  }
}


// 3. 核函数实现（模板函数需在头文件中实现，否则链接报错）
template <typename T>
__global__ void mma2fp16(const T *A, const T *B, cutlass::half_t *C, int M, int N, int K, int dtype) {

  switch(dtype){
    //FP16
    case 1:{
      //Each thread has a copy of this tile
      Afrag_16x16<T> a_tile;
      Bfrag_16x8 <T> b_tile;
      CFrag_16x8 <cutlass::half_t> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }

      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;
      
      uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
      uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);

      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,  %1},"
                    "{%2,  %3,  %4,  %5},"
                    "{%6,  %7},"
                    "{%0,  %1};\n"
                  : "+r"(c_reg0), "+r"(c_reg1)
                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                    "r"(b_regs[0]), "r"(b_regs[1]));

      unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
      unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);
      
      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }

      break;
    }
    //FP8E4M3 or FP8E5M2
    case 13: 
    case 14: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8 <cutlass::half_t> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }


      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;
      
      uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
      uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);

      if(dtype == 13){
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                      "{%0,  %1},"
                      "{%2,  %3,  %4,  %5},"
                      "{%6,  %7},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else if(dtype == 14){
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
                      "{%0,  %1},"
                      "{%2,  %3,  %4,  %5},"
                      "{%6,  %7},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
      } else {
        printf("dtype is error, dtype is %d",dtype);
      }

      unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
      unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP6E2M3 or FP6E3M2
    case 15:
    case 16: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8 <cutlass::half_t> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }

      if constexpr (std::is_same_v<T, cutlass::float_e2m3_t>) {
        uint32_t a_reg0 = pack_e2m3_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e2m3_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e2m3_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e2m3_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e2m3_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e2m3_to_u32(&b_tile.x[4]);

        uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
        uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);

        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e2m3.f16 "
              "{%0,  %1},"
              "{%2,  %3,  %4,  %5},"
              "{%6,  %7},"
              "{%0,  %1};\n"
            : "+r"(c_reg0), "+r"(c_reg1)
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1));

        unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
        unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);
      }

      if constexpr (std::is_same_v<T, cutlass::float_e3m2_t>) {
        uint32_t a_reg0 = pack_e3m2_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e3m2_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e3m2_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e3m2_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e3m2_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e3m2_to_u32(&b_tile.x[4]);

        uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
        uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);

        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e3m2.f16 "
                      "{%0,  %1},"
                      "{%2,  %3,  %4,  %5},"
                      "{%6,  %7},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1));

        unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
        unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP4M2M1
    case 18: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8 <cutlass::half_t> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }

      if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        uint32_t a_reg0 = pack4_fp4_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack4_fp4_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack4_fp4_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack4_fp4_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack4_fp4_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack4_fp4_to_u32(&b_tile.x[4]);

        uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
        uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e2m1.f16 "
                      "{%0,  %1},"
                      "{%2,  %3,  %4,  %5},"
                      "{%6,  %7},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1));

        unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
        unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);
      }
      
      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
  }
}

// 3. 核函数实现（模板函数需在头文件中实现，否则链接报错）
template <typename T>
__global__ void mma2fp32_scale(const T *A, int8_t* a_scale, const T *B, int8_t* b_scale, float *C, int M, int N, int K, int dtype) {
  switch(dtype){
    //MXFP8E4M3 or MXFP8E5M2
    case 7: 
    case 8: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      // mma instruction expects 32-bit int registers, we cast
      const int *a_regs = (const int *)a_tile.x;
      const int *b_regs = (const int *)b_tile.x;

      //在这里我们暂时看传入的第一个scale参数
      uint8_t a_scale_u8 = (uint8_t)a_scale[0];
  
      uint8_t b_scale_u8 = (uint8_t)b_scale[0];

      uint32_t a_scale_int = (uint32_t)a_scale_u8;
      uint32_t b_scale_int = (uint32_t)b_scale_u8;

      static constexpr uint16_t tidA = 0;
      static constexpr uint16_t bidA = 0;
      static constexpr uint16_t tidB = 0;
      static constexpr uint16_t bidB = 0;

      if(dtype == 7){// mxfp8e4m3
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));

      } else if(dtype == 8){//mxfp8e5m2
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e5m2.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));

      } else {
        printf("dtype is error, dtype is %d",dtype);
      }
      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP6E2M3 or FP6E3M2
    case 11:
    case 12: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      //在这里我们暂时看传入的第一个scale参数
      uint8_t a_scale_u8 = (uint8_t)a_scale[0];
  
      uint8_t b_scale_u8 = (uint8_t)b_scale[0];

      uint32_t a_scale_int = (uint32_t)a_scale_u8;
      uint32_t b_scale_int = (uint32_t)b_scale_u8;

      static constexpr uint16_t tidA = 0;
      static constexpr uint16_t bidA = 0;
      static constexpr uint16_t tidB = 0;
      static constexpr uint16_t bidB = 0;

      if constexpr (std::is_same_v<T, cutlass::float_e2m3_t>) {
        uint32_t a_reg0 = pack_e2m3_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e2m3_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e2m3_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e2m3_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e2m3_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e2m3_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e2m3.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));
      }

      if constexpr (std::is_same_v<T, cutlass::float_e3m2_t>) {
        uint32_t a_reg0 = pack_e3m2_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack_e3m2_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack_e3m2_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack_e3m2_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack_e3m2_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack_e3m2_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);

        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP4M2M1(scale 1X)
    case 17: {
      //Each thread has a copy of this tile
      Afrag_16x32<T> a_tile;
      Bfrag_32x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      //在这里我们暂时看传入的第一个scale参数
      uint8_t a_scale_u8 = (uint8_t)a_scale[0];
  
      uint8_t b_scale_u8 = (uint8_t)b_scale[0];

      uint32_t a_scale_int = (uint32_t)a_scale_u8;
      uint32_t b_scale_int = (uint32_t)b_scale_u8;

      static constexpr uint16_t tidA = 0;
      static constexpr uint16_t bidA = 0;
      static constexpr uint16_t tidB = 0;
      static constexpr uint16_t bidB = 0;

      if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        uint32_t a_reg0 = pack4_fp4_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack4_fp4_to_u32(&a_tile.x[4]);
        uint32_t a_reg2 = pack4_fp4_to_u32(&a_tile.x[8]);
        uint32_t a_reg3 = pack4_fp4_to_u32(&a_tile.x[12]);

        uint32_t b_reg0 = pack4_fp4_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack4_fp4_to_u32(&b_tile.x[4]);

        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP4M2M1(scale 4X)
    case 9: {
      //Each thread has a copy of this tile
      Afrag_16x64<T> a_tile;
      Bfrag_64x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      //在这里我们暂时看传入的前4个参数
      uint32_t a_scale_int =
          ((uint32_t)(uint8_t)a_scale[0])       |
          ((uint32_t)(uint8_t)a_scale[1] << 8)  |
          ((uint32_t)(uint8_t)a_scale[2] << 16) |
          ((uint32_t)(uint8_t)a_scale[3] << 24);

      uint32_t b_scale_int =
          ((uint32_t)(uint8_t)b_scale[0])       |
          ((uint32_t)(uint8_t)b_scale[1] << 8)  |
          ((uint32_t)(uint8_t)b_scale[2] << 16) |
          ((uint32_t)(uint8_t)b_scale[3] << 24);

      static constexpr uint16_t tidA = 0;
      static constexpr uint16_t bidA = 0;
      static constexpr uint16_t tidB = 0;
      static constexpr uint16_t bidB = 0;

      if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        uint32_t a_reg0 = pack8_fp4_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack8_fp4_to_u32(&a_tile.x[8]);
        uint32_t a_reg2 = pack8_fp4_to_u32(&a_tile.x[16]);
        uint32_t a_reg3 = pack8_fp4_to_u32(&a_tile.x[24]);

        uint32_t b_reg0 = pack8_fp4_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack8_fp4_to_u32(&b_tile.x[8]);

        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
    //FP4M2M1(scale 2X)
    case 10: {
      //Each thread has a copy of this tile
      Afrag_16x64<T> a_tile;
      Bfrag_64x8 <T> b_tile;
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      //Load A & B fragments
      for (int idx = 0; idx < a_tile.ne; ++idx) {
        a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < b_tile.ne; ++idx) {
        b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      }
      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      //在这里我们暂时看传入的前2个参数
      uint32_t a_scale_int =
          ((uint32_t)(uint8_t)a_scale[0])       |
          ((uint32_t)(uint8_t)a_scale[1] << 8)  |
          ((uint32_t)(uint8_t)a_scale[0] << 16) |
          ((uint32_t)(uint8_t)a_scale[1] << 24);

      uint32_t b_scale_int =
          ((uint32_t)(uint8_t)b_scale[0])       |
          ((uint32_t)(uint8_t)b_scale[1] << 8)  |
          ((uint32_t)(uint8_t)b_scale[0] << 16) |
          ((uint32_t)(uint8_t)b_scale[1] << 24);

      static constexpr uint16_t tidA = 0;
      static constexpr uint16_t bidA = 0;
      static constexpr uint16_t tidB = 0;
      static constexpr uint16_t bidB = 0;

      if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        uint32_t a_reg0 = pack8_fp4_to_u32(&a_tile.x[0]);
        uint32_t a_reg1 = pack8_fp4_to_u32(&a_tile.x[8]);
        uint32_t a_reg2 = pack8_fp4_to_u32(&a_tile.x[16]);
        uint32_t a_reg3 = pack8_fp4_to_u32(&a_tile.x[24]);

        uint32_t b_reg0 = pack8_fp4_to_u32(&b_tile.x[0]);
        uint32_t b_reg1 = pack8_fp4_to_u32(&b_tile.x[8]);

        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3},"
                    "{%10},"
                    "{%11, %12},"
                    "{%13},"
                    "{%14, %15};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
                      "r"(b_reg0), "r"(b_reg1),
                      "r"(a_scale_int), "h"(bidA), "h"(tidA),
                      "r"(b_scale_int), "h"(bidB), "h"(tidB));
      }

      //3. Write back to C
      for (int i = 0; i < c_tile.ne; ++i) {
        int row = c_tile.get_row(tid, i);
        int col = c_tile.get_col(tid, i);
        C[row * N + col] = c_tile.x[i];
      }
      break;
    }
  }
}


template <typename T>
void runMma2fp32(int M, int N, int K,
                          T* h_A, T* h_B, float* h_C,
                          T* d_A, T* d_B, float* d_C,
                          int dtype, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C),cudaMemcpyHostToDevice);

    // start the function
    mma2fp32<T><<<1,32>>> (d_A, d_B, d_C, M, N, K, dtype);

    //  Device -> Host
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);

    // print the result of C
    if(print_en){
      for(int i=0;i<M;i++){
          for(int j=0;j<N;j++){
              printf("%0.1f ", h_C[i*N+j]);
          }
          printf("\n");
      }
      printf("\n");
    }
}

template <typename T>
void runMma2fp32_scale(int M, int N, int K,
                          T* h_A, T* h_B, float* h_C,
                          T* d_A, T* d_B, float* d_C,
                          int8_t* a_scale, int8_t* b_scale,
                          int dtype, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C),cudaMemcpyHostToDevice);

    // start the function
    mma2fp32_scale<T><<<1,32>>> (d_A, a_scale, d_B, b_scale, d_C, M, N, K, dtype);
    //  Device -> Host
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);

    // print the result of C
    if(print_en){
      for(int i=0;i<M;i++){
          for(int j=0;j<N;j++){
              printf("%0.1f ", h_C[i*N+j]);
          }
          printf("\n");
      }
      printf("\n");
    }
}

template <typename T>
void runMma2fp16(int M, int N, int K,
                          T* h_A, T* h_B, cutlass::half_t* h_C,
                          T* d_A, T* d_B, cutlass::half_t* d_C,
                          int dtype, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C),cudaMemcpyHostToDevice);

    // start the function
    mma2fp16<T><<<1,32>>> (d_A, d_B, d_C, M, N, K, dtype);

    //  Device -> Host
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);

    // print the result of C
    if(print_en){
      for(int i=0;i<M;i++){
          for(int j=0;j<N;j++){
              printf("%0.1f ", float(h_C[i*N+j]));
          }
          printf("\n");
      }
      printf("\n");
    }
}

#endif // MMA_TF32_KERNEL_H_