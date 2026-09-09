#ifndef MMA_TF32_KERNEL_H_
#define MMA_TF32_KERNEL_H_

// 必要的CUDA和Cutlass头文件（封装时需包含依赖）
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "mma_descriptor.h"
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

template <typename T>
struct Afrag_16x8_f16 {
  static constexpr size_t ne = 4;
  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + (l % 2);
  }
};

template <typename T>
struct Afrag_16x4 {
  static constexpr size_t ne = 2;
  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * l;
  }

  static __device__ size_t get_col(int tid, int) {
    return tid % 4;
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

template <typename T>
struct Bfrag_8x8_f16 {
  static constexpr size_t ne = 2;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    return 2 * (tid % 4) + l;
  }

  static __device__ size_t get_col(int tid, int) {
    return tid >> 2;
  }
};

template <typename T>
struct Bfrag_4x8 {
  static constexpr size_t ne = 1;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int) {
    return tid % 4;
  }

  static __device__ size_t get_col(int tid, int) {
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

template <typename T>
struct Afrag_8x16 {
  static constexpr size_t ne = 4;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int) {
    return tid >> 2;
  }

  static __device__ size_t get_col(int tid, int l) {
    return 4 * (tid % 4) + l;
  }
};

template <typename T>
struct Afrag_8x32 {
  static constexpr size_t ne = 8;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int) {
    return tid >> 2;
  }

  static __device__ size_t get_col(int tid, int l) {
    return 8 * (tid % 4) + l;
  }
};

template <typename T>
struct Bfrag_16x8_m8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    return 4 * (tid % 4) + l;
  }

  static __device__ size_t get_col(int tid, int) {
    return tid >> 2;
  }
};

template <typename T>
struct Bfrag_32x8_m8 {
  static constexpr size_t ne = 8;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int l) {
    return 8 * (tid % 4) + l;
  }

  static __device__ size_t get_col(int tid, int) {
    return tid >> 2;
  }
};

template <typename T>
struct CFrag_8x8 {
  static constexpr size_t ne = 2;
  T x[ne] = {};

  static __device__ size_t get_row(int tid, int) {
    return tid >> 2;
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + l;
  }
};

template <typename T>
__device__ __forceinline__ uint32_t pack_masked_bytes_to_u32(const T* src, int count, uint8_t mask, int value_shift = 0)
{
    uint32_t v = 0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        uint32_t raw = (uint32_t)(p[i] & mask);
        raw <<= value_shift;
        v |= raw << (i * 8);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ uint32_t pack_nibbles_to_u32(const T* src, int count)
{
    uint32_t v = 0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        v |= (uint32_t)(p[i] & 0x0f) << (i * 4);
    }
    return v;
}

template <MmaOperandKind Kind, typename T>
__device__ __forceinline__ void pack_f8f6f4_a_regs(const T* src, uint32_t (&regs)[4])
{
    if constexpr (Kind == MmaOperandKind::E4M3 || Kind == MmaOperandKind::E5M2) {
        const uint32_t* raw = reinterpret_cast<const uint32_t*>(src);
        regs[0] = raw[0];
        regs[1] = raw[1];
        regs[2] = raw[2];
        regs[3] = raw[3];
    } else if constexpr (Kind == MmaOperandKind::E2M3 || Kind == MmaOperandKind::E3M2) {
        regs[0] = pack_masked_bytes_to_u32(src + 0, 4, 0x3f);
        regs[1] = pack_masked_bytes_to_u32(src + 4, 4, 0x3f);
        regs[2] = pack_masked_bytes_to_u32(src + 8, 4, 0x3f);
        regs[3] = pack_masked_bytes_to_u32(src + 12, 4, 0x3f);
    } else { // e2m1
        regs[0] = pack_masked_bytes_to_u32(src + 0, 4, 0x0f, 2);
        regs[1] = pack_masked_bytes_to_u32(src + 4, 4, 0x0f, 2);
        regs[2] = pack_masked_bytes_to_u32(src + 8, 4, 0x0f, 2);
        regs[3] = pack_masked_bytes_to_u32(src + 12, 4, 0x0f, 2);
    }
}

template <MmaOperandKind Kind, typename T>
__device__ __forceinline__ void pack_f8f6f4_b_regs(const T* src, uint32_t (&regs)[2])
{
    if constexpr (Kind == MmaOperandKind::E4M3 || Kind == MmaOperandKind::E5M2) {
        const uint32_t* raw = reinterpret_cast<const uint32_t*>(src);
        regs[0] = raw[0];
        regs[1] = raw[1];
    } else if constexpr (Kind == MmaOperandKind::E2M3 || Kind == MmaOperandKind::E3M2) {
        regs[0] = pack_masked_bytes_to_u32(src + 0, 4, 0x3f);
        regs[1] = pack_masked_bytes_to_u32(src + 4, 4, 0x3f);
    } else { // e2m1
        regs[0] = pack_masked_bytes_to_u32(src + 0, 4, 0x0f, 2);
        regs[1] = pack_masked_bytes_to_u32(src + 4, 4, 0x0f, 2);
    }
}

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

// 将4个scale字节打包到一个32位寄存器，供mxf4 block_scale指令使用。
__host__ __device__ __forceinline__ uint32_t pack_scale_u8x4(
    uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3) {
  return (uint32_t)s0 |
         ((uint32_t)s1 << 8) |
         ((uint32_t)s2 << 16) |
         ((uint32_t)s3 << 24);
}

// 默认256b路径：4个block各自独立scale（dtype9/10通用）。
__host__ __device__ __forceinline__ uint32_t pack_mxf4_scale_default(const int8_t* scale) {
  return pack_scale_u8x4((uint8_t)scale[0], (uint8_t)scale[1], (uint8_t)scale[2], (uint8_t)scale[3]);
}

// MXFP4 external ABI: one scale covers 32 FP4 elements. The hardware's four
// 16-element scale slots therefore receive [s0, s0, s1, s1].
__host__ __device__ __forceinline__ uint32_t pack_mxfp4_scale_per_32_elements(const int8_t* scale) {
  return pack_scale_u8x4((uint8_t)scale[0], (uint8_t)scale[0],
                         (uint8_t)scale[1], (uint8_t)scale[1]);
}

// 128b拓扑路径：仅前两个有效block参与，后两个scale固定为0。
// blocksize16=1: 前两个block用各自scale；blocksize16=0: 前两个共享scale[0]。
__host__ __device__ __forceinline__ uint32_t pack_mxf4_scale_topology_128(
    const int8_t* scale, int blocksize16) {
  const uint8_t s0 = (uint8_t)scale[0];
  const uint8_t s1 = (uint8_t)(blocksize16 ? scale[1] : scale[0]);
  return pack_scale_u8x4(s0, s1, 0, 0);
}

__host__ __device__ __forceinline__ uint32_t pack_mxfp4_scale_topology_128_per_32_elements(
    const int8_t* scale) {
  return pack_scale_u8x4((uint8_t)scale[0], (uint8_t)scale[0], 0, 0);
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
      const int tid = threadIdx.x;
      if (K == 8) {
        Afrag_16x8_f16<T> a_tile;
        Bfrag_8x8_f16<T> b_tile;
        CFrag_16x8<float> c_tile;

        for (int idx = 0; idx < a_tile.ne; ++idx) {
          a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < b_tile.ne; ++idx) {
          b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < c_tile.ne; ++idx) {
          c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
        }

        const int *a_regs = (const int *)a_tile.x;
        const int *b_regs = (const int *)b_tile.x;

        if(dtype == 1){
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                      "{%0, %1, %2, %3}, "
                      "{%4, %5}, "
                      "{%6}, "
                      "{%0, %1, %2, %3};\n"
                      : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                        "+f"(c_tile.x[3])
                      : "r"(a_regs[0]), "r"(a_regs[1]),
                        "r"(b_regs[0]));
        } else if(dtype == 2){
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                      "{%0, %1, %2, %3}, "
                      "{%4, %5}, "
                      "{%6}, "
                      "{%0, %1, %2, %3};\n"
                      : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                        "+f"(c_tile.x[3])
                      : "r"(a_regs[0]), "r"(a_regs[1]),
                        "r"(b_regs[0]));
        } else {
          printf("dtype is error, dtype is %d",dtype);
        }

        for (int i = 0; i < c_tile.ne; ++i) {
          int row = c_tile.get_row(tid, i);
          int col = c_tile.get_col(tid, i);
          C[row * N + col] = c_tile.x[i];
        }
      } else {
        //Each thread has a copy of this tile
        Afrag_16x16<T> a_tile;
        Bfrag_16x8<T> b_tile;
        CFrag_16x8<float> c_tile;

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
      }
      break;
    }
    //TF32
    case 3: {
      //Each thread has a copy of this tile
      CFrag_16x8<float> c_tile;

      const int tid = threadIdx.x;

      for (int idx = 0; idx < c_tile.ne; ++idx) {
          c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      if (K == 4) {
        Afrag_16x4<T> a_tile;
        Bfrag_4x8<T> b_tile;

        for (int idx = 0; idx < a_tile.ne; ++idx) {
          a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < b_tile.ne; ++idx) {
          b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
        }

        const int *a_regs = (const int *)a_tile.x;
        const int *b_regs = (const int *)b_tile.x;

        asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]),
                      "+f"(c_tile.x[3])
                    : "r"(a_regs[0]), "r"(a_regs[1]),
                      "r"(b_regs[0]));
      } else {
        Afrag_16x8<T> a_tile;
        Bfrag_8x8 <T> b_tile;

        //Load A & B fragments
        for (int idx = 0; idx < a_tile.ne; ++idx) {
          a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < b_tile.ne; ++idx) {
          b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
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
      CFrag_16x8 <cutlass::half_t> c_tile;

      const int tid = threadIdx.x;

      for (int idx = 0; idx < c_tile.ne; ++idx) {
        c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
      }

      uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
      uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);
      if (K == 8) {
        Afrag_16x8_f16<T> a_tile;
        Bfrag_8x8_f16<T> b_tile;

        for (int idx = 0; idx < a_tile.ne; ++idx) {
          a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < b_tile.ne; ++idx) {
          b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
        }
        const int *a_regs = (const int *)a_tile.x;
        const int *b_regs = (const int *)b_tile.x;

        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                      "{%0,  %1},"
                      "{%2,  %3},"
                      "{%4},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_regs[0]), "r"(a_regs[1]),
                      "r"(b_regs[0]));
      } else {
        //Each thread has a copy of this tile
        Afrag_16x16<T> a_tile;
        Bfrag_16x8 <T> b_tile;

        for (int idx = 0; idx < a_tile.ne; ++idx) {
          a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
        }
        for (int idx = 0; idx < b_tile.ne; ++idx) {
          b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
        }

        // mma instruction expects 32-bit int registers, we cast
        const int *a_regs = (const int *)a_tile.x;
        const int *b_regs = (const int *)b_tile.x;
        
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                      "{%0,  %1},"
                      "{%2,  %3,  %4,  %5},"
                      "{%6,  %7},"
                      "{%0,  %1};\n"
                    : "+r"(c_reg0), "+r"(c_reg1)
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(b_regs[0]), "r"(b_regs[1]));
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
    //FP8E4M3 or FP8E5M2
  }
}

// 3. 核函数实现（模板函数需在头文件中实现，否则链接报错）
template <typename T>
__global__ void mma2fp32_scale(const T *A, int8_t* a_scale, const T *B, int8_t* b_scale, float *C, int M, int N, int K, int dtype, int mxf4_topology_128 = 0, int blocksize16 = 1) {
  switch(dtype){
    //FP4E2M1(scale 4X)
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

      // 默认256b模式：4个block各自scale。
      uint32_t a_scale_int = pack_mxf4_scale_default(a_scale);
      uint32_t b_scale_int = pack_mxf4_scale_default(b_scale);

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
    //FP4E2M1(scale 4X for MXFP4)
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

      // 统一scale打包策略，减少分支重复并固定128b拓扑语义。
      uint32_t a_scale_int = mxf4_topology_128
          ? pack_mxfp4_scale_topology_128_per_32_elements(a_scale)
          : pack_mxfp4_scale_per_32_elements(a_scale);
      uint32_t b_scale_int = mxf4_topology_128
          ? pack_mxfp4_scale_topology_128_per_32_elements(b_scale)
          : pack_mxfp4_scale_per_32_elements(b_scale);

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

        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
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


__global__ void mma2fp32_scale_mixed_f8f6f4(const uint8_t *A, int8_t* a_scale, const uint8_t *B, int8_t* b_scale, float *C, int M, int N, int K, int a_kind, int b_kind) {
  Afrag_16x32<uint8_t> a_tile;
  Bfrag_32x8<uint8_t> b_tile;
  CFrag_16x8<float> c_tile;
  const int tid = threadIdx.x;

  for (int idx = 0; idx < a_tile.ne; ++idx) {
    a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < b_tile.ne; ++idx) {
    b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < c_tile.ne; ++idx) {
    c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
  }

  uint8_t a_scale_u8 = (uint8_t)a_scale[0];
  uint8_t b_scale_u8 = (uint8_t)b_scale[0];
  uint32_t a_scale_int = (uint32_t)a_scale_u8;
  uint32_t b_scale_int = (uint32_t)b_scale_u8;
  static constexpr uint16_t tidA = 0;
  static constexpr uint16_t bidA = 0;
  static constexpr uint16_t tidB = 0;
  static constexpr uint16_t bidB = 0;

  uint32_t a_regs[4] = {0, 0, 0, 0};
  uint32_t b_regs[2] = {0, 0};

#define MMA_MXF8F6F4_SCALE_CASE(AKIND_ENUM, BKIND_ENUM, ATOK, BTOK) \
  if (a_kind == (int)AKIND_ENUM && b_kind == (int)BKIND_ENUM) { \
    pack_f8f6f4_a_regs<AKIND_ENUM>(a_tile.x, a_regs); \
    pack_f8f6f4_b_regs<BKIND_ENUM>(b_tile.x, b_regs); \
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32." ATOK "." BTOK ".f32.ue8m0 " \
                 "{%0, %1, %2, %3}, " \
                 "{%4, %5, %6, %7}, " \
                 "{%8, %9}, " \
                 "{%0, %1, %2, %3}," \
                 "{%10}," \
                 "{%11, %12}," \
                 "{%13}," \
                 "{%14, %15};\n" \
                 : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]), "+f"(c_tile.x[3]) \
                 : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), \
                   "r"(b_regs[0]), "r"(b_regs[1]), \
                   "r"(a_scale_int), "h"(bidA), "h"(tidA), \
                   "r"(b_scale_int), "h"(bidB), "h"(tidB)); \
  }

  MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E4M3, MmaOperandKind::E4M3, "e4m3", "e4m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E4M3, MmaOperandKind::E5M2, "e4m3", "e5m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M3, "e4m3", "e2m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E4M3, MmaOperandKind::E3M2, "e4m3", "e3m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M1, "e4m3", "e2m1")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E5M2, MmaOperandKind::E4M3, "e5m2", "e4m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E5M2, MmaOperandKind::E5M2, "e5m2", "e5m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M3, "e5m2", "e2m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E5M2, MmaOperandKind::E3M2, "e5m2", "e3m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M1, "e5m2", "e2m1")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M3, MmaOperandKind::E4M3, "e2m3", "e4m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M3, MmaOperandKind::E5M2, "e2m3", "e5m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M3, "e2m3", "e2m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M3, MmaOperandKind::E3M2, "e2m3", "e3m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M1, "e2m3", "e2m1")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E3M2, MmaOperandKind::E4M3, "e3m2", "e4m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E3M2, MmaOperandKind::E5M2, "e3m2", "e5m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M3, "e3m2", "e2m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E3M2, MmaOperandKind::E3M2, "e3m2", "e3m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M1, "e3m2", "e2m1")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M1, MmaOperandKind::E4M3, "e2m1", "e4m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M1, MmaOperandKind::E5M2, "e2m1", "e5m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M3, "e2m1", "e2m3")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M1, MmaOperandKind::E3M2, "e2m1", "e3m2")
  else MMA_MXF8F6F4_SCALE_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M1, "e2m1", "e2m1")

#undef MMA_MXF8F6F4_SCALE_CASE

  for (int i = 0; i < c_tile.ne; ++i) {
    int row = c_tile.get_row(tid, i);
    int col = c_tile.get_col(tid, i);
    C[row * N + col] = c_tile.x[i];
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

__global__ void mma2fp32_mixed_f8f6f4(const uint8_t *A, const uint8_t *B, float *C, int M, int N, int K, int a_kind, int b_kind) {
  Afrag_16x32<uint8_t> a_tile;
  Bfrag_32x8<uint8_t> b_tile;
  CFrag_16x8<float> c_tile;
  const int tid = threadIdx.x;

  for (int idx = 0; idx < a_tile.ne; ++idx) {
    a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < b_tile.ne; ++idx) {
    b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < c_tile.ne; ++idx) {
    c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
  }

  uint32_t a_regs[4] = {0, 0, 0, 0};
  uint32_t b_regs[2] = {0, 0};

#define MMA_F8F6F4_CASE(AKIND_ENUM, BKIND_ENUM, ATOK, BTOK) \
  if (a_kind == (int)AKIND_ENUM && b_kind == (int)BKIND_ENUM) { \
    pack_f8f6f4_a_regs<AKIND_ENUM>(a_tile.x, a_regs); \
    pack_f8f6f4_b_regs<BKIND_ENUM>(b_tile.x, b_regs); \
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32." ATOK "." BTOK ".f32 " \
                 "{%0, %1, %2, %3}, " \
                 "{%4, %5, %6, %7}, " \
                 "{%8, %9}, " \
                 "{%0, %1, %2, %3};\n" \
                 : "+f"(c_tile.x[0]), "+f"(c_tile.x[1]), "+f"(c_tile.x[2]), "+f"(c_tile.x[3]) \
                 : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), \
                   "r"(b_regs[0]), "r"(b_regs[1])); \
  }

  MMA_F8F6F4_CASE(MmaOperandKind::E4M3, MmaOperandKind::E4M3, "e4m3", "e4m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E4M3, MmaOperandKind::E5M2, "e4m3", "e5m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M3, "e4m3", "e2m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E4M3, MmaOperandKind::E3M2, "e4m3", "e3m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M1, "e4m3", "e2m1")
  else MMA_F8F6F4_CASE(MmaOperandKind::E5M2, MmaOperandKind::E4M3, "e5m2", "e4m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E5M2, MmaOperandKind::E5M2, "e5m2", "e5m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M3, "e5m2", "e2m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E5M2, MmaOperandKind::E3M2, "e5m2", "e3m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M1, "e5m2", "e2m1")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M3, MmaOperandKind::E4M3, "e2m3", "e4m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M3, MmaOperandKind::E5M2, "e2m3", "e5m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M3, "e2m3", "e2m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M3, MmaOperandKind::E3M2, "e2m3", "e3m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M1, "e2m3", "e2m1")
  else MMA_F8F6F4_CASE(MmaOperandKind::E3M2, MmaOperandKind::E4M3, "e3m2", "e4m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E3M2, MmaOperandKind::E5M2, "e3m2", "e5m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M3, "e3m2", "e2m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E3M2, MmaOperandKind::E3M2, "e3m2", "e3m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M1, "e3m2", "e2m1")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M1, MmaOperandKind::E4M3, "e2m1", "e4m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M1, MmaOperandKind::E5M2, "e2m1", "e5m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M3, "e2m1", "e2m3")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M1, MmaOperandKind::E3M2, "e2m1", "e3m2")
  else MMA_F8F6F4_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M1, "e2m1", "e2m1")

#undef MMA_F8F6F4_CASE

  for (int i = 0; i < c_tile.ne; ++i) {
    int row = c_tile.get_row(tid, i);
    int col = c_tile.get_col(tid, i);
    C[row * N + col] = c_tile.x[i];
  }
}

__global__ void mma2fp16_mixed_f8f6f4(const uint8_t *A, const uint8_t *B, cutlass::half_t *C, int M, int N, int K, int a_kind, int b_kind) {
  Afrag_16x32<uint8_t> a_tile;
  Bfrag_32x8<uint8_t> b_tile;
  CFrag_16x8<cutlass::half_t> c_tile;
  const int tid = threadIdx.x;

  for (int idx = 0; idx < a_tile.ne; ++idx) {
    a_tile.x[idx] = A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < b_tile.ne; ++idx) {
    b_tile.x[idx] = B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
  }
  for (int idx = 0; idx < c_tile.ne; ++idx) {
    c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
  }

  uint32_t a_regs[4] = {0, 0, 0, 0};
  uint32_t b_regs[2] = {0, 0};
  uint32_t c_reg0 = pack4_fp16_to_u32(&c_tile.x[0]);
  uint32_t c_reg1 = pack4_fp16_to_u32(&c_tile.x[2]);

#define MMA_F8F6F4_FP16_CASE(AKIND_ENUM, BKIND_ENUM, ATOK, BTOK) \
  if (a_kind == (int)AKIND_ENUM && b_kind == (int)BKIND_ENUM) { \
    pack_f8f6f4_a_regs<AKIND_ENUM>(a_tile.x, a_regs); \
    pack_f8f6f4_b_regs<BKIND_ENUM>(b_tile.x, b_regs); \
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16." ATOK "." BTOK ".f16 " \
                 "{%0, %1}," \
                 "{%2, %3, %4, %5}," \
                 "{%6, %7}," \
                 "{%0, %1};\n" \
                 : "+r"(c_reg0), "+r"(c_reg1) \
                 : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), \
                   "r"(b_regs[0]), "r"(b_regs[1])); \
  }

  MMA_F8F6F4_FP16_CASE(MmaOperandKind::E4M3, MmaOperandKind::E4M3, "e4m3", "e4m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E4M3, MmaOperandKind::E5M2, "e4m3", "e5m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M3, "e4m3", "e2m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E4M3, MmaOperandKind::E3M2, "e4m3", "e3m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E4M3, MmaOperandKind::E2M1, "e4m3", "e2m1")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E5M2, MmaOperandKind::E4M3, "e5m2", "e4m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E5M2, MmaOperandKind::E5M2, "e5m2", "e5m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M3, "e5m2", "e2m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E5M2, MmaOperandKind::E3M2, "e5m2", "e3m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E5M2, MmaOperandKind::E2M1, "e5m2", "e2m1")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M3, MmaOperandKind::E4M3, "e2m3", "e4m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M3, MmaOperandKind::E5M2, "e2m3", "e5m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M3, "e2m3", "e2m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M3, MmaOperandKind::E3M2, "e2m3", "e3m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M3, MmaOperandKind::E2M1, "e2m3", "e2m1")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E3M2, MmaOperandKind::E4M3, "e3m2", "e4m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E3M2, MmaOperandKind::E5M2, "e3m2", "e5m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M3, "e3m2", "e2m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E3M2, MmaOperandKind::E3M2, "e3m2", "e3m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E3M2, MmaOperandKind::E2M1, "e3m2", "e2m1")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M1, MmaOperandKind::E4M3, "e2m1", "e4m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M1, MmaOperandKind::E5M2, "e2m1", "e5m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M3, "e2m1", "e2m3")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M1, MmaOperandKind::E3M2, "e2m1", "e3m2")
  else MMA_F8F6F4_FP16_CASE(MmaOperandKind::E2M1, MmaOperandKind::E2M1, "e2m1", "e2m1")

#undef MMA_F8F6F4_FP16_CASE

  unpack4_u32_to_fp16(c_reg0, &c_tile.x[0]);
  unpack4_u32_to_fp16(c_reg1, &c_tile.x[2]);

  for (int i = 0; i < c_tile.ne; ++i) {
    int row = c_tile.get_row(tid, i);
    int col = c_tile.get_col(tid, i);
    C[row * N + col] = c_tile.x[i];
  }
}

__global__ void mma2s32_integer(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K, int shape, int a_kind, int b_kind, int satfinite) {
  const int tid = threadIdx.x;

#define MMA_INT_DO(PTX) asm volatile(PTX);

  if (shape == (int)MmaShape::M8N8K16) {
    Afrag_8x16<uint8_t> a_tile;
    Bfrag_16x8_m8<uint8_t> b_tile;
    CFrag_8x8<int32_t> c_tile;
    for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
    for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
    for (int idx = 0; idx < c_tile.ne; ++idx) c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
    uint32_t a_reg = *reinterpret_cast<uint32_t*>(a_tile.x);
    uint32_t b_reg = *reinterpret_cast<uint32_t*>(b_tile.x);
    if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::S8) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::U8) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else if (a_kind == (int)MmaOperandKind::U8 && b_kind == (int)MmaOperandKind::S8) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    }
    for (int i = 0; i < c_tile.ne; ++i) C[c_tile.get_row(tid, i) * N + c_tile.get_col(tid, i)] = c_tile.x[i];
    return;
  }

  if (shape == (int)MmaShape::M8N8K32) {
    Afrag_8x32<uint8_t> a_tile;
    Bfrag_32x8_m8<uint8_t> b_tile;
    CFrag_8x8<int32_t> c_tile;
    for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
    for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
    for (int idx = 0; idx < c_tile.ne; ++idx) c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
    uint32_t a_reg = pack_nibbles_to_u32(a_tile.x, 8);
    uint32_t b_reg = pack_nibbles_to_u32(b_tile.x, 8);
    if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::S4) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::U4) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else if (a_kind == (int)MmaOperandKind::U4 && b_kind == (int)MmaOperandKind::S4) {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    } else {
      if (satfinite) asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
      else asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]) : "r"(a_reg), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]));
    }
    for (int i = 0; i < c_tile.ne; ++i) C[c_tile.get_row(tid, i) * N + c_tile.get_col(tid, i)] = c_tile.x[i];
    return;
  }

  if (shape == (int)MmaShape::M16N8K16) {
    Afrag_16x16<uint8_t> a_tile;
    Bfrag_16x8<uint8_t> b_tile;
    CFrag_16x8<int32_t> c_tile;
    for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
    for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
    for (int idx = 0; idx < c_tile.ne; ++idx) c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
    const uint32_t* a_regs = reinterpret_cast<const uint32_t*>(a_tile.x);
    uint32_t b_reg = *reinterpret_cast<uint32_t*>(b_tile.x);
    if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::S8) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
      else asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
    } else if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::U8) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
      else asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
    } else if (a_kind == (int)MmaOperandKind::U8 && b_kind == (int)MmaOperandKind::S8) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
      else asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
    } else {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
      else asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg));
    }
    for (int i = 0; i < c_tile.ne; ++i) C[c_tile.get_row(tid, i) * N + c_tile.get_col(tid, i)] = c_tile.x[i];
    return;
  }

  if (shape == (int)MmaShape::M16N8K32) {
    CFrag_16x8<int32_t> c_tile;
    for (int idx = 0; idx < c_tile.ne; ++idx) c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];

    if (a_kind == (int)MmaOperandKind::S8 || a_kind == (int)MmaOperandKind::U8) {
      Afrag_16x32<uint8_t> a_tile;
      Bfrag_32x8<uint8_t> b_tile;
      for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      const uint32_t* a_regs = reinterpret_cast<const uint32_t*>(a_tile.x);
      const uint32_t* b_regs = reinterpret_cast<const uint32_t*>(b_tile.x);
      if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::S8) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                      "r"(b_regs[0]), "r"(b_regs[1]),
                                      "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                            "r"(b_regs[0]), "r"(b_regs[1]),
                            "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else if (a_kind == (int)MmaOperandKind::S8 && b_kind == (int)MmaOperandKind::U8) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                      "r"(b_regs[0]), "r"(b_regs[1]),
                                      "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                            "r"(b_regs[0]), "r"(b_regs[1]),
                            "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else if (a_kind == (int)MmaOperandKind::U8 && b_kind == (int)MmaOperandKind::S8) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                      "r"(b_regs[0]), "r"(b_regs[1]),
                                      "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                            "r"(b_regs[0]), "r"(b_regs[1]),
                            "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                      "r"(b_regs[0]), "r"(b_regs[1]),
                                      "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                            "r"(b_regs[0]), "r"(b_regs[1]),
                            "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      }
    } else {
      Afrag_16x32<uint8_t> a_tile;
      Bfrag_32x8<uint8_t> b_tile;
      for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
      for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
      uint32_t a_regs[2] = {pack_nibbles_to_u32(a_tile.x + 0, 8), pack_nibbles_to_u32(a_tile.x + 8, 8)};
      uint32_t b_reg = pack_nibbles_to_u32(b_tile.x, 8);
      if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::S4) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::U4) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else if (a_kind == (int)MmaOperandKind::U4 && b_kind == (int)MmaOperandKind::S4) {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      } else {
        if (satfinite) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32.satfinite {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                    : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
        else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                          : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_reg), "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      }
    }
    for (int i = 0; i < c_tile.ne; ++i) C[c_tile.get_row(tid, i) * N + c_tile.get_col(tid, i)] = c_tile.x[i];
    return;
  }

  if (shape == (int)MmaShape::M16N8K64) {
    Afrag_16x64<uint8_t> a_tile;
    Bfrag_64x8<uint8_t> b_tile;
    CFrag_16x8<int32_t> c_tile;
    for (int idx = 0; idx < a_tile.ne; ++idx) a_tile.x[idx] = (uint8_t)A[a_tile.get_row(tid, idx) * K + a_tile.get_col(tid, idx)];
    for (int idx = 0; idx < b_tile.ne; ++idx) b_tile.x[idx] = (uint8_t)B[b_tile.get_row(tid, idx) * N + b_tile.get_col(tid, idx)];
    for (int idx = 0; idx < c_tile.ne; ++idx) c_tile.x[idx] = C[c_tile.get_row(tid, idx) * N + c_tile.get_col(tid, idx)];
    uint32_t a_regs[4] = {
      pack_nibbles_to_u32(a_tile.x + 0, 8),
      pack_nibbles_to_u32(a_tile.x + 8, 8),
      pack_nibbles_to_u32(a_tile.x + 16, 8),
      pack_nibbles_to_u32(a_tile.x + 24, 8),
    };
    uint32_t b_regs[2] = {
      pack_nibbles_to_u32(b_tile.x + 0, 8),
      pack_nibbles_to_u32(b_tile.x + 8, 8),
    };
    if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::S4) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                    "r"(b_regs[0]), "r"(b_regs[1]),
                                    "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      else asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                          "r"(b_regs[0]), "r"(b_regs[1]),
                          "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
    } else if (a_kind == (int)MmaOperandKind::S4 && b_kind == (int)MmaOperandKind::U4) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                    "r"(b_regs[0]), "r"(b_regs[1]),
                                    "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      else asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                          "r"(b_regs[0]), "r"(b_regs[1]),
                          "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
    } else if (a_kind == (int)MmaOperandKind::U4 && b_kind == (int)MmaOperandKind::S4) {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                    "r"(b_regs[0]), "r"(b_regs[1]),
                                    "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      else asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                          "r"(b_regs[0]), "r"(b_regs[1]),
                          "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
    } else {
      if (satfinite) asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                  : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                                  : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                                    "r"(b_regs[0]), "r"(b_regs[1]),
                                    "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
      else asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "+r"(c_tile.x[0]), "+r"(c_tile.x[1]), "+r"(c_tile.x[2]), "+r"(c_tile.x[3])
                        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                          "r"(b_regs[0]), "r"(b_regs[1]),
                          "r"(c_tile.x[0]), "r"(c_tile.x[1]), "r"(c_tile.x[2]), "r"(c_tile.x[3]));
    }
    for (int i = 0; i < c_tile.ne; ++i) C[c_tile.get_row(tid, i) * N + c_tile.get_col(tid, i)] = c_tile.x[i];
  }
}

inline void runMma2fp32_mixed_f8f6f4(int M, int N, int K,
                                     uint8_t* h_A, uint8_t* h_B, float* h_C,
                                     uint8_t* d_A, uint8_t* d_B, float* d_C,
                                     int a_kind, int b_kind, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C), cudaMemcpyHostToDevice);
    mma2fp32_mixed_f8f6f4<<<1,32>>>(d_A, d_B, d_C, M, N, K, a_kind, b_kind);
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);
    if (print_en) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) printf("%0.1f ", h_C[i*N+j]);
        printf("\n");
      }
      printf("\n");
    }
}

inline void runMma2fp16_mixed_f8f6f4(int M, int N, int K,
                                     uint8_t* h_A, uint8_t* h_B, cutlass::half_t* h_C,
                                     uint8_t* d_A, uint8_t* d_B, cutlass::half_t* d_C,
                                     int a_kind, int b_kind, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C), cudaMemcpyHostToDevice);
    mma2fp16_mixed_f8f6f4<<<1,32>>>(d_A, d_B, d_C, M, N, K, a_kind, b_kind);
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);
    if (print_en) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) printf("%0.1f ", float(h_C[i*N+j]));
        printf("\n");
      }
      printf("\n");
    }
}

inline void runMma2s32_integer(int M, int N, int K,
                               int8_t* h_A, int8_t* h_B, int32_t* h_C,
                               int8_t* d_A, int8_t* d_B, int32_t* d_C,
                               int shape, int a_kind, int b_kind, bool satfinite, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C), cudaMemcpyHostToDevice);
    mma2s32_integer<<<1,32>>>(d_A, d_B, d_C, M, N, K, shape, a_kind, b_kind, satfinite ? 1 : 0);
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);
    if (print_en) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) printf("%d ", h_C[i*N+j]);
        printf("\n");
      }
      printf("\n");
    }
}

inline void runMma2fp32_scale_mixed_f8f6f4(int M, int N, int K,
                                           uint8_t* h_A, uint8_t* h_B, float* h_C,
                                           uint8_t* d_A, uint8_t* d_B, float* d_C,
                                           int8_t* a_scale, int8_t* b_scale,
                                           int a_kind, int b_kind, bool print_en = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C), cudaMemcpyHostToDevice);
    mma2fp32_scale_mixed_f8f6f4<<<1,32>>>(d_A, a_scale, d_B, b_scale, d_C, M, N, K, a_kind, b_kind);
    cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost);

    if (print_en) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) printf("%0.1f ", h_C[i*N+j]);
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
                          int dtype, bool print_en = 1,
                          int mxf4_topology_128 = 0,
                          int blocksize16 = 1)
{
    cudaMemcpy(d_A, h_A, M*K*sizeof(*h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(*h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(*h_C),cudaMemcpyHostToDevice);

    // start the function
    mma2fp32_scale<T><<<1,32>>> (d_A, a_scale, d_B, b_scale, d_C, M, N, K, dtype, mxf4_topology_128, blocksize16);
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
