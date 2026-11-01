// The 16 FP32-output MMA instructions under test, their fragment layouts, and the probe kernels.
#pragma once
#include <cstdint>
#include "formats.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// epr:   elements per 32-bit register
// shift: left shift of the raw encoding inside its slot (E2M1 sits in bits [5:2] of an 8-bit slot
//        in kind::f8f6f4 / kind::mxf8f6f4)
// vec:   number of scale groups along K (0 = no scale)
struct Instr { const char* name; const char* ptx; Fmt fmt; Sc sc; int vec; int K; int epr; int shift; };

#define PTX_F8(T)  "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32." T "." T ".f32"
#define PTX_MX8(T) "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32." T "." T ".f32.ue8m0"
#define PTX_MXF4   "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
#define PTX_NVF4_2 "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
#define PTX_NVF4_4 "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"

static const int NUM_INSTR = 16;
static const Instr INS[NUM_INSTR] = {
  {"I1  TF32",               "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",   F_TF32, S_NONE,  0,  8, 1, 0},
  {"I2  FP16",               "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",    F_F16,  S_NONE,  0, 16, 2, 0},
  {"I3  BF16",               "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",  F_BF16, S_NONE,  0, 16, 2, 0},
  {"I4  f8f6f4 E5M2",        PTX_F8("e5m2"),  F_E5M2, S_NONE,  0, 32, 4, 0},
  {"I5  f8f6f4 E4M3",        PTX_F8("e4m3"),  F_E4M3, S_NONE,  0, 32, 4, 0},
  {"I6  f8f6f4 E3M2",        PTX_F8("e3m2"),  F_E3M2, S_NONE,  0, 32, 4, 0},
  {"I7  f8f6f4 E2M3",        PTX_F8("e2m3"),  F_E2M3, S_NONE,  0, 32, 4, 0},
  {"I8  f8f6f4 E2M1",        PTX_F8("e2m1"),  F_E2M1, S_NONE,  0, 32, 4, 2},
  {"I9  mxf8f6f4 1X E5M2",   PTX_MX8("e5m2"), F_E5M2, S_UE8M0, 1, 32, 4, 0},
  {"I10 mxf8f6f4 1X E4M3",   PTX_MX8("e4m3"), F_E4M3, S_UE8M0, 1, 32, 4, 0},
  {"I11 mxf8f6f4 1X E3M2",   PTX_MX8("e3m2"), F_E3M2, S_UE8M0, 1, 32, 4, 0},
  {"I12 mxf8f6f4 1X E2M3",   PTX_MX8("e2m3"), F_E2M3, S_UE8M0, 1, 32, 4, 0},
  {"I13 mxf8f6f4 1X E2M1",   PTX_MX8("e2m1"), F_E2M1, S_UE8M0, 1, 32, 4, 2},
  {"I14 mxf4 2X",            PTX_MXF4,        F_E2M1, S_UE8M0, 2, 64, 8, 0},
  {"I15 mxf4nvf4 2X UE8M0",  PTX_NVF4_2,      F_E2M1, S_UE8M0, 2, 64, 8, 0},
  {"I16 mxf4nvf4 4X UE4M3",  PTX_NVF4_4,      F_E2M1, S_UE4M3, 4, 64, 8, 0},
};

// Fractional bits kept after alignment (measured by test3): 35 on the dedicated FP4 path, 25 elsewhere.
static inline int precision_bits(int id) { return id >= 13 ? 35 : 25; }

#define MMA_PLAIN(S) asm volatile(S " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
  : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) \
  : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
// scale_A and scale_B registers, each followed by byte-id and thread-id selectors (both 0)
#define MMA_SCALED(S) asm volatile(S " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n" \
  : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) \
  : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), \
    "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z))

__device__ __forceinline__ void do_mma(int id, uint32_t (&a)[4], uint32_t (&b)[2], float (&d)[4],
                                       uint32_t sfa, uint32_t sfb) {
  uint16_t z = 0;
  switch (id) {
    case 0:  MMA_PLAIN("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"); break;
    case 1:  MMA_PLAIN("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"); break;
    case 2:  MMA_PLAIN("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"); break;
    case 3:  MMA_PLAIN(PTX_F8("e5m2")); break;
    case 4:  MMA_PLAIN(PTX_F8("e4m3")); break;
    case 5:  MMA_PLAIN(PTX_F8("e3m2")); break;
    case 6:  MMA_PLAIN(PTX_F8("e2m3")); break;
    case 7:  MMA_PLAIN(PTX_F8("e2m1")); break;
    case 8:  MMA_SCALED(PTX_MX8("e5m2")); break;
    case 9:  MMA_SCALED(PTX_MX8("e4m3")); break;
    case 10: MMA_SCALED(PTX_MX8("e3m2")); break;
    case 11: MMA_SCALED(PTX_MX8("e2m3")); break;
    case 12: MMA_SCALED(PTX_MX8("e2m1")); break;
    case 13: MMA_SCALED(PTX_MXF4); break;
    case 14: MMA_SCALED(PTX_NVF4_2); break;
    case 15: MMA_SCALED(PTX_NVF4_4); break;
  }
}

// Fragment layout of m16n8kK with row.col operands (g = lane >> 2, t = lane % 4).
// Register r, element i of a lane holds:
//   A: row = g + 8*(r%2),      col = epr*t + i + (K/2)*(r/2)
//   B: row = epr*t + i + (K/2)*r, col = g
//   C/D (register l): row = g + 8*(l/2), col = 2*t + l%2
__host__ __device__ inline void a_pos(int K, int epr, int lane, int r, int i, int& row, int& col) {
  int g = lane >> 2, t = lane & 3;
  row = g + 8 * (r & 1);
  col = t * epr + i + (K / 2) * (r >> 1);
}
__host__ __device__ inline void b_pos(int K, int epr, int lane, int r, int i, int& row, int& col) {
  int g = lane >> 2, t = lane & 3;
  row = t * epr + i + (K / 2) * r;
  col = g;
}
__host__ __device__ inline void c_pos(int lane, int l, int& row, int& col) {
  int g = lane >> 2, t = lane & 3;
  row = g + 8 * (l >> 1);
  col = 2 * t + (l & 1);
}

// One warp (one block) per problem. A is 16xK, B is Kx8, C/D are 16x8, all row-major.
// A and B hold slot values (raw encoding already shifted into its slot).
__global__ void k_full(int id, int K, int epr, const uint32_t* A, const uint32_t* B, const float* C,
                       const uint32_t* SFA, const uint32_t* SFB, float* D) {
  int inst = blockIdx.x, lane = threadIdx.x;
  const uint32_t* Ai = A + (size_t)inst * 16 * K;
  const uint32_t* Bi = B + (size_t)inst * K * 8;
  const float*    Ci = C + (size_t)inst * 128;
  int sb = 32 / epr;
  uint32_t mask = sb == 32 ? 0xFFFFFFFFu : ((1u << sb) - 1);
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
  float d[4];
  for (int r = 0; r < 4; ++r)
    for (int i = 0; i < epr; ++i) { int row, col; a_pos(K, epr, lane, r, i, row, col); a[r] |= (Ai[row * K + col] & mask) << (sb * i); }
  for (int r = 0; r < 2; ++r)
    for (int i = 0; i < epr; ++i) { int row, col; b_pos(K, epr, lane, r, i, row, col); b[r] |= (Bi[row * 8 + col] & mask) << (sb * i); }
  for (int l = 0; l < 4; ++l) { int row, col; c_pos(lane, l, row, col); d[l] = Ci[row * 8 + col]; }
  do_mma(id, a, b, d, SFA[inst], SFB[inst]);
  for (int l = 0; l < 4; ++l) { int row, col; c_pos(lane, l, row, col); D[(size_t)inst * 128 + row * 8 + col] = d[l]; }
}

// Broadcast probe (test1): every row of A equals AR, every column of B equals BC, every C equals Cv.
// Writes the number of D elements whose bits differ from D[0][0], and the bits of D[0][0].
__global__ void k_bcast(int id, int K, int epr, const uint32_t* AR, const uint32_t* BC, const uint32_t* Cv,
                        const uint32_t* SFA, const uint32_t* SFB, int* mism, uint32_t* d00) {
  int inst = blockIdx.x, lane = threadIdx.x;
  int sb = 32 / epr;
  uint32_t mask = sb == 32 ? 0xFFFFFFFFu : ((1u << sb) - 1);
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
  float d[4];
  for (int r = 0; r < 4; ++r)
    for (int i = 0; i < epr; ++i) { int row, col; a_pos(K, epr, lane, r, i, row, col); a[r] |= (AR[(size_t)inst * K + col] & mask) << (sb * i); }
  for (int r = 0; r < 2; ++r)
    for (int i = 0; i < epr; ++i) { int row, col; b_pos(K, epr, lane, r, i, row, col); b[r] |= (BC[(size_t)inst * K + row] & mask) << (sb * i); }
  float c = __uint_as_float(Cv[inst]);
  for (int l = 0; l < 4; ++l) d[l] = c;
  do_mma(id, a, b, d, SFA[inst], SFB[inst]);
  uint32_t ref = __shfl_sync(0xFFFFFFFFu, __float_as_uint(d[0]), 0);
  unsigned cnt = 0;
  for (int l = 0; l < 4; ++l) cnt += (__float_as_uint(d[l]) != ref);
  cnt = __reduce_add_sync(0xFFFFFFFFu, cnt);
  if (lane == 0) { mism[inst] = (int)cnt; d00[inst] = ref; }
}
