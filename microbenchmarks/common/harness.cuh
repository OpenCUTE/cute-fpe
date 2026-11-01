// Problem construction, batched execution, and PASS/FAIL reporting shared by all probes.
#pragma once
#include <algorithm>
#include <cstdarg>
#include <string>
#include <vector>
#include "mma.cuh"

// ---------------- deterministic RNG (splitmix64) ----------------
static uint64_t g_rng_state = 0x2026102900ULL;
static inline void rng_seed(uint64_t s) { g_rng_state = s; }
static inline uint64_t rnd() {
  uint64_t z = (g_rng_state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static inline uint32_t rndu(uint32_t n) { return (uint32_t)(rnd() % n); }

// ---------------- operand helpers ----------------
// Slot value of an exactly representable element value.
static inline uint32_t slot(const Instr& I, double v) { return encode(I.fmt, v) << I.shift; }

// Scale register written identically by every thread; byte j scales K-group j.
// 1X uses byte 0 only; 2X (byte-id 0) uses bytes 0 and 1; 4X uses all four bytes.
static inline uint32_t sreg(const Instr& I, double s0, double s1 = 1, double s2 = 1, double s3 = 1) {
  if (I.sc == S_NONE) return 0;
  uint32_t b0 = encode_sc(I.sc, s0), b1 = encode_sc(I.sc, s1), b2 = encode_sc(I.sc, s2), b3 = encode_sc(I.sc, s3);
  if (I.vec == 1) return b0 * 0x01010101u;
  if (I.vec == 2) return b0 | (b1 << 8) | (b0 << 16) | (b1 << 24);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
// Replace byte 0 of a scale register with a raw byte.
static inline uint32_t with_byte0(uint32_t reg, uint32_t raw) { return (reg & ~0xFFu) | raw; }
static inline int group(const Instr& I, int k) { return I.vec <= 1 ? 0 : k / (I.K / I.vec); }

// ---------------- problems ----------------
struct Prob { std::vector<uint32_t> A, B; std::vector<float> C; uint32_t sfa, sfb; };

// All-zero A, B, C; all scales 1.0.
static inline Prob mk(const Instr& I) {
  Prob p;
  p.A.assign(16 * I.K, 0); p.B.assign(I.K * 8, 0); p.C.assign(128, 0.f);
  p.sfa = p.sfb = sreg(I, 1, 1, 1, 1);
  return p;
}

// Runs every problem with instruction id and returns the full 16x8 D of each (row-major, 128 per problem).
static std::vector<float> run_full(int id, const std::vector<Prob>& ps) {
  const Instr& I = INS[id]; int K = I.K; size_t n = ps.size();
  const size_t CH = 4096;
  static uint32_t *dA = nullptr, *dB, *dSA, *dSB; static float *dC, *dD;
  if (!dA) {
    CK(cudaMalloc(&dA, CH * 16 * 64 * 4)); CK(cudaMalloc(&dB, CH * 64 * 8 * 4));
    CK(cudaMalloc(&dSA, CH * 4)); CK(cudaMalloc(&dSB, CH * 4));
    CK(cudaMalloc(&dC, CH * 128 * 4)); CK(cudaMalloc(&dD, CH * 128 * 4));
  }
  std::vector<float> out(n * 128);
  for (size_t s = 0; s < n; s += CH) {
    size_t m = std::min(CH, n - s);
    std::vector<uint32_t> hA(m * 16 * K), hB(m * K * 8), hSA(m), hSB(m);
    std::vector<float> hC(m * 128);
    for (size_t j = 0; j < m; ++j) {
      const Prob& p = ps[s + j];
      memcpy(&hA[j * 16 * K], p.A.data(), 16 * K * 4);
      memcpy(&hB[j * K * 8], p.B.data(), K * 8 * 4);
      memcpy(&hC[j * 128], p.C.data(), 128 * 4);
      hSA[j] = p.sfa; hSB[j] = p.sfb;
    }
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, hC.data(), hC.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSA, hSA.data(), m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSB, hSB.data(), m * 4, cudaMemcpyHostToDevice));
    k_full<<<(unsigned)m, 32>>>(id, K, I.epr, dA, dB, dC, dSA, dSB, dD);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&out[s * 128], dD, m * 128 * 4, cudaMemcpyDeviceToHost));
  }
  return out;
}
static inline float run1(int id, const Prob& p) { return run_full(id, std::vector<Prob>{p})[0]; }

// ---------------- reporting ----------------
static void print_header(const char* title) {
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("==== %s ====\n", title);
  printf("device: %s (sm_%d%d), driver API %d, runtime %d\n\n", prop.name, prop.major, prop.minor, drv, rt);
}

struct Checker {
  int pass = 0, fail = 0;
  void check(bool ok, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "PASS" : "FAIL");
    vprintf(fmt, ap); printf("\n");
    va_end(ap);
    (ok ? pass : fail)++;
  }
  int finish(const char* test) {
    printf("\n%s: %d checks, %d passed, %d failed -> %s\n", test, pass + fail, pass, fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
  }
};

static inline std::string hexf(float x) { char b[64]; snprintf(b, sizeof b, "%a", x); return b; }
