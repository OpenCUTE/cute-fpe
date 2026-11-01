// Host-side encode/decode of the operand formats and scale formats used by the probes.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

enum Fmt { F_TF32, F_F16, F_BF16, F_E5M2, F_E4M3, F_E3M2, F_E2M3, F_E2M1 };
enum Sc  { S_NONE, S_UE8M0, S_UE4M3 };

// Bits of the raw encoding (TF32 is handled as a 32-bit FP32 pattern).
static inline int fbits(Fmt f) {
  switch (f) {
    case F_TF32: return 32;
    case F_F16: case F_BF16: return 16;
    case F_E5M2: case F_E4M3: return 8;
    case F_E3M2: case F_E2M3: return 6;
    default: return 4;
  }
}

// Generic minifloat decoder.
// kind 0: IEEE-style (all-ones exponent is Inf/NaN); 1: E4M3-style (only all-ones is NaN); 2: all finite.
static inline double mini(uint32_t raw, int E, int M, int bias, int kind, bool sign) {
  uint32_t s = sign ? (raw >> (E + M)) & 1 : 0;
  uint32_t e = (raw >> M) & ((1u << E) - 1), m = raw & ((1u << M) - 1);
  uint32_t emax = (1u << E) - 1;
  if (kind == 0 && e == emax) return m ? NAN : (s ? -INFINITY : INFINITY);
  if (kind == 1 && e == emax && m == (1u << M) - 1) return NAN;
  double v = e == 0 ? ldexp((double)m, 1 - bias - M) : ldexp((double)((1u << M) + m), (int)e - bias - M);
  return s ? -v : v;
}

static inline double decode(Fmt f, uint32_t raw) {
  switch (f) {
    case F_TF32: { uint32_t u = raw & 0xFFFFE000u; float x; memcpy(&x, &u, 4); return x; }
    case F_F16:  return mini(raw, 5, 10, 15, 0, true);
    case F_BF16: return mini(raw, 8, 7, 127, 0, true);
    case F_E5M2: return mini(raw, 5, 2, 15, 0, true);
    case F_E4M3: return mini(raw, 4, 3, 7, 1, true);
    case F_E3M2: return mini(raw, 3, 2, 3, 2, true);
    case F_E2M3: return mini(raw, 2, 3, 1, 2, true);
    default:     return mini(raw, 2, 1, 1, 2, true);
  }
}

static inline double decode_sc(Sc s, uint32_t raw) {
  if (s == S_UE8M0) return raw == 255 ? NAN : ldexp(1.0, (int)raw - 127);
  return mini(raw & 0x7F, 4, 3, 7, 1, false);   // UE4M3: the top bit is not part of the value
}

static inline bool same_value(double x, double v) { return x == v && std::signbit(x) == std::signbit(v); }

// Exact encoder: aborts if the value is not representable.
static inline uint32_t encode(Fmt f, double v) {
  if (f == F_TF32) {
    float x = (float)v; uint32_t u; memcpy(&u, &x, 4);
    if ((double)x != v || (u & 0x1FFF)) { printf("tf32 cannot represent %g\n", v); exit(2); }
    return u;
  }
  int w = fbits(f);
  for (uint32_t r = 0; r < (1u << w); ++r) if (same_value(decode(f, r), v)) return r;
  printf("format %d cannot represent %g\n", (int)f, v); exit(2);
}

static inline uint32_t encode_sc(Sc s, double v) {
  for (uint32_t r = 0; r < (s == S_UE8M0 ? 256u : 128u); ++r) if (decode_sc(s, r) == v) return r;
  printf("scale format %d cannot represent %g\n", (int)s, v); exit(2);
}

static inline uint32_t fb(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline float bf(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
