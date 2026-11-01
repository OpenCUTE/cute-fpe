// test5: special values (NaN, Inf, out-of-range intermediates, NaN scales) and signed zero.
// Every case has an expected FP32 bit pattern for D[0][0].
#include "../../common/harness.cuh"

// Raw encodings (before slot shift) of the special values of each element format.
struct Codes { bool inf, nan; uint32_t pinf, ninf, qnan, sign, maxv; };
static Codes codes(Fmt f) {
  switch (f) {
    case F_TF32: return {true,  true,  0x7F800000, 0xFF800000, 0x7FC00000, 0x80000000, 0x7F000000};  // "max" = 2^127
    case F_F16:  return {true,  true,  0x7C00, 0xFC00, 0x7E00, 0x8000, 0x7BFF};
    case F_BF16: return {true,  true,  0x7F80, 0xFF80, 0x7FC0, 0x8000, 0x7F00};                      // "max" = 2^127
    case F_E5M2: return {true,  true,  0x7C, 0xFC, 0x7E, 0x80, 0x7B};
    case F_E4M3: return {false, true,  0, 0, 0x7F, 0x80, 0x7E};
    case F_E3M2: return {false, false, 0, 0, 0, 0x20, 0x1F};
    case F_E2M3: return {false, false, 0, 0, 0, 0x20, 0x1F};
    default:     return {false, false, 0, 0, 0, 0x8, 0x7};
  }
}

static const uint32_t K_PINF = 0x7F800000, K_NINF = 0xFF800000, K_QNAN = 0x7FC00000, K_CNAN = 0x7FFFFFFF, K_NZERO = 0x80000000;
typedef std::vector<std::pair<int, uint32_t>> Elems;   // (k, slot value)

struct Case {
  std::string name; Elems a, b; uint32_t c; uint32_t expect;
  bool set_sf = false; uint32_t sfa = 0, sfb = 0;
};
static Case scaled(Case c, uint32_t sfa, uint32_t sfb) { c.set_sf = true; c.sfa = sfa; c.sfb = sfb; return c; }

static void run_cases(Checker& ck, int id, const std::vector<Case>& cs) {
  const Instr& I = INS[id];
  std::vector<Prob> ps;
  for (const Case& c : cs) {
    Prob p = mk(I);
    for (auto& e : c.a) p.A[e.first] = e.second;
    for (auto& e : c.b) p.B[e.first * 8] = e.second;
    p.C[0] = bf(c.c);
    if (c.set_sf) { p.sfa = c.sfa; p.sfb = c.sfb; }
    ps.push_back(p);
  }
  std::vector<float> D = run_full(id, ps);
  for (size_t i = 0; i < cs.size(); ++i) {
    uint32_t u = fb(D[i * 128]);
    ck.check(u == cs[i].expect, "%-24s %-52s D = 0x%08X (expected 0x%08X)", I.name, cs[i].name.c_str(), u, cs[i].expect);
  }
}

int main() {
  print_header("test5: special values and signed zero");
  Checker ck;

  printf("-- 4.1 special element values\n");
  for (int id : {0, 1, 2, 3, 4, 8, 9}) {
    const Instr& I = INS[id]; Codes cd = codes(I.fmt); int s = I.shift;
    uint32_t one = slot(I, 1.0);
    std::vector<Case> cs;
    if (cd.inf) {
      uint32_t pi = cd.pinf << s, ni = cd.ninf << s, mx = cd.maxv << s, nmx = (cd.maxv | cd.sign) << s;
      cs.push_back({"E1 a0=+Inf b0=1", {{0, pi}}, {{0, one}}, 0, K_PINF});
      cs.push_back({"E2 a0=-Inf b0=1", {{0, ni}}, {{0, one}}, 0, K_NINF});
      cs.push_back({"E3 a0=+Inf b0=0", {{0, pi}}, {{0, 0}}, 0, K_CNAN});
      cs.push_back({"E4 a0=+Inf a1=-Inf b0=b1=1", {{0, pi}, {1, ni}}, {{0, one}, {1, one}}, 0, K_CNAN});
      cs.push_back({"E5 a0=a1=+Inf b0=b1=1", {{0, pi}, {1, pi}}, {{0, one}, {1, one}}, 0, K_PINF});
      cs.push_back({"E6 a0=+Inf b0=1 C=-Inf", {{0, pi}}, {{0, one}}, K_NINF, K_CNAN});
      cs.push_back({"E7 a0=+Inf b0=1 C=+Inf", {{0, pi}}, {{0, one}}, K_PINF, K_PINF});
      cs.push_back({"E8 a0=+Inf b0=1, a1=-max b1=max", {{0, pi}, {1, nmx}}, {{0, one}, {1, mx}}, 0, K_PINF});
    }
    if (cd.nan) {
      uint32_t qn = cd.qnan << s;
      cs.push_back({"E9 a0=NaN b0=1", {{0, qn}}, {{0, one}}, 0, K_CNAN});
      cs.push_back({"E10 a0=NaN b0=0", {{0, qn}}, {{0, 0}}, 0, K_CNAN});
      cs.push_back({"E11 a0=NaN b0=1 C=+Inf", {{0, qn}}, {{0, one}}, K_PINF, K_CNAN});
    }
    run_cases(ck, id, cs);
  }

  printf("\n-- 4.2 special C with finite A, B\n");
  for (int id = 0; id < NUM_INSTR; ++id) {
    uint32_t one = slot(INS[id], 1.0);
    run_cases(ck, id, {
      {"C1 C=0x7FC00000 (qNaN), a=b=0", {}, {}, K_QNAN, K_CNAN},
      {"C2 C=0x7F800001 (sNaN), a=b=0", {}, {}, 0x7F800001, K_CNAN},
      {"C3 C=0xFFC01234 (negative NaN with payload), a0=b0=1", {{0, one}}, {{0, one}}, 0xFFC01234, K_CNAN},
      {"C4 C=+Inf, a0=b0=1", {{0, one}}, {{0, one}}, K_PINF, K_PINF},
      {"C5 C=-Inf, a0=b0=1", {{0, one}}, {{0, one}}, K_NINF, K_NINF}});
  }

  printf("\n-- 4.3 out-of-range intermediates\n");
  for (int id = 0; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id]; Codes cd = codes(I.fmt); int s = I.shift;
    uint32_t mx = cd.maxv << s, nmx = (cd.maxv | cd.sign) << s, one = slot(I, 1.0);
    double m = decode(I.fmt, cd.maxv), sq = m * m;
    uint32_t v1 = std::isfinite((float)sq) ? fb((float)sq) : K_PINF;
    std::vector<Case> cs;
    cs.push_back({"V1 a0=b0=max", {{0, mx}}, {{0, mx}}, 0, v1});
    if (I.sc == S_UE8M0) {
      const uint32_t BIG = 0xFEFEFEFEu;                 // 2^127 in every byte
      cs.push_back(scaled({"V2 a0=b0=max, scale_A=scale_B=2^127", {{0, mx}}, {{0, mx}}, 0, K_PINF}, BIG, BIG));
      cs.push_back(scaled({"V3 V2 + a1=-max b1=max (cancel)", {{0, mx}, {1, nmx}}, {{0, mx}, {1, mx}}, 0, 0}, BIG, BIG));
      cs.push_back(scaled({"V4 V2 + C=-Inf", {{0, mx}}, {{0, mx}}, K_NINF, K_NINF}, BIG, BIG));
      cs.push_back(scaled({"V5 a0=b0=1, scale_A=2^127, scale_B=2^1", {{0, one}}, {{0, one}}, 0, K_PINF}, BIG, 0x80808080u));
    } else if (I.fmt == F_TF32 || I.fmt == F_BF16) {
      cs.push_back({"V2 a0=b0=2^127 (2^254 overflows FP32)", {{0, mx}}, {{0, mx}}, 0, K_PINF});
      cs.push_back({"V3 V2 + a1=-2^127 b1=2^127 (cancel)", {{0, mx}, {1, nmx}}, {{0, mx}, {1, mx}}, 0, 0});
      cs.push_back({"V4 V2 + C=-Inf", {{0, mx}}, {{0, mx}}, K_NINF, K_NINF});
    } else if (I.sc == S_UE4M3) {
      uint32_t r = sreg(I, 448, 1, 1, 1);
      cs.push_back(scaled({"V2 a0=b0=6, scale_A0=scale_B0=448", {{0, mx}}, {{0, mx}}, 0, fb(7225344.0f)}, r, r));
    }
    run_cases(ck, id, cs);
  }

  printf("\n-- 4.4 NaN scale\n");
  for (int id = 8; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id]; uint32_t one = slot(I, 1.0);
    uint32_t base = sreg(I, 1, 1, 1, 1), nraw = I.sc == S_UE8M0 ? 0xFF : 0x7F, nanA = with_byte0(base, nraw);
    Elems az; for (int k = 0; k < 16; ++k) az.push_back({k, one});
    std::vector<Case> cs;
    cs.push_back(scaled({"N1 scale_A0=NaN, a0=b0=1", {{0, one}}, {{0, one}}, 0, K_CNAN}, nanA, base));
    cs.push_back(scaled({"N2 scale_B0=NaN, a0=b0=1", {{0, one}}, {{0, one}}, 0, K_CNAN}, base, nanA));
    cs.push_back(scaled({"N3 scale_A0=NaN, k=0..15: a=1, b=0", az, {}, 0, K_CNAN}, nanA, base));
    cs.push_back(scaled({"N4 scale_A0=NaN, a=b=0", {}, {}, 0, K_CNAN}, nanA, base));
    cs.push_back(scaled({"N5 scale_A0=NaN, a0=b0=1, C=+Inf", {{0, one}}, {{0, one}}, K_PINF, K_CNAN}, nanA, base));
    if (I.vec >= 2) {
      int kg = I.K / I.vec;
      cs.push_back(scaled({"N6 scale_A0=NaN, only group 2 non-zero", {{kg, one}}, {{kg, one}}, 0, K_CNAN}, nanA, base));
    }
    if (I.sc == S_UE4M3) {
      cs.push_back(scaled({"N7 scale_A0=0xFF", {{0, one}}, {{0, one}}, 0, K_CNAN}, with_byte0(base, 0xFF), base));
      cs.push_back(scaled({"N8 scale_A0=0x7E (448)", {{0, one}}, {{0, one}}, 0, fb(448.0f)}, with_byte0(base, 0x7E), base));
    }
    if (I.sc == S_UE8M0)
      cs.push_back(scaled({"N9 scale_A0=0xFE (2^127)", {{0, one}}, {{0, one}}, 0, 0x7F000000}, with_byte0(base, 0xFE), base));
    run_cases(ck, id, cs);
  }

  printf("\n-- 4.5 signed zero (expected +0 everywhere)\n");
  for (int id = 0; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id]; Codes cd = codes(I.fmt); int s = I.shift; int K = I.K;
    uint32_t one = slot(I, 1.0), mone = slot(I, -1.0), nz = cd.sign << s;
    Elems an, bo, ap, amix, amone, bz;
    for (int k = 0; k < K; ++k) {
      an.push_back({k, nz}); bo.push_back({k, one}); ap.push_back({k, 0});
      amix.push_back({k, k % 2 ? nz : 0u}); amone.push_back({k, mone}); bz.push_back({k, 0});
    }
    run_cases(ck, id, {
      {"Z1 all p=-0 (a=-0, b=1), C=-0", an, bo, K_NZERO, 0},
      {"Z2 all p=-0 (a=-1, b=+0), C=-0", amone, bz, K_NZERO, 0},
      {"Z3 all p=-0, C=+0", an, bo, 0, 0},
      {"Z4 all p=+0, C=-0", ap, bo, K_NZERO, 0},
      {"Z5 mixed +0/-0, C=-0", amix, bo, K_NZERO, 0},
      {"Z6 mixed +0/-0, C=+0", amix, bo, 0, 0},
      {"Z7 a0*b0=1, a1*b1=-1, C=-0", {{0, one}, {1, mone}}, {{0, one}, {1, one}}, K_NZERO, 0},
      {"Z8 a0*b0=-1, C=+1", {{0, mone}}, {{0, one}}, 0x3F800000, 0}});
  }
  return ck.finish("test5");
}
