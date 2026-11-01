// test8: exponent semantics of a structural zero (one factor is 0) versus a cancellation to zero.
// A large term that is zero is placed next to a small C. If its exponent took part in the
// alignment, C would be shifted out (D = 0); if the exponent is set to the minimum, D = C.
#include "../../common/harness.cuh"

// Largest power of two (exponent <= 20) representable in the element format.
static int max_pow2(Fmt f) {
  int best = 0;
  for (int e = 0; e <= 20; ++e) {
    double v = ldexp(1.0, e); bool ok = f == F_TF32;
    for (uint32_t r = 0; !ok && r < (1u << fbits(f)); ++r) ok = decode(f, r) == v;
    if (ok) best = e;
  }
  return best;
}

int main() {
  print_header("test8: structural zero vs cancellation to zero");
  Checker ck;

  printf("-- (1) no scale (I1-I8): X = 2^e (largest power of two, e <= 20), C = 2^(2e-30)\n");
  for (int id = 0; id < 8; ++id) {
    const Instr& I = INS[id]; int K = I.K;
    int e = max_pow2(I.fmt); double X = ldexp(1.0, e); float C = ldexpf(1.0f, 2 * e - 30);
    std::vector<Prob> ps;
    { Prob p = mk(I); p.A[0] = slot(I, 0.0); p.B[0] = slot(I, X); p.C[0] = C; ps.push_back(p); }
    { Prob p = mk(I); p.A[0] = slot(I, X); p.B[0] = slot(I, 0.0); p.C[0] = C; ps.push_back(p); }
    { Prob p = mk(I);
      for (int k = 0; k < K; ++k) { p.A[k] = slot(I, k % 2 ? 0.0 : X); p.B[k * 8] = slot(I, k % 2 ? X : 0.0); }
      p.C[0] = C; ps.push_back(p); }
    { Prob p = mk(I); p.A[0] = slot(I, X); p.A[1] = slot(I, -X); p.B[0] = slot(I, X); p.B[8] = slot(I, X);
      p.C[0] = C; ps.push_back(p); }
    std::vector<float> D = run_full(id, ps);
    ck.check(D[0] == C && D[128] == C && D[256] == C && D[384] == 0.0f,
             "%-24s X = 2^%-2d C = %-9s Z1 = %-9s Z2 = %-9s Z3 = %-9s N = %s", I.name, e, hexf(C).c_str(),
             hexf(D[0]).c_str(), hexf(D[128]).c_str(), hexf(D[256]).c_str(), hexf(D[384]).c_str());
  }

  printf("\n-- (2) UE8M0 scale (I9-I15): scale_A = 2^40, scale_B = 1, C = 1\n");
  for (int id = 8; id < 15; ++id) {
    const Instr& I = INS[id]; int K = I.K;
    double S = ldexp(1.0, 40);
    auto prep = [&]() { Prob p = mk(I); p.sfa = sreg(I, S, S, S, S); p.sfb = sreg(I, 1, 1, 1, 1); p.C[0] = 1.0f; return p; };
    std::vector<Prob> ps;
    { Prob p = prep(); for (int k = 0; k < K; ++k) p.A[k] = slot(I, 1.0); ps.push_back(p); }
    { Prob p = prep(); for (int k = 0; k < K; ++k) { p.A[k] = slot(I, k % 2 ? 0.0 : 1.0); p.B[k * 8] = slot(I, k % 2 ? 1.0 : 0.0); } ps.push_back(p); }
    { Prob p = prep(); for (int k = 0; k < K; ++k) p.A[k] = slot(I, 1.0);
      p.A[1] = slot(I, -1.0); p.B[0] = slot(I, 1.0); p.B[8] = slot(I, 1.0); ps.push_back(p); }
    { Prob p = prep(); p.A[0] = slot(I, 1.0); p.B[0] = slot(I, 1.0); ps.push_back(p); }
    std::vector<float> D = run_full(id, ps);
    ck.check(D[0] == 1.0f && D[128] == 1.0f && D[256] == 0.0f && D[384] == ldexpf(1.0f, 40),
             "%-24s scale_A = 0x%08X  Z = %s  Zmix = %s  N = %s  control = %s", I.name, ps[0].sfa,
             hexf(D[0]).c_str(), hexf(D[128]).c_str(), hexf(D[256]).c_str(), hexf(D[384]).c_str());
  }

  printf("\n-- (3) NVFP4 (I16): group 1 only, C = 2^-40\n");
  {
    const int id = 15; const Instr& I = INS[id];
    const float C = ldexpf(1.0f, -40);
    struct R { const char* name; Prob p; float expect; };
    std::vector<R> rs;
    auto base = [&]() { Prob p = mk(I); p.C[0] = C; return p; };
    auto big = [&](uint32_t sa0raw, uint32_t sb0raw) {
      Prob p = base();
      for (int k = 0; k < 16; ++k) { p.A[k] = slot(I, 4.0); p.B[k * 8] = slot(I, 4.0); }
      p.sfa = with_byte0(sreg(I, 1, 1, 1, 1), sa0raw); p.sfb = with_byte0(sreg(I, 1, 1, 1, 1), sb0raw);
      return p; };
    { Prob p = base(); p.sfa = p.sfb = sreg(I, 256, 1, 1, 1); rs.push_back({"R  reference: a = b = 0", p, C}); }
    { Prob p = base(); for (int k = 0; k < 16; ++k) p.A[k] = slot(I, 1.0);
      p.sfa = p.sfb = sreg(I, 256, 1, 1, 1); rs.push_back({"Z  structural zero: a = 1, b = 0", p, C}); }
    { Prob p = base(); for (int k = 0; k < 16; ++k) p.A[k] = slot(I, 1.0);
      p.A[1] = slot(I, -1.0); p.B[0] = slot(I, 1.0); p.B[8] = slot(I, 1.0);
      p.sfa = p.sfb = sreg(I, 256, 1, 1, 1); rs.push_back({"N  cancellation: 1 + (-1)", p, 0.0f}); }
    rs.push_back({"S1 a = b = 4, scale_A0 = 0x00", big(0x00, 0x78), C});
    rs.push_back({"S2 a = b = 4, scale_A0 = 0x80", big(0x80, 0x78), C});
    rs.push_back({"S3 a = b = 4, scale_B0 = 0x00", big(0x78, 0x00), C});
    rs.push_back({"control a = b = 4, scale_A0 = 0x01 (2^-9)", big(0x01, 0x78), 128.0f});
    std::vector<Prob> ps; for (auto& r : rs) ps.push_back(r.p);
    std::vector<float> D = run_full(id, ps);
    for (size_t i = 0; i < rs.size(); ++i)
      ck.check(D[i * 128] == rs[i].expect, "%-44s scale_A = 0x%08X scale_B = 0x%08X  D = %s (expected %s)", rs[i].name,
               rs[i].p.sfa, rs[i].p.sfb, hexf(D[i * 128]).c_str(), hexf(rs[i].expect).c_str());
  }
  return ck.finish("test8");
}
