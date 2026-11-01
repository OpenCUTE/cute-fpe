// test2: summation order and grouping.
//   Method 1 (TF32, FP16, BF16, E5M2/E4M3 with and without MX): coarse C-ulp sweep + cancellation pairs
//   Method 2 (E3M2, E2M3, E2M1 with and without MX): +1/-1 pairs at every position with C = 2^-24
//   Method 3 (MXFP4 on kind::mxf4 / kind::mxf4nvf4 2X): grouping width of the dedicated FP4 path
#include "../../common/harness.cuh"

static void method1(Checker& ck, int id) {
  const Instr& I = INS[id]; int K = I.K, lK = (int)std::log2((double)K);
  // 1.1 coarse filter: A = B = 1, C from 2^(23+log2 K) down to 2^24
  Prob base = mk(I);
  for (auto& v : base.A) v = slot(I, 1.0);
  for (auto& v : base.B) v = slot(I, 1.0);
  int g = 1, eHit = 0; float dHit = 0;
  for (int e = 23 + lK; e >= 24; --e) {
    Prob p = base; for (auto& c : p.C) c = ldexpf(1.0f, e);
    float D = run1(id, p);
    if (D != ldexpf(1.0f, e)) { g = 1 << (e - 23); eHit = e; dHit = D; break; }
  }
  // 1.2 +-X pair at (k1, k2), every other product y^2, C = y^2
  const double X = 256, y = ldexp(1.0, -6);
  std::vector<Prob> ps;
  for (int k1 = 0; k1 < K; ++k1) for (int k2 = 0; k2 < K; ++k2) {
    if (k1 == k2) continue;
    Prob p = mk(I);
    for (int k = 0; k < K; ++k) {
      p.A[k] = slot(I, k == k1 ? X : k == k2 ? -X : y);
      p.B[k * 8] = slot(I, (k == k1 || k == k2) ? X : y);
    }
    for (auto& c : p.C) c = (float)(y * y);
    ps.push_back(p);
  }
  std::vector<float> D = run_full(id, ps);
  int zero = 0;
  for (size_t i = 0; i < ps.size(); ++i) zero += D[i * 128] == 0.0f;
  printf("  %-24s K=%-2d 1.1: first D != C at C = 2^%d (D = %s) -> g = %d;  1.2: D = 0 in %d/%zu pairs\n",
         I.name, K, eHit, hexf(dHit).c_str(), g, zero, ps.size());
  ck.check(eHit == 25 && dHit == ldexpf(1.0f, 25) + (float)K, "%-24s 1.1 hits at C = 2^25 with D = 2^25 + K (set by p = 25, not by grouping)", I.name);
  ck.check(zero == (int)ps.size(), "%-24s 1.2 every pair gives D = 0 -> fully fused summation", I.name);
}

static void method2(Checker& ck, int id) {
  const Instr& I = INS[id]; int K = I.K;
  std::vector<Prob> ps;
  for (int k1 = 0; k1 < K; ++k1) for (int k2 = 0; k2 < K; ++k2) {
    if (k1 == k2) continue;
    Prob p = mk(I);
    for (int k = 0; k < K; ++k) p.B[k * 8] = slot(I, 1.0);
    p.A[k1] = slot(I, 1.0); p.A[k2] = slot(I, -1.0);
    for (auto& c : p.C) c = ldexpf(1.0f, -24);
    ps.push_back(p);
  }
  std::vector<float> D = run_full(id, ps);
  int keep = 0;
  for (size_t i = 0; i < ps.size(); ++i) keep += D[i * 128] == ldexpf(1.0f, -24);
  ck.check(keep == (int)ps.size(), "%-24s D = 2^-24 in %d/%zu pairs -> fully fused summation", I.name, keep, ps.size());
}

// Big +1/-1 pair in the scale group big_grp (scale 2^hi); probes p1 < p2 = 1 scan the other 32-wide group.
// Returns counts {same-16 D=2, same-16 D=0, cross-16 D=2, cross-16 D=0, other}.
static std::vector<int> method3(int id, int hi, int big_grp) {
  const Instr& I = INS[id];
  int bk0 = big_grp == 0 ? 0 : 32, pk0 = big_grp == 0 ? 32 : 0;
  std::vector<Prob> ps; std::vector<std::pair<int, int>> pr;
  for (int p1 = pk0; p1 < pk0 + 32; ++p1) for (int p2 = p1 + 1; p2 < pk0 + 32; ++p2) {
    Prob p = mk(I);
    for (int k = 0; k < 64; ++k) p.B[k * 8] = slot(I, 1.0);
    p.A[bk0] = slot(I, 1.0); p.A[bk0 + 1] = slot(I, -1.0);
    p.A[p1] = slot(I, 1.0); p.A[p2] = slot(I, 1.0);
    double s0 = big_grp == 0 ? ldexp(1.0, hi) : 1.0, s1 = big_grp == 0 ? 1.0 : ldexp(1.0, hi);
    p.sfa = sreg(I, s0, s1); p.sfb = sreg(I, 1, 1);
    ps.push_back(p); pr.push_back({p1, p2});
  }
  std::vector<float> D = run_full(id, ps);
  std::vector<int> c(5, 0);
  for (size_t i = 0; i < ps.size(); ++i) {
    float d = D[i * 128];
    bool same = (pr[i].first - pk0) / 16 == (pr[i].second - pk0) / 16;
    if (d == 2) c[same ? 0 : 2]++; else if (d == 0) c[same ? 1 : 3]++; else c[4]++;
  }
  printf("  %-24s pair in k=%d..%d, scale 2^%d, probes in k=%d..%d: same block of 16 -> D=2: %d, D=0: %d | across blocks -> D=2: %d, D=0: %d | other: %d\n",
         I.name, bk0, bk0 + 1, hi, pk0, pk0 + 31, c[0], c[1], c[2], c[3], c[4]);
  return c;
}

int main() {
  print_header("test2: summation order and grouping");
  Checker ck;
  printf("-- method 1\n");
  for (int id : {0, 1, 2, 3, 4, 8, 9}) method1(ck, id);
  printf("\n-- method 2\n");
  for (int id : {5, 6, 7, 10, 11, 12}) method2(ck, id);
  printf("\n-- method 3\n");
  for (int id : {13, 14}) {
    std::vector<int> a = method3(id, 36, 0), b = method3(id, 37, 0), c = method3(id, 36, 1);
    ck.check(a[0] == 240 && a[3] == 256 && a[1] + a[2] + a[4] == 0, "%-24s scale 2^36: D = 2 iff both probes are in the same block of 16", INS[id].name);
    ck.check(b[1] == 240 && b[3] == 256 && b[0] + b[2] + b[4] == 0, "%-24s control scale 2^37: D = 0 for all 496 pairs", INS[id].name);
    ck.check(c[0] == 240 && c[3] == 256 && c[1] + c[2] + c[4] == 0, "%-24s reversed (pair in k=32..33, probes in k=0..31): same result", INS[id].name);
  }
  return ck.finish("test2");
}
