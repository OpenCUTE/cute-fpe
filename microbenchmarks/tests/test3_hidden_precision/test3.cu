// test3: hidden accumulation precision p (values more than p bits below the aligned maximum are dropped).
#include "../../common/harness.cuh"

// Method 1.1: +-2^16 pair at k = 0, 1; every other product 2^(t-6); C[0][0] = 2^(t-6). p = 21 - t.
static int method1(int id) {
  const Instr& I = INS[id]; int K = I.K;
  for (int t = 0; t >= -6; --t) {
    Prob q = mk(I);
    for (int k = 0; k < K; ++k) {
      q.A[k] = slot(I, k == 0 ? 256.0 : k == 1 ? -256.0 : ldexp(1.0, t));
      q.B[k * 8] = slot(I, k <= 1 ? 256.0 : ldexp(1.0, -6));
    }
    q.C[0] = ldexpf(1.0f, t - 6);
    if (run1(id, q) == 0.0f) { printf("  %-24s first D = 0 at t = %d\n", I.name, t); return 21 - t; }
  }
  printf("  %-24s D never 0 for t = 0..-6\n", I.name);
  return -1;
}

// Method 2: +-P pair at k = 0, 1; C = 2^c halved until D = 0. p = log2(P) - c - 1.
static int method2(int id) {
  const Instr& I = INS[id]; int K = I.K;
  bool mx = I.sc != S_NONE;
  int lP = mx ? 24 : 4;
  for (int c = 0; c >= -40; --c) {
    Prob q = mk(I);
    if (mx) {        // A = +-1, B column 0 = 1, scale_A = scale_B = 2^12
      q.A[0] = slot(I, 1.0); q.A[1] = slot(I, -1.0);
      for (int k = 0; k < K; ++k) q.B[k * 8] = slot(I, 1.0);
      q.sfa = q.sfb = sreg(I, 4096.0);
    } else {         // A = +-4, B = 4 everywhere
      q.A[0] = slot(I, 4.0); q.A[1] = slot(I, -4.0);
      for (auto& b : q.B) b = slot(I, 4.0);
    }
    for (auto& cc : q.C) cc = ldexpf(1.0f, c);
    if (run1(id, q) == 0.0f) { printf("  %-24s P = 2^%d, first D = 0 at C = 2^%d\n", I.name, lP, c); return lP - c - 1; }
  }
  return -1;
}

int main() {
  print_header("test3: hidden accumulation precision");
  Checker ck;
  printf("-- method 1.1\n");
  for (int id : {0, 1, 2, 3, 4, 8, 9}) { int p = method1(id); ck.check(p == 25, "%-24s p = %d (expected 25)", INS[id].name, p); }
  printf("\n-- method 2\n");
  for (int id : {5, 6, 7, 10, 11, 12}) { int p = method2(id); ck.check(p == 25, "%-24s p = %d (expected 25)", INS[id].name, p); }

  printf("\n-- method 3.1 MXFP4 (I14): A[0][0] = 1, A[0][63] = -1, B = 1, C = 1, both A scales 2^Sa\n");
  {
    const Instr& I = INS[13]; int hit = -1;
    for (int Sa = 0; Sa <= 60; ++Sa) {
      Prob q = mk(I);
      q.A[0] = slot(I, 1.0); q.A[63] = slot(I, -1.0);
      for (auto& b : q.B) b = slot(I, 1.0);
      q.sfa = sreg(I, ldexp(1.0, Sa), ldexp(1.0, Sa)); q.sfb = sreg(I, 1, 1);
      for (auto& c : q.C) c = 1.0f;
      if (run1(13, q) == 0.0f) { hit = Sa; break; }
    }
    printf("  %-24s first D = 0 at Sa = %d\n", I.name, hit);
    ck.check(hit - 1 == 35, "%-24s p = %d (expected 35)", I.name, hit - 1);
  }
  printf("\n-- method 3.2 NVFP4 (I16): C = 2^24 cancelled by group 4, small product 2^(n-7) in group 1\n");
  {
    const Instr& I = INS[15]; int hit = 99;
    for (int n = 0; n >= -9; --n) {
      Prob q = mk(I);
      for (int k = 48; k < 64; ++k) { q.A[k] = slot(I, -4.0); q.B[k * 8] = slot(I, 4.0); }
      q.A[0] = slot(I, 1.0); q.B[0] = slot(I, 1.0);
      q.sfa = sreg(I, ldexp(1.0, n), 1, 1, 256); q.sfb = sreg(I, ldexp(1.0, -7), 1, 1, 256);
      for (auto& c : q.C) c = ldexpf(1.0f, 24);
      if (run1(15, q) == 0.0f) { hit = n; break; }
    }
    printf("  %-24s first D = 0 at n = %d\n", I.name, hit);
    ck.check(30 - hit == 35, "%-24s p = %d (expected 35)", I.name, 30 - hit);
  }
  return ck.finish("test3");
}
