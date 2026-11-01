// test9: random bits in [1:0] of the E2M1 slot, compared with the same 6-bit byte decoded as E2M3.
//   R1: E2M1 instruction, raw6 = (h4 << 2) | r2   -> D1
//   R2: E2M3 instruction, same raw6               -> D2
//   R3: E2M1 instruction, h4 << 2 (r2 = 0)        -> D3
// H1 (low bits ignored): D1 == D3 always.  H2 (shared 6-bit decoder): D1 == D2 always.
// Usage: ./test9 [rounds, default 100000]
#include "../../common/harness.cuh"

static void run_path(Checker& ck, int id1, int id3, long rounds) {
  const Instr& I1 = INS[id1]; const Instr& I3 = INS[id3];
  for (int phase = 0; phase < 2; ++phase) {
    long n = phase == 0 ? 2000 : rounds, d12 = 0, d13 = 0;
    const long CH = 8192;
    for (long s = 0; s < n; s += CH) {
      long m = std::min(CH, n - s);
      std::vector<Prob> p1, p2, p3;
      for (long j = 0; j < m; ++j) {
        Prob a = mk(I1), b = mk(I3), c = mk(I1);
        for (int i = 0; i < 16 * 32; ++i) {
          uint32_t h4 = rndu(16), r2 = phase == 0 ? 0 : rndu(4);
          a.A[i] = b.A[i] = (h4 << 2) | r2; c.A[i] = h4 << 2;
        }
        for (int i = 0; i < 32 * 8; ++i) {
          uint32_t h4 = rndu(16), r2 = phase == 0 ? 0 : rndu(4);
          a.B[i] = b.B[i] = (h4 << 2) | r2; c.B[i] = h4 << 2;
        }
        for (int i = 0; i < 128; ++i) {                  // C = +-n * 2^-4, n in 0..63
          float v = ldexpf((float)rndu(64), -4); if (rnd() & 1) v = -v;
          a.C[i] = b.C[i] = c.C[i] = v;
        }
        p1.push_back(a); p2.push_back(b); p3.push_back(c);
      }
      std::vector<float> D1 = run_full(id1, p1), D2 = run_full(id3, p2), D3 = run_full(id1, p3);
      for (long j = 0; j < m; ++j) {
        bool x12 = false, x13 = false;
        for (int e = 0; e < 128; ++e) {
          x12 |= fb(D1[j * 128 + e]) != fb(D2[j * 128 + e]);
          x13 |= fb(D1[j * 128 + e]) != fb(D3[j * 128 + e]);
        }
        d12 += x12; d13 += x13;
      }
    }
    printf("  %-24s vs %-24s %s rounds = %ld  D1 != D2: %ld  D1 != D3: %ld\n", I1.name, I3.name,
           phase == 0 ? "[self-check]" : "[main]      ", n, d12, d13);
    if (phase == 0)
      ck.check(d12 == 0 && d13 == 0, "%-24s self-check (r2 = 0): D1, D2, D3 bit-identical in all %ld rounds", I1.name, n);
    else {
      ck.check(d12 == 0, "%-24s D1 == D2 in %ld/%ld rounds", I1.name, n - d12, n);
      ck.check(d13 > 0, "%-24s D1 != D3 in %ld/%ld rounds -> H2 (shared 6-bit decoder)", I1.name, d13, n);
    }
  }
}

int main(int argc, char** argv) {
  long rounds = argc > 1 ? atol(argv[1]) : 100000;
  print_header("test9: FP4 low-bit perturbation vs FP6 E2M3");
  Checker ck;
  run_path(ck, 7, 6, rounds);     // kind::f8f6f4
  run_path(ck, 12, 11, rounds);   // kind::mxf8f6f4 1X, scales 1.0
  return ck.finish("test9");
}
