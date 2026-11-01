// test6: order of scale application on NVFP4 (kind::mxf4nvf4 scale_vec::4X).
// M1: group dot product times scale_A*scale_B exactly, then alignment/truncation.
// M2: alignment/truncation by the scale exponent first, scale mantissa multiplied afterwards.
#include "../../common/harness.cuh"

static const int ID = 15;

// Group 1: A = B = 4, scales 2^8 -> +2^24, cancelled by C = -2^24 (when big).
// Group 4: first n elements A = B = 1, scale_A3 = sa3, scale_B3 = sb3.
static Prob build(bool big, int n, double sa3, double sb3) {
  const Instr& I = INS[ID];
  Prob p = mk(I);
  if (big) { for (int k = 0; k < 16; ++k) { p.A[k] = slot(I, 4.0); p.B[k * 8] = slot(I, 4.0); } p.C[0] = -ldexpf(1.0f, 24); }
  for (int k = 48; k < 48 + n; ++k) { p.A[k] = slot(I, 1.0); p.B[k * 8] = slot(I, 1.0); }
  p.sfa = sreg(I, 256, 1, 1, sa3); p.sfb = sreg(I, 256, 1, 1, sb3);
  return p;
}

int main() {
  print_header("test6: scale application order (NVFP4)");
  Checker ck;
  const double L = ldexp(1.0, -11);   // LSB after aligning to 2^24 with 35 fractional bits

  Prob m = build(true, 2, 1.5 * ldexp(1.0, -6), ldexp(1.0, -6));
  ck.check(m.sfa == 0x0C383878u && m.sfb == 0x08383878u, "scale registers 0x%08X / 0x%08X (documented 0x0C383878 / 0x08383878)", m.sfa, m.sfb);
  float ref = run1(ID, build(false, 2, 1.5 * ldexp(1.0, -6), ldexp(1.0, -6)));
  float d = run1(ID, m);
  ck.check(ref == (float)(1.5 * L), "reference (group 4 only) = %s (expected 1.5 * 2^-11)", hexf(ref).c_str());
  ck.check(d == (float)L, "main experiment D = %s (M1 predicts 2^-11, M2 predicts 1.5 * 2^-11)", hexf(d).c_str());

  printf("\n-- scan: n in {1,2}, scale_A3 = m * 2^ea (m in {1,1.25,1.5,1.75}, ea in -4..-7), scale_B3 = 2^-6\n");
  printf("  %-3s %-12s %-14s %-14s %-14s %-14s\n", "n", "scale_A3", "reference", "M1", "M2", "measured");
  const double mant[4] = {1, 1.25, 1.5, 1.75};
  int m1 = 0, m2 = 0, diff = 0, diffM1 = 0, refok = 0, tot = 0;
  for (int ea = -4; ea >= -7; --ea) for (double mt : mant) for (int n : {1, 2}) {
    double sa3 = mt * ldexp(1.0, ea), sb3 = ldexp(1.0, -6);
    float r = run1(ID, build(false, n, sa3, sb3));
    float D = run1(ID, build(true, n, sa3, sb3));
    double exact = n * sa3 * sb3;
    double p1 = std::floor(exact / L) * L;
    double p2 = std::floor(n * ldexp(1.0, ea) * sb3 / L) * L * mt;
    refok += (double)r == exact; m1 += (double)D == p1; m2 += (double)D == p2;
    if (p1 != p2) { ++diff; diffM1 += (double)D == p1; }
    ++tot;
    printf("  %-3d %-12s %-14s %-14s %-14s %-14s%s\n", n, hexf((float)sa3).c_str(), hexf(r).c_str(), hexf((float)p1).c_str(),
           hexf((float)p2).c_str(), hexf(D).c_str(), p1 != p2 ? "  <- distinguishing" : "");
  }
  ck.check(refok == tot, "scan references exact in %d/%d cases", refok, tot);
  ck.check(m1 == tot, "M1 matches %d/%d cases", m1, tot);
  ck.check(diff > 0 && diffM1 == diff, "M1 wins all %d distinguishing cases (M2 matches %d/%d overall)", diff, m2, tot);
  return ck.finish("test6");
}
