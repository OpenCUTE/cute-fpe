// test4: rounding of the FP32 output. x = c + a0*b0 is placed between (or on the midpoint of)
// two adjacent FP32 values around C0 = 2^23 and the output is matched against each rounding mode.
// Stage 2 (T1-T4, RN midpoint rules) only applies when stage 1 is RN; its outputs are printed for reference.
#include "../../common/harness.cuh"

int main() {
  print_header("test4: FP32 output rounding");
  Checker ck;
  const double C0 = 8388608.0;
  struct Case { const char* name; double c, a, b; };
  const Case cs[16] = {
    {"P1", 0, .5, 1.5}, {"P2", 0, .5, .5}, {"P3", 0, -.5, 1.5}, {"P4", 0, -.5, .5},
    {"P5", 0, 1, .5}, {"P6", 0, 1, 1.5}, {"P7", 0, 1, -.5}, {"P8", 0, 1, -1.5},
    {"S1", C0, .5, 1.5}, {"S2", C0, .5, .5}, {"S3", -C0, -.5, 1.5}, {"S4", -C0, -.5, .5},
    {"T1", C0, 1, .5}, {"T2", C0, 1, 1.5}, {"T3", -C0, 1, -.5}, {"T4", -C0, 1, -1.5}};
  const char* sN[5] = {"RU", "RD", "RZ", "RA", "RN"};
  const double sT[5][4] = {{C0+1, C0+1, -C0, -C0}, {C0, C0, -C0-1, -C0-1}, {C0, C0, -C0, -C0},
                           {C0+1, C0+1, -C0-1, -C0-1}, {C0+1, C0, -C0-1, -C0}};
  for (int id = 0; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id];
    std::vector<Prob> ps;
    for (const Case& c : cs) { Prob p = mk(I); p.A[0] = slot(I, c.a); p.B[0] = slot(I, c.b); p.C[0] = (float)c.c; ps.push_back(p); }
    std::vector<float> D = run_full(id, ps);
    bool pok = true;
    for (int i = 0; i < 8; ++i) pok &= (double)D[i * 128] == cs[i].a * cs[i].b;
    const char* s = "UNCLASSIFIED";
    for (int r = 0; r < 5; ++r) { bool m = true; for (int i = 0; i < 4; ++i) m &= (double)D[(8 + i) * 128] == sT[r][i]; if (m) s = sN[r]; }
    printf("  %-24s S1-S4 = [%.0f %.0f %.0f %.0f] -> %-4s  T1-T4 = [%.0f %.0f %.0f %.0f] (reference only)\n", I.name,
           D[8*128], D[9*128], D[10*128], D[11*128], s, D[12*128], D[13*128], D[14*128], D[15*128]);
    ck.check(pok, "%-24s controls P1-P8 return the exact product", I.name);
    ck.check(std::string(s) == "RZ", "%-24s stage 1 classifies as %s (expected RZ)", I.name, s);
  }
  return ck.finish("test4");
}
