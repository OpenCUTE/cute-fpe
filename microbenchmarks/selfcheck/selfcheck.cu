// Self-check of the harness: fragment layout, slot placement and scale-byte mapping.
// Random exactly-representable matrices are run on every instruction and compared with an exact
// double-precision reference. Every later test assumes this check passes.
#include "../common/harness.cuh"

int main() {
  print_header("selfcheck: fragment layout and scale mapping");
  Checker ck;
  const double vals[] = {0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2};
  const double svals[] = {0.5, 1, 2};
  const int NP = 300;
  for (int id = 0; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id]; int K = I.K;
    std::vector<Prob> ps; std::vector<double> ref(NP * 128);
    for (int n = 0; n < NP; ++n) {
      Prob p = mk(I);
      double av[16][64], bv[64][8], sA[4] = {1, 1, 1, 1}, sB[4] = {1, 1, 1, 1};
      if (I.sc != S_NONE) for (int j = 0; j < 4; ++j) { sA[j] = svals[rndu(3)]; sB[j] = svals[rndu(3)]; }
      p.sfa = sreg(I, sA[0], sA[1], sA[2], sA[3]);
      p.sfb = sreg(I, sB[0], sB[1], sB[2], sB[3]);
      for (int m = 0; m < 16; ++m) for (int k = 0; k < K; ++k) { av[m][k] = vals[rndu(9)]; p.A[m * K + k] = slot(I, av[m][k]); }
      for (int k = 0; k < K; ++k) for (int c = 0; c < 8; ++c) { bv[k][c] = vals[rndu(9)]; p.B[k * 8 + c] = slot(I, bv[k][c]); }
      for (int m = 0; m < 16; ++m) for (int c = 0; c < 8; ++c) {
        double cv = (double)((int)rndu(17) - 8), s = cv;
        p.C[m * 8 + c] = (float)cv;
        for (int k = 0; k < K; ++k) s += av[m][k] * bv[k][c] * sA[group(I, k)] * sB[group(I, k)];
        ref[n * 128 + m * 8 + c] = s;
      }
      ps.push_back(p);
    }
    std::vector<float> D = run_full(id, ps);
    int bad = 0;
    for (int i = 0; i < NP * 128; ++i) bad += (double)D[i] != ref[i];
    ck.check(bad == 0, "%-24s %d/%d elements match the exact reference", I.name, NP * 128 - bad, NP * 128);
  }
  return ck.finish("selfcheck");
}
