// test1: broadcast one A row, one B column, one C and one scale vector to the whole tile and check
// that all 128 D elements are bit-identical to D[0][0].
// Usage: ./test1 [rounds per distribution, default 1000000]
#include "../../common/harness.cuh"

// Distribution 1 (mode 0): uniformly random raw bits. Distribution 2 (mode 1): finite, moderate magnitude.
static uint32_t gen_elem(const Instr& I, int mode) {
  int w = fbits(I.fmt); uint32_t mask = w == 32 ? 0xFFFFFFFFu : ((1u << w) - 1);
  if (mode == 0) return ((uint32_t)rnd() & mask) << I.shift;
  if (I.fmt == F_TF32 || I.fmt == F_BF16) {
    uint32_t s = rnd() & 1, e = 127 - 20 + rndu(41), m = (uint32_t)rnd() & 0x7FFFFF;
    uint32_t u = (s << 31) | (e << 23) | m;
    return I.fmt == F_TF32 ? u : (u >> 16);
  }
  uint32_t r;
  do { r = (uint32_t)rnd() & mask; } while (!std::isfinite(decode(I.fmt, r)));
  return r << I.shift;
}
static uint32_t gen_c(int mode) {
  if (mode == 0) return (uint32_t)rnd();
  uint32_t s = rnd() & 1, e = 127 - 8 + rndu(17), m = (uint32_t)rnd() & 0x7FFFFF;
  return (s << 31) | (e << 23) | m;
}
static uint32_t gen_scale(const Instr& I, int mode) {
  if (I.sc == S_NONE) return 0;
  if (mode == 0) return I.sc == S_UE8M0 ? (uint32_t)rnd() : ((uint32_t)rnd() & 0x7F7F7F7Fu);
  uint32_t v = 0;
  for (int j = 0; j < 4; ++j) {
    uint32_t b = I.sc == S_UE8M0 ? 124 + rndu(7) : (((4 + rndu(7)) << 3) | rndu(8));
    v |= b << (8 * j);
  }
  return v;
}

int main(int argc, char** argv) {
  long rounds = argc > 1 ? atol(argv[1]) : 1000000;
  print_header("test1: output uniformity");
  printf("rounds per distribution: %ld (distribution 1 = raw random bits, distribution 2 = finite moderate values)\n\n", rounds);
  Checker ck;
  const size_t NB = 1 << 18;
  uint32_t *dAR, *dBC, *dCv, *dSA, *dSB, *dD00; int* dM;
  CK(cudaMalloc(&dAR, NB * 64 * 4)); CK(cudaMalloc(&dBC, NB * 64 * 4));
  CK(cudaMalloc(&dCv, NB * 4)); CK(cudaMalloc(&dSA, NB * 4)); CK(cudaMalloc(&dSB, NB * 4));
  CK(cudaMalloc(&dD00, NB * 4)); CK(cudaMalloc(&dM, NB * 4));
  std::vector<uint32_t> AR(NB * 64), BC(NB * 64), Cv(NB), SA(NB), SB(NB), D00(NB);
  std::vector<int> Mi(NB);
  printf("  %-24s %-6s %-12s %-12s %-12s\n", "instruction", "dist", "bad rounds", "D00 != C", "D00 is NaN");
  for (int id = 0; id < NUM_INSTR; ++id) {
    const Instr& I = INS[id]; int K = I.K;
    // write-back check: A = B = 0, C[0][1] = 1, all other C = 0
    Prob p = mk(I); p.C[1] = 1.0f;
    std::vector<float> D = run_full(id, {p});
    ck.check(D[1] == 1.0f && D[0] == 0.0f, "%-24s write-back check: D[0][1] = %g, D[0][0] = %g", I.name, D[1], D[0]);
    for (int mode = 0; mode < 2; ++mode) {
      long bad = 0, vis = 0, nan = 0;
      for (long done = 0; done < rounds; done += NB) {
        size_t m = (size_t)std::min<long>(NB, rounds - done);
        for (size_t j = 0; j < m; ++j) {
          for (int k = 0; k < K; ++k) { AR[j * K + k] = gen_elem(I, mode); BC[j * K + k] = gen_elem(I, mode); }
          Cv[j] = gen_c(mode); SA[j] = gen_scale(I, mode); SB[j] = gen_scale(I, mode);
        }
        CK(cudaMemcpy(dAR, AR.data(), m * K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBC, BC.data(), m * K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dCv, Cv.data(), m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dSA, SA.data(), m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dSB, SB.data(), m * 4, cudaMemcpyHostToDevice));
        k_bcast<<<(unsigned)m, 32>>>(id, K, I.epr, dAR, dBC, dCv, dSA, dSB, dM, dD00);
        CK(cudaGetLastError());
        CK(cudaMemcpy(Mi.data(), dM, m * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(D00.data(), dD00, m * 4, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < m; ++j) { bad += Mi[j] != 0; vis += D00[j] != Cv[j]; nan += std::isnan(bf(D00[j])); }
      }
      printf("  %-24s %-6d %-12ld %-12ld %-12ld\n", I.name, mode + 1, bad, vis, nan);
      ck.check(bad == 0, "%-24s distribution %d: all 128 D elements equal D[0][0] in %ld/%ld rounds", I.name, mode + 1, rounds - bad, rounds);
    }
  }
  return ck.finish("test1");
}
