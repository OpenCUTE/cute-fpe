// test7: lower bound of the alignment exponent (-133 on the common path, -139 on the FP4 path).
// A large product +2^m and a small product -2^e (C = +0). The first e at which the small product is
// truncated gives emax_eff = e_drop + p + 1.
#include <climits>
#include "../../common/harness.cuh"

// Puts sign * 2^e at position k. TF32/BF16: a = sign*2^ea, b = 2^eb (normal values).
// UE8M0 instructions: a = sign, b = 1, and the group scale product becomes 2^e.
static void put(const Instr& I, Prob& p, int k, int e, double sign, double sc[4]) {
  if (I.sc == S_NONE) {
    int ea = e / 2, eb = e - ea;
    p.A[k] = slot(I, sign * ldexp(1.0, ea));
    p.B[k * 8] = slot(I, ldexp(1.0, eb));
  } else {
    p.A[k] = slot(I, sign);
    p.B[k * 8] = slot(I, 1.0);
    sc[group(I, k)] = ldexp(1.0, e);
  }
}
// scale_A * scale_B = sc[j]: exponent split as trunc(t/2) and t - trunc(t/2).
static void set_scales(const Instr& I, Prob& p, double sc[4]) {
  if (I.sc == S_NONE) return;
  double sa[4], sb[4];
  for (int j = 0; j < 4; ++j) {
    int e = (int)std::lround(std::log2(sc[j])), ea = e / 2, eb = e - ea;
    sa[j] = ldexp(1.0, ea); sb[j] = ldexp(1.0, eb);
  }
  p.sfa = sreg(I, sa[0], sa[1], sa[2], sa[3]);
  p.sfb = sreg(I, sb[0], sb[1], sb[2], sb[3]);
}

// Scans the problems built by build(e) for e = e_hi .. e_lo against build_ref.
// Returns the first truncated e (or INT_MIN) and whether truncation is monotonic below it.
template <class F>
static int scan(int id, F build, const Prob& ref_prob, int e_hi, int e_lo, bool& mono) {
  std::vector<Prob> ps{ref_prob};
  for (int e = e_hi; e >= e_lo; --e) ps.push_back(build(e));
  std::vector<float> D = run_full(id, ps);
  int first = INT_MIN; mono = true;
  for (int i = 1; i < (int)ps.size(); ++i) {
    int e = e_hi - (i - 1);
    bool dropped = D[i * 128] == D[0];
    if (dropped && first == INT_MIN) first = e;
    if (!dropped && first != INT_MIN) mono = false;
  }
  return first;
}

int main() {
  print_header("test7: alignment exponent floor");
  Checker ck;
  const int ms[] = {-130, -136, -140, -145, -148};

  printf("-- largest term is a product +2^m; small product -2^e in another scale group (k = 1 without scale)\n");
  for (int id : {0, 2, 13, 14}) {
    const Instr& I = INS[id]; int p = precision_bits(id), floor_ = id >= 13 ? -139 : -133;
    int ks = I.vec >= 2 ? I.K / I.vec : 1;
    for (int m : ms) {
      auto build = [&](int e) {
        Prob q = mk(I); double sc[4] = {1, 1, 1, 1};
        put(I, q, 0, m, +1.0, sc);
        if (e != INT_MIN) put(I, q, ks, e, -1.0, sc);
        set_scales(I, q, sc);
        return q;
      };
      bool mono; int ed = scan(id, build, build(INT_MIN), m - 1, -230, mono);
      int eff = ed + p + 1, expect = std::max(m, floor_);
      ck.check(ed != INT_MIN && mono && eff == expect, "%-24s m = %4d: e_drop = %4d, emax_eff = %4d (expected %4d)", I.name, m, ed, eff, expect);
    }
  }

  printf("\n-- 1X (one shared scale S = 2^s, s = m - 2u): large product 2^u * 2^u * S, small -(2^x)(2^y) * S\n");
  struct F { int id, u, emin; };
  for (F f : {F{8, 15, -14}, F{9, 8, -6}, F{10, 4, -2}}) {
    const Instr& I = INS[f.id];
    for (int m : ms) {
      int s = m - 2 * f.u, e_lo = s + 2 * f.emin;
      auto build = [&](int e) {
        Prob q = mk(I);
        q.A[0] = slot(I, ldexp(1.0, f.u)); q.B[0] = slot(I, ldexp(1.0, f.u));
        if (e != INT_MIN) {
          int xy = e - s, x = xy / 2, y = xy - x;      // xy >= 2*emin, so x, y >= emin
          q.A[1] = slot(I, -ldexp(1.0, x)); q.B[8] = slot(I, ldexp(1.0, y));
        }
        int sa = s / 2, sb = s - sa;
        q.sfa = sreg(I, ldexp(1.0, sa)); q.sfb = sreg(I, ldexp(1.0, sb));
        return q;
      };
      bool mono; int ed = scan(f.id, build, build(INT_MIN), m - 1, e_lo, mono);
      int expect = std::max(m, -133);
      if (ed == INT_MIN) { printf("  [ -- ] %-24s m = %4d: not measurable (e only reaches %d)\n", I.name, m, e_lo); continue; }
      ck.check(mono && ed + 26 == expect, "%-24s m = %4d: e_drop = %4d, emax_eff = %4d (expected %4d)", I.name, m, ed, ed + 26, expect);
    }
  }

  printf("\n-- side result: C = 2^c is the largest term, small product -2^e at k = 0\n");
  const int cs[] = {-148, -140, -136, -134, -133, -132, -130, -126, -120};
  for (int id : {0, 2, 8, 9, 10, 11, 12, 13, 14}) {
    const Instr& I = INS[id]; int p = precision_bits(id);
    std::string row;
    bool ok = true;
    for (int c : cs) {
      auto build = [&](int e) {
        Prob q = mk(I); double sc[4] = {1, 1, 1, 1};
        if (e != INT_MIN) put(I, q, 0, e, -1.0, sc);
        set_scales(I, q, sc);
        q.C[0] = ldexpf(1.0f, c);
        return q;
      };
      bool mono; int ed = scan(id, build, build(INT_MIN), c - 1, -230, mono);
      int eff = ed + p + 1;
      ok &= mono && eff == std::max(c, -126);
      row += " " + std::to_string(eff);
    }
    ck.check(ok, "%-24s emax_eff for c = -148..-120:%s (expected max(c, -126))", I.name, row.c_str());
  }
  return ck.finish("test7");
}
