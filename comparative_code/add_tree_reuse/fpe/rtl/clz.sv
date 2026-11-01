`include "defines.sv"

// ============================================================
// CLZ —— Count Leading Zeros
// 输入 LEN 位数据，输出前导零个数（0 ~ LEN）
//
// 参数要求：LEN 必须是 2 的幂次（4/8/16/32/64 ...）
// 输出宽度：$clog2(LEN+1) 位，可完整表示 0~LEN
//
// 实现方式：树形折半递归
//   - 将输入对半分为高半、低半
//   - 若高半全0，则 CLZ = LEN/2 + CLZ(低半)
//   - 否则        CLZ = CLZ(高半)
// ============================================================

module CLZ #(
    parameter LEN = 8   // 必须为 2 的幂次
)(
    input  wire [LEN-1:0]            in,
    output wire [$clog2(LEN+1)-1:0]  out
);

generate
    if (LEN == 2) begin : base_case
        // ---- 基础情况：2bit ----
        // 输出宽度 $clog2(3) = 2，可表示 0~2
        assign out = in[1] ? 2'd0 :   // 高位为1 → CLZ=0
                     in[0] ? 2'd1 :   // 高位为0，低位为1 → CLZ=1
                             2'd2;    // 全0 → CLZ=2
    end else begin : recursive_case
        // ---- 递归情况：LEN > 2 ----
        localparam HALF    = LEN / 2;
        localparam OUT_W   = $clog2(LEN  + 1);  // 本级输出宽度
        localparam HALF_OW = $clog2(HALF + 1);  // 子级输出宽度

        wire [HALF_OW-1:0] clz_hi, clz_lo;

        // 高半部分：in[LEN-1 : LEN/2]
        CLZ #(.LEN(HALF)) u_hi (
            .in  (in[LEN-1 : HALF]),
            .out (clz_hi)
        );

        // 低半部分：in[LEN/2-1 : 0]
        CLZ #(.LEN(HALF)) u_lo (
            .in  (in[HALF-1 : 0]),
            .out (clz_lo)
        );

        // 合并：高半全0时 CLZ = HALF + CLZ(低半)，否则 CLZ = CLZ(高半)
        // HALF 是2的幂次，clz_hi == HALF 等价于最高位为1，其余位全0
        assign out = (clz_hi == HALF[HALF_OW-1:0])
                     ? ({1'b0, clz_lo} + OUT_W'(HALF))   // 高半全0
                     :  {1'b0, clz_hi};                  // 高半不全0
    end
endgenerate

endmodule
