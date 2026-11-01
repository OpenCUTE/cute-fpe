`include "defines.sv"

module add1_stage (
    input  wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0] sign_after_add0 [0:`P4_ADD0_OUT_NUM-1],
    input  wire         [7:0]                     scale_sig_i      [0:`Scale_NUM - 1],
    input  wire         [5:0]                     block_shift_i    [0:`Scale_NUM - 1],
    input  wire         is_mxint4,

    output wire signed  [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0]   sign_after_add1  [0:`P4_ADD1_OUT_NUM-1],
    output wire signed [`decimal_places1+ `block_int_w-1:0] block_sig_af_shift [0:`P4_ADD0_OUT_NUM-1]
);

// -------------------------------------------------------
// 中间信号
// -------------------------------------------------------
// p4_add0 的输出：共 P4_ADD0_OUT_NUM 路，每路位宽 28+clog2(ADD_NUM0)+clog2(ADD_NUM1)
// wire signed  [`decimal_places1 + `block_int_w -1:0]                       FP4_block_sig [0:`P4_ADD0_OUT_NUM-1];

// -------------------------------------------------------
// 重复 LANE_NUM/ADD_NUM1/ADD_NUM2 次：fp8_add1 例化
// 每个实例消耗 ADD_NUM2 路 sign_after_add0，产生 1 路输出
// -------------------------------------------------------
genvar i;
generate
    for (i = 0; i < `P4_ADD1_OUT_NUM; i++) begin : gen_fp8_add1
        fp8_add1 u_fp8_add1 (
            .sign_after_add0  (sign_after_add0[i*`ADD_NUM2 : i*`ADD_NUM2+`ADD_NUM2-1]),
            .sign_after_add1  (sign_after_add1[i])
        );
    end
endgenerate

// -------------------------------------------------------
// scale_black_sig_mul：FP4 尾数乘法 + 移位（单次例化，处理全部通道）
// -------------------------------------------------------

wire signed [12:0] sign_after_add0_low [0:`P4_ADD0_OUT_NUM-1];

genvar j;
generate
    for (j = 0; j < `P4_ADD0_OUT_NUM; j = j + 1) begin : GEN_LOW13
        assign sign_after_add0_low[j] = sign_after_add0[j][12:0];
    end
endgenerate

block_mul_shift u_block_mul_shift (
    .sign_after_add0(sign_after_add0_low),//对于16个元素一组而言，有效数据是1bit符号位+12bit数据
    .scale_sig_i    (scale_sig_i    ),
    .is_mxint4      (is_mxint4      ),

    .block_shift_i  (block_shift_i  ),

    .block_sig_af_shift (block_sig_af_shift)
);

// scale_black_sig_mul u_scale_black_sig_mul (
//     .sign_after_add0  (sign_after_add0),
//     .scale_sig_i      (scale_sig_i  ),
//     .is_mxint4        (is_mxint4    ),

//     .FP4_block_sig    (FP4_block_sig)
// );

// block_shift u_block_shift (
//     .FP4_block_sig (FP4_block_sig),
//     .block_shift_i (block_shift_i),
    
//     .block_sig_af_shift (block_sig_af_shift)
// );

endmodule
