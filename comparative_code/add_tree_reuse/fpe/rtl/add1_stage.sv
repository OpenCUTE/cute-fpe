`include "defines.sv"

module add1_stage (
    input  wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0] sign_after_add0 [0:`P4_ADD0_OUT_NUM-1],
    input  wire         [7:0]                     scale_sig_i      [0:`Scale_NUM - 1],
    input  wire         [5:0]                     block_shift_i    [0:`Scale_NUM - 1],
    input  wire signed [`BLOCK_SUM_W-1:0] block_sum_i [0:`Scale_NUM-1],
    input  wire         is_mxint4,

    output wire signed  [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0]   sign_after_add1  [0:`P4_ADD1_OUT_NUM-1],
    output wire signed [`decimal_places1+ `block_int_w-1:0] block_sig_af_shift [0:`P4_ADD0_OUT_NUM-1]
);

// 浮点/INT8 沿用原有归约路径。
genvar i;
generate
    for (i = 0; i < `P4_ADD1_OUT_NUM; i++) begin : gen_fp8_add1
        fp8_add1 u_fp8_add1 (
            .sign_after_add0  (sign_after_add0[i*`ADD_NUM2 : i*`ADD_NUM2+`ADD_NUM2-1]),
            .sign_after_add1  (sign_after_add1[i])
        );
    end
endgenerate

// NVFP4/MXFP4/MXINT4 使用 pipe2 的独立 block_sum，保持 scale 分组顺序。
block_mul_shift u_block_mul_shift (
    .sign_after_add0    (block_sum_i),
    .scale_sig_i        (scale_sig_i),
    .is_mxint4          (is_mxint4),
    .block_shift_i      (block_shift_i),
    .block_sig_af_shift (block_sig_af_shift)
);

endmodule
