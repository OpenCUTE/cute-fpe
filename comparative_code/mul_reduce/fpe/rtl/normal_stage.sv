`include "defines.sv"

module normal_stage (
    input wire signed [9:0]  E_MAX_i,
    input wire signed [9:0]  block_max_i,

    // input wire  [1:0]  data_type_i,
    input wire         is_MXINTFP4_i,

    input wire  signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0] sign_fp8_i,
    input wire signed [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig_i,

    // fp32_deal 输出：归一化后的 fp32 各字段
    output wire        fp32_sign,
    output wire [23:0] fp32_sig,
    output wire signed [9:0]  fp32_exp
);


// -------------------------------------------------------
// 例化 fp32_deal
// 根据数据类型、指数信息，将定点结果转换为 fp32 各字段
// -------------------------------------------------------
fp32_deal u_fp32_deal (
    .sign_fp8_i    (sign_fp8_i),
    .nvmxfp4_sig   (nvmxfp4_sig_i),

    .is_MXINTFP4_i (is_MXINTFP4_i),

    .E_MAX_i       (E_MAX_i),
    .block_max_i   (block_max_i),

    .fp32_sign     (fp32_sign),
    .fp32_sig      (fp32_sig),
    .fp32_exp      (fp32_exp)
);
endmodule
