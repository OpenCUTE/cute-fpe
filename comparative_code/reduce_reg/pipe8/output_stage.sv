`include "defines.sv"

module output_stage (
    input wire NAN_i, NINF_i, PINF_i,
    // input wire [1 :0] data_type_i,
    input wire         is_INT_i,
    input wire [31:0] INT32_i,

    input wire fp32_sign,
    input wire [23:0]  fp32_sig,
    input wire signed [9 :0]  fp32_exp,

    output wire [31:0] D_bits
);

wire [31:0] fp32_temp;
//判断这个是否是0，如果是0，则只输出+0，不输出-0
wire fp32_is_nz;
assign fp32_is_nz = |fp32_temp[30:0];

assign D_bits = (is_INT_i) ? INT32_i    : 
                fp32_is_nz ? fp32_temp  : 32'd0;

// -------------------------------------------------------
// 例化 p6_except_deal
// 处理 NAN / PINF / NINF 特殊值，输出最终 fp32
// -------------------------------------------------------
p8_except_deal u_p8_except_deal (
    .NAN_i         (NAN_i),
    .PINF_i        (PINF_i),
    .NINF_i        (NINF_i),

    .fp32_sign     (fp32_sign),
    .fp32_sig      (fp32_sig),
    .fp32_exp      (fp32_exp),

    .fp32          (fp32_temp)
);


endmodule
