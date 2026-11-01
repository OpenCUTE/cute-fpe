`include "defines.sv"

module pipe8_stage (
    input wire clock,
    input wire reset,

    // 来自 pipe7 的输入
    input wire         NAN_i,
    input wire         PINF_i,
    input wire         NINF_i,
    // input wire [1:0]   data_type_i,
    input wire         is_INT_i,
    input wire [31:0]  INT32_i,

    input wire         fp32_sign_i,
    input wire  [23:0] fp32_sig_i,
    input wire  signed [9:0] fp32_exp_i,

    // 握手信号
    // pipe7 -- pipe8
    input  wire valid7_i,
    output wire allow8_in_o,
    // pipe8 -- out
    output reg  valid8_o,
    input  wire allow9_in_i,

    // 打一拍后输出
    output reg  [31:0] D_bits_o
);

// -------------------------------------------------------
// 握手逻辑
// -------------------------------------------------------
assign allow8_in_o = (!valid8_o) || allow9_in_i;

always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        valid8_o <= 1'b0;
    end else if (allow8_in_o) begin
        valid8_o <= valid7_i;
    end
end

// -------------------------------------------------------
// 中间信号：output_stage 组合逻辑输出
// -------------------------------------------------------
wire [31:0] D_bits_w;

output_stage u_output_stage (
    .NAN_i      (NAN_i),
    .NINF_i     (NINF_i),
    .PINF_i     (PINF_i),
    .is_INT_i   (is_INT_i),
    .INT32_i    (INT32_i),

    .fp32_sign  (fp32_sign_i),
    .fp32_sig   (fp32_sig_i),
    .fp32_exp   (fp32_exp_i),
    .D_bits     (D_bits_w)
);

// -------------------------------------------------------
// 打一拍寄存输出
// -------------------------------------------------------
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        D_bits_o <= 32'd0;
    end else if (allow8_in_o) begin
        D_bits_o <= D_bits_w;
    end
end

endmodule
