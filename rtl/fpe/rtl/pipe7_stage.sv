`include "defines.sv"

module pipe7_stage (
    input wire clock,
    input wire reset,

    // 输出：add2_stage 结果
    input wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0] sign_fp8_i,
    input wire signed [`decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig_i,

    input wire signed [9:0] E_MAX_i,
    input wire signed [9:0] block_max_i,
    // input wire [1:0]   data_type_i,
    input wire         is_MXINTFP4_i,
    input wire         is_INT_i,

    // 打一拍输出
    input wire         NAN_i,
    input wire         PINF_i,
    input wire         NINF_i,

    // 握手信号
    // pipe6 -- pipe7
    input  wire valid6_i,
    output wire allow7_in_o,
    // pipe7 -- out
    output reg  valid7_o,
    input  wire allow8_in_i,

    // 打一拍后输出
    output reg         NAN_o,
    output reg         PINF_o,
    output reg         NINF_o,
    // output reg [1:0]   data_type_o,
    output reg         is_INT_o,

    output reg [31:0]  INT32_o,

    output reg         fp32_sign_o,
    output reg  [23:0] fp32_sig_o,
    output reg  [9:0]  fp32_exp_o      // signed，用 reg 存储后以有符号方式使用
);

// -------------------------------------------------------
// 握手逻辑
// -------------------------------------------------------
assign allow7_in_o = (!valid7_o) || allow8_in_i;

always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        valid7_o <= 1'b0;
    end else if (allow7_in_o) begin
        valid7_o <= valid6_i;
    end
    // else: 保持不变（valid7_o <= valid7_o 可省略）
end

// -------------------------------------------------------
// 中间信号：normal_stage 组合逻辑输出
// -------------------------------------------------------
wire        fp32_sign_w;
wire [23:0] fp32_sig_w;
wire signed [9:0]  fp32_exp_w;

normal_stage u_normal_stage (
    .E_MAX_i       (E_MAX_i),
    .block_max_i   (block_max_i),
    .is_MXINTFP4_i (is_MXINTFP4_i),
    .sign_fp8_i    (sign_fp8_i),
    .nvmxfp4_sig_i (nvmxfp4_sig_i),
    .fp32_sign     (fp32_sign_w),
    .fp32_sig      (fp32_sig_w),
    .fp32_exp      (fp32_exp_w)
);

// -------------------------------------------------------
// 打一拍寄存输出
// -------------------------------------------------------
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        NAN_o       <= 1'b0;
        PINF_o      <= 1'b0;
        NINF_o      <= 1'b0;
        // data_type_o <=  '0;
        is_INT_o       <= 1'b0;
        INT32_o     <=  '0;

        fp32_sign_o <= 1'b0;
        fp32_sig_o  <= 24'd0;
        fp32_exp_o  <= 10'd0;
    end else if (allow7_in_o) begin
        NAN_o       <= NAN_i;
        PINF_o      <= PINF_i;
        NINF_o      <= NINF_i;
        // data_type_o <= data_type_i;
        is_INT_o       <= is_INT_i;
        INT32_o     <= sign_fp8_i[31:0];

        fp32_sign_o <= fp32_sign_w;
        fp32_sig_o  <= fp32_sig_w;
        fp32_exp_o  <= fp32_exp_w;
    end
end

endmodule
