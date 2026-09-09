`include "defines.sv"

module pipe6_stage (
    input  wire clock,
    input  wire reset,

    // 穿透，打一拍输出到下一级
    input  wire signed [9:0]  E_MAX_i,
    input  wire signed [9:0]  block_max_i,

    // 不穿透，本级使用
    input  wire [5:0]  nvc_shift_i,
    input  wire [4:0]  c_shift_i,
    input  wire [31:0] C_sig_i,
    input  wire        C_sign_i,

    // 穿透，打一拍输出到下一级
    input  wire        NAN_i,
    input  wire        PINF_i,
    input  wire        NINF_i,
    // input  wire [1:0]  data_type_i,
    input  wire        is_MXINTFP4_i,
    input  wire        is_INT_i,

    // 不穿透，本级使用
    input  wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0]
                       sign_after_add1_i    [0:`P4_ADD1_OUT_NUM-1],
    input  wire signed [`decimal_places1+`block_int_w-1:0]
                       block_sig_af_shift_i [0:`P4_ADD0_OUT_NUM-1],

    //握手信号
    //pipe5 -- pipe6
    input  wire valid5_i,
    output wire allow6_in_o,
    //pipe6 -- out
    output reg  valid6_o,
    input  wire allow7_in_i,

    // =========================================================
    // 输出：add2_stage 结果，打一拍
    // =========================================================
    output reg [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0]
                       sign_fp8_o,
    output reg [`decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM):0]
                       nvmxfp4_sig_o,

    // =========================================================
    // 穿透信号，打一拍输出
    // =========================================================
    output reg signed [9:0] E_MAX_o,
    output reg signed [9:0] block_max_o,
    output reg         NAN_o,
    output reg         PINF_o,
    output reg         NINF_o,
    // output reg [1:0]   data_type_o
    output reg         is_MXINTFP4_o,
    output reg         is_INT_o
);

//握手逻辑
assign allow6_in_o = (!valid6_o) || allow7_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid6_o    <= 1'd0;
    end
    else if(allow6_in_o) begin
        valid6_o    <= valid5_i;
    end
    else begin
        valid6_o    <= valid6_o;
    end
end

// =========================================================================
// localparam：统一管理位宽
// =========================================================================
localparam FP8_W    = 28 + $clog2(`ADD_NUM0) + $clog2(`ADD_NUM1) + $clog2(`ADD_NUM2) + 1;
localparam NV_SIG_W = `decimal_places1 + `block_int_w + $clog2(`P4_ADD0_OUT_NUM) + 1;

// =========================================================================
// 内部连线：接收 add2_stage 组合逻辑输出
// =========================================================================
wire [FP8_W-1:0]    sign_fp8_w;
wire [NV_SIG_W-1:0] nvmxfp4_sig_w;

// =========================================================================
// 例化 add2_stage
// =========================================================================
add2_stage u_add2_stage (
    .nvc_shift_i          (nvc_shift_i),
    .c_shift_i            (c_shift_i),
    .C_sig_i              (C_sig_i),
    .C_sign_i             (C_sign_i),
    .is_INT_i             (is_INT_i),
    .sign_after_add1_i    (sign_after_add1_i),
    .block_sig_af_shift_i (block_sig_af_shift_i),
    .sign_fp8             (sign_fp8_w),
    .nvmxfp4_sig          (nvmxfp4_sig_w)
);

// =========================================================================
// always_ff：所有输出打一拍
// =========================================================================
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        // add2_stage 结果清零
        sign_fp8_o      <= '0;
        nvmxfp4_sig_o   <= '0;
        // 穿透信号清零
        E_MAX_o         <= '0;
        block_max_o     <= '0;
        NAN_o           <= 1'b0;
        PINF_o          <= 1'b0;
        NINF_o          <= 1'b0;
        // data_type_o     <= 2'b0;
        is_MXINTFP4_o       <= 1'b0;
        is_INT_o            <= 1'b0;
    end else if(allow6_in_o) begin
        // --- add2_stage 结果寄存 ---
        sign_fp8_o      <= sign_fp8_w;
        nvmxfp4_sig_o   <= nvmxfp4_sig_w;
        // --- 穿透信号寄存 ---
        E_MAX_o         <= E_MAX_i;
        block_max_o     <= block_max_i;
        NAN_o           <= NAN_i;
        PINF_o          <= PINF_i;
        NINF_o          <= NINF_i;
        // data_type_o     <= data_type_i;
        is_MXINTFP4_o       <= is_MXINTFP4_i;
        is_INT_o            <= is_INT_i;
    end
end

endmodule
