`include "defines.sv"

module pipe5_stage (
    input wire clock,
    input wire reset,

    // =========================================================
    // 输入：add0_stage 结果
    // =========================================================
    input wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0]
                      sign_after_add0_i [0:`P4_ADD0_OUT_NUM-1],

    //不穿透，本级使用
    input wire [5:0]  block_shift_i [0:`Scale_NUM - 1],
    input wire [7:0]  scale_sig_i   [0:`Scale_NUM - 1],

    //穿透，打一拍到下一级
    input wire signed [9:0] block_max_i,
    input wire [5:0]  nvc_shift_i,
    input wire signed [9:0] E_MAX_i,
    input wire [4:0]  c_shift_i,

    //穿透，打一拍到下一级
    input wire        NAN_i,
    input wire        PINF_i,
    input wire        NINF_i,

    //穿透，打一拍到下一级
    input wire [31:0] C_sig_i,
    input wire        C_sign_i,
    // input wire [1:0]  data_type_i,
    input wire        is_MXNVFP4_i,
    input wire        is_INT8_i,
    input wire        is_INT4_i,
    input wire        is_MXINT4_i,

    //握手信号
    //pipe4 -- pipe5
    input  wire valid4_i,
    output wire allow5_in_o,
    //pipe5 -- pipe6
    output reg  valid5_o,
    input  wire allow6_in_i,

    // =========================================================
    // 输出：add1_stage 结果，打一拍
    // =========================================================
    output reg signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0]
                      sign_after_add1_o  [0:`P4_ADD1_OUT_NUM-1],
    output reg signed [`decimal_places1+`block_int_w-1:0]
                      block_sig_af_shift_o [0:`P4_ADD0_OUT_NUM-1],

    // =========================================================
    // 穿透信号，打一拍输出
    // =========================================================
    output reg signed [9:0] block_max_o,
    output reg [5:0]  nvc_shift_o,
    output reg signed [9:0] E_MAX_o,
    output reg [4:0]  c_shift_o,
    output reg        NAN_o,
    output reg        PINF_o,
    output reg        NINF_o,
    output reg [31:0] C_sig_o,
    output reg        C_sign_o,
    // output reg [1:0]  data_type_o
    output reg        is_MXINTFP4_o,
    output reg        is_INT_o
);

//握手逻辑
assign allow5_in_o = (!valid5_o) || allow6_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid5_o    <= 1'd0;
    end
    else if(allow5_in_o) begin
        valid5_o    <= valid4_i;
    end
    else begin
        valid5_o    <= valid5_o;
    end
end

// =========================================================================
// localparam：统一管理位宽，避免重复书写复杂表达式
// =========================================================================
localparam ADD1_W  = 28 + $clog2(`ADD_NUM0) + $clog2(`ADD_NUM1) + $clog2(`ADD_NUM2);
localparam BSAS_W  = `decimal_places1 + `block_int_w;

// =========================================================================
// 内部连线：接收 add1_stage 组合逻辑输出
// =========================================================================
wire signed [ADD1_W-1:0] sign_after_add1_w  [0:`P4_ADD1_OUT_NUM-1];
wire signed [BSAS_W-1:0] block_sig_af_shift_w [0:`P4_ADD0_OUT_NUM-1];

// =========================================================================
// 例化 add1_stage
// =========================================================================
add1_stage u_add1_stage (
    .sign_after_add0   (sign_after_add0_i),
    .scale_sig_i       (scale_sig_i  ),
    .block_shift_i     (block_shift_i),
    .is_mxint4         (is_MXINT4_i  ),

    .sign_after_add1   (sign_after_add1_w),
    .block_sig_af_shift(block_sig_af_shift_w)
);

// =========================================================================
// always_ff：所有输出打一拍
// =========================================================================
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        // add1_stage 结果清零
        for (int i = 0; i < `P4_ADD1_OUT_NUM; i++)
            sign_after_add1_o[i] <= '0;
        for (int i = 0; i < `P4_ADD0_OUT_NUM; i++)
            block_sig_af_shift_o[i] <= '0;

        // 穿透信号清零
        block_max_o  <= '0;
        nvc_shift_o  <= '0;
        E_MAX_o      <= '0;
        c_shift_o    <= '0;
        NAN_o        <= 1'b0;
        PINF_o       <= 1'b0;
        NINF_o       <= 1'b0;
        C_sig_o      <= 32'b0;
        C_sign_o     <= 1'b0;
        // data_type_o  <= 2'b0;
        is_MXINTFP4_o       <= 1'b0;
        is_INT_o            <= 1'b0;
    end else if(allow5_in_o) begin
        // --- add1_stage 结果寄存 ---
        for (int i = 0; i < `P4_ADD1_OUT_NUM; i++)
            sign_after_add1_o[i] <= sign_after_add1_w[i];
        for (int i = 0; i < `P4_ADD0_OUT_NUM; i++)
            block_sig_af_shift_o[i] <= block_sig_af_shift_w[i];

        // --- 穿透信号寄存 ---
        block_max_o  <= block_max_i;
        nvc_shift_o  <= nvc_shift_i;
        E_MAX_o      <= E_MAX_i;
        c_shift_o    <= c_shift_i;
        NAN_o        <= NAN_i;
        PINF_o       <= PINF_i;
        NINF_o       <= NINF_i;
        C_sig_o      <= C_sig_i;
        C_sign_o     <= C_sign_i;
        // data_type_o  <= data_type_i;
        is_MXINTFP4_o       <= is_MXNVFP4_i || is_MXINT4_i;
        is_INT_o            <= is_INT8_i || is_INT4_i;
    end
end

endmodule
