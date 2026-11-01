`include "defines.sv"

module pipe4_stage (
    input wire clock,
    input wire reset,

    //打一拍之后输出
    input wire signed [9:0] block_max_i,
    input wire [5:0]  block_shift_i [0:`Scale_NUM - 1],
    input wire [5:0]  nvc_shift_i,
    input wire signed [9:0] E_MAX_i,
    input wire [4:0]  c_shift_i,

    //不穿透
    input wire [4:0]  exp_shift_out_i [0:`LANE_NUM - 1][0:3],
    input wire [21:0] product_out_i   [0:`LANE_NUM-1][0:1],
    input wire [3:0]  vec4_sign_i     [0:`LANE_NUM-1],

    //穿透信号，打一拍
    input wire        NAN_i,
    input wire        PINF_i,
    input wire        NINF_i,

    //打一拍输出
    input wire [31:0] C_sig_i,
    input wire        C_sign_i,
    input wire [7:0]  scale_sig_i [0:`Scale_NUM - 1],
    // input wire [1:0]  data_type_i,
    input wire        is_High_precision_i,
    input wire        is_low_precision_i,
    input wire        is_MXNVFP4_i,
    input wire        is_INT8_i,
    input wire        is_INT4_i,
    input wire        is_MXINT4_i,
    input wire        is_MXINTFP4_i,

    //握手信号
    //pipe3 -- pipe4
    input  wire valid3_i,
    output wire allow4_in_o,
    //pipe4 -- pipe5
    output reg  valid4_o,
    input  wire allow5_in_i,

    // =========================================================
    // 输出：add0_stage 结果，打一拍
    // =========================================================
    output reg signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0]
                      sign_after_add0_o [0:`P4_ADD0_OUT_NUM-1],

    // =========================================================
    // 穿透信号，打一拍输出
    // =========================================================
    output reg signed [9:0] block_max_o,
    output reg [5:0]  block_shift_o [0:`Scale_NUM - 1],
    output reg [5:0]  nvc_shift_o,
    output reg signed [9:0] E_MAX_o,
    output reg [4:0]  c_shift_o,
    output reg        NAN_o,
    output reg        PINF_o,
    output reg        NINF_o,
    output reg [31:0] C_sig_o,
    output reg        C_sign_o,
    output reg [7:0]  scale_sig_o [0:`Scale_NUM - 1],
    // output reg [1:0]  data_type_o
    output reg        is_MXNVFP4_o,
    output reg        is_INT8_o,
    output reg        is_INT4_o,
    output reg        is_MXINT4_o
);

//握手逻辑
assign allow4_in_o = (!valid4_o) || allow5_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid4_o    <= 1'd0;
    end
    else if(allow4_in_o) begin
        valid4_o    <= valid3_i;
    end
    else begin
        valid4_o    <= valid4_o;
    end
end

// =========================================================================
// 内部连线：接收 add0_stage 组合逻辑输出
// =========================================================================
localparam ADD0_W = 28 + $clog2(`ADD_NUM0) + $clog2(`ADD_NUM1);

wire signed [ADD0_W-1:0] sign_after_add0_w [0:`P4_ADD0_OUT_NUM-1];

// =========================================================================
// 例化 add0_stage
// =========================================================================
add0_stage u_add0_stage (
    .exp_shift_out_i (exp_shift_out_i),
    .product_out_i   (product_out_i),
    .vec4_sign_i     (vec4_sign_i),
    .is_High_precision_i(is_High_precision_i),
    .is_low_precision_i (is_low_precision_i ),
    .is_MXINTFP4_i      (is_MXINTFP4_i      ),
    // .is_INT8_i        (is_INT8_i          ),
    .sign_after_add0 (sign_after_add0_w)
);

// =========================================================================
// always_ff：所有输出打一拍
// =========================================================================
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        // add0_stage 结果清零
        for (int i = 0; i < `P4_ADD0_OUT_NUM; i++)
            sign_after_add0_o[i] <= '0;

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
        is_MXNVFP4_o        <= 1'b0;
        is_INT8_o           <= 1'b0;
        is_INT4_o           <= 1'b0;
        is_MXINT4_o         <= 1'b0;

        for (int i = 0; i < `Scale_NUM; i++) begin
            block_shift_o[i] <= '0;
            scale_sig_o[i]   <= '0;
        end
    end else if(allow4_in_o) begin
        // --- add0_stage 结果寄存 ---
        for (int i = 0; i < `P4_ADD0_OUT_NUM; i++)
            sign_after_add0_o[i] <= sign_after_add0_w[i];

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
        is_MXNVFP4_o        <= is_MXNVFP4_i;
        is_INT8_o           <= is_INT8_i;
        is_INT4_o           <= is_INT4_i;
        is_MXINT4_o         <= is_MXINT4_i;

        for (int i = 0; i < `Scale_NUM; i++) begin
            block_shift_o[i] <= block_shift_i[i];
            scale_sig_o[i]   <= scale_sig_i[i];
        end
    end
end

endmodule
