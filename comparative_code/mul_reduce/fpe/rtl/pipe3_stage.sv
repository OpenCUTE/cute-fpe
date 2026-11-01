`include "defines.sv"

module pipe3_stage (
    input wire clock,
    input wire reset,
    //解码C
    input wire [7 :0] C_exp_i, //不穿透，这一级使用
    input wire [31:0] C_sig_i, //穿透
    input wire        C_sign_i,//穿透
    input wire        C_ZERO_i,//不穿透，这一级使用

    // EabMAX_choose 输出
    input wire [8:0] Eab_MAX0_i, Eab_MAX1_i,//不穿透，这一级使用

    // scale 最大指数搜索结果
    input wire [8:0] scale_max_i, //不穿透，这一级使用

    input wire threshold_mxintfp4_en_i,
    input wire [9:0] threshold_normal_exp_0_i, threshold_normal_exp_1_i,
    
    input wire signed [8:0] scale_exp_s0_i,scale_exp_s1_i,

    //不穿透，这级使用
    input wire [5:0]  exp_E6_i [0:`LANE_NUM-1][0:3],

    input wire signed [7:0] bias_i, //不穿透，这一级使用

    input wire  [8:0]  scale_exp_i  [0:`Scale_NUM - 1],//不穿透，这一级使用

    //穿透，打一拍
    input wire         NAN_i,
    input wire         PINF_i,
    input wire         NINF_i,
    input wire  [7:0]  scale_sig_i   [0:`Scale_NUM - 1],
    input wire  [21:0] product_out_i [0:`LANE_NUM-1][0:1],
    input wire  [3:0]  vec4_sign_i   [0:`LANE_NUM-1],
    // input wire  [1:0]  data_type_i,
    input wire         is_High_precision_i,
    input wire         is_low_precision_i,
    input wire         is_MXNVFP4_i,
    input wire         is_INT8_i,
    input wire         is_INT4_i,
    input wire         is_MXINT4_i,
    input wire         scale_type_i,

    //握手信号
    //pipe2 -- pipe3
    input  wire valid2_i,
    output wire allow3_in_o,
    //pipe3 -- pipe4
    output reg  valid3_o,
    input  wire allow4_in_i,

    //输出
    //shift1_stage.sv 的输出信号，打一拍
    output reg signed [9:0]  block_max_o,
    output reg [5:0]  block_shift_o [0:`Scale_NUM - 1],
    output reg [5:0]  nvc_shift_o,
    output reg signed [9:0]  E_MAX_o,
    output reg [4:0]  c_shift_o,
    output reg [4:0]  exp_shift_out_o [0:`LANE_NUM - 1][0:3],

    //穿透信号，打一拍
    output reg        NAN_o,
    output reg        PINF_o,
    output reg        NINF_o,
    output reg [31:0] C_sig_o,
    output reg        C_sign_o,
    output reg [7:0]  scale_sig_o   [0:`Scale_NUM - 1],
    output reg [21:0] product_out_o [0:`LANE_NUM-1][0:1],
    output reg [3:0]  vec4_sign_o   [0:`LANE_NUM-1],
    // output reg [1:0]  data_type_o
    output reg        is_High_precision_o,
    output reg        is_low_precision_o,
    output reg        is_MXNVFP4_o,
    output reg        is_INT8_o,
    output reg        is_INT4_o,
    output reg        is_MXINT4_o,
    output reg        is_MXINTFP4_o
);

//握手逻辑
assign allow3_in_o = (!valid3_o) || allow4_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid3_o    <= 1'd0;
    end
    else if(allow3_in_o) begin
        valid3_o    <= valid2_i;
    end
    else begin
        valid3_o    <= valid3_o;
    end
end

// =========================================================================
// 内部连线：接收 shift1_stage 组合逻辑输出
// =========================================================================
wire signed [9:0] block_max_w;
wire [5:0]  block_shift_w [0:`Scale_NUM - 1];
wire [5:0]  nvc_shift_w;
wire signed [9:0]  E_MAX_w;
wire [4:0]  c_shift_w;
wire [4:0]  exp_shift_out_w [0:`LANE_NUM - 1][0:3];

// =========================================================================
// 例化 shift1_stage
// =========================================================================
shift1_stage u_shift1_stage (
    .scale_type     (scale_type_i),
    .scale_exp      (scale_exp_i),
    .scale_max      (scale_max_i),
    .threshold_mxintfp4_en (threshold_mxintfp4_en_i),
    .threshold_normal_exp_0 (threshold_normal_exp_0_i),
    .threshold_normal_exp_1 (threshold_normal_exp_1_i),

    .scale_exp_s0   (scale_exp_s0_i),
    .scale_exp_s1   (scale_exp_s1_i),
    .C_exp          (C_exp_i),
    .C_ZERO         (C_ZERO_i),
    .bias           (bias_i),
    .Eab_MAX0       (Eab_MAX0_i),
    .Eab_MAX1       (Eab_MAX1_i),
    .exp_E6_i       (exp_E6_i),
    .is_High_precision_i(is_High_precision_i),
    .is_low_precision_i (is_low_precision_i ),
    .is_INT8_i          (is_INT8_i          ),
    .block_max      (block_max_w),
    .block_shift    (block_shift_w),
    .nvc_shift      (nvc_shift_w),
    .E_MAX          (E_MAX_w),
    .c_shift        (c_shift_w),
    .exp_shift_out  (exp_shift_out_w)
);

// =========================================================================
// always_ff：将 shift1_stage 输出及穿透信号统一打一拍
// =========================================================================
always_ff @(posedge clock or posedge reset) begin
    if (reset) begin
        // shift1_stage 结果清零
        block_max_o  <= '0;
        nvc_shift_o  <= '0;
        E_MAX_o      <= '0;
        c_shift_o    <= '0;
        // 穿透信号清零
        NAN_o        <= 1'b0;
        PINF_o       <= 1'b0;
        NINF_o       <= 1'b0;
        C_sig_o      <= 32'b0;
        C_sign_o     <= 1'b0;
        // data_type_o  <= 2'b0;
        is_High_precision_o <= 1'b0;
        is_low_precision_o  <= 1'b0;
        is_MXNVFP4_o        <= 1'b0;
        is_INT8_o           <= 1'b0;
        is_INT4_o           <= 1'b0;
        is_MXINT4_o         <= 1'b0;
        is_MXINTFP4_o       <= 1'b0;

        for (int i = 0; i < `Scale_NUM; i++) begin
            block_shift_o[i] <= '0;
            scale_sig_o[i]   <= '0;
        end
        for (int i = 0; i < `LANE_NUM; i++) begin
            vec4_sign_o[i]      <= '0;
            product_out_o[i][0] <= '0;
            product_out_o[i][1] <= '0;
            for (int j = 0; j < 4; j++)
                exp_shift_out_o[i][j] <= '0;
        end
    end else if(allow3_in_o) begin
        // --- shift1_stage 结果寄存 ---
        block_max_o <= block_max_w;
        nvc_shift_o <= nvc_shift_w;
        E_MAX_o     <= E_MAX_w;
        c_shift_o   <= c_shift_w;

        for (int i = 0; i < `Scale_NUM; i++)
            block_shift_o[i] <= block_shift_w[i];

        for (int i = 0; i < `LANE_NUM; i++)
            for (int j = 0; j < 4; j++)
                exp_shift_out_o[i][j] <= exp_shift_out_w[i][j];

        // --- 穿透信号寄存 ---
        NAN_o        <= NAN_i;
        PINF_o       <= PINF_i;
        NINF_o       <= NINF_i;
        C_sig_o      <= C_sig_i;
        C_sign_o     <= C_sign_i;
        // data_type_o  <= data_type_i;
        is_High_precision_o <= is_High_precision_i;
        is_low_precision_o  <= is_low_precision_i;
        is_MXNVFP4_o        <= is_MXNVFP4_i;
        is_INT8_o           <= is_INT8_i;
        is_INT4_o           <= is_INT4_i;
        is_MXINT4_o         <= is_MXINT4_i;
        is_MXINTFP4_o       <= is_MXINT4_i || is_INT4_i || is_MXNVFP4_i;

        for (int i = 0; i < `Scale_NUM; i++)
            scale_sig_o[i] <= scale_sig_i[i];

        for (int i = 0; i < `LANE_NUM; i++) begin
            vec4_sign_o[i]      <= vec4_sign_i[i];
            product_out_o[i][0] <= product_out_i[i][0];
            product_out_o[i][1] <= product_out_i[i][1];
        end
    end
end

endmodule
