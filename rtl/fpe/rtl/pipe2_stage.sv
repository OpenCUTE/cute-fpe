`include "defines.sv"

module pipe2_stage (
    input wire clock,
    input wire reset,

    // 透传异常标志
    input wire         NAN_i,
    input wire         PINF_i,
    input wire         NINF_i,

    input wire  [`ResultWidth - 1 : 0] CAdd_bits_i,

    // scale 解码结果
    input wire  [8:0]  scale_exp_i  [0:`Scale_NUM - 1],
    input wire  [7:0]  scale_sig_i  [0:`Scale_NUM - 1],  // 穿透

    // 控制标志（穿透）
    // input wire  [1:0]  data_type_i,
    input wire         is_High_precision_i,
    input wire         is_low_precision_i,
    input wire         is_MXNVFP4_i,
    input wire         is_INT8_i,
    input wire         is_INT4_i,
    input wire         is_MXINT4_i,
    input wire         scale_type_i,

    input wire         is_threshold_normal_i,
    input wire         is_threshold_mxintfp4_i,

    input wire signed [7:0] bias_i,

    // 尾数乘法结果
    input wire  [21:0] product_out_i [0:`LANE_NUM-1][0:1],

    input wire  [5:0]  exp_E6_i [0:`LANE_NUM-1][0:3],

    // 掩码输出（block0 / block1 各两路，宽度 BLOCK_NUM）
    input wire  [`BLOCK_NUM -1: 0]  mask0_0_i, mask0_1_i,
    input wire  [`BLOCK_NUM -1: 0]  mask1_0_i, mask1_1_i,
    input wire  [`BLOCK_NUM -1: 0]  remain_data_E3_0_0_i [0:2],
    input wire  [`BLOCK_NUM -1: 0]  remain_data_E3_0_1_i [0:2],
    input wire  [`BLOCK_NUM -1: 0]  remain_data_E3_1_0_i [0:2],
    input wire  [`BLOCK_NUM -1: 0]  remain_data_E3_1_1_i [0:2],

    input wire  [3:0]  vec4_sign_i [0:`LANE_NUM-1],

    input wire  [2:0] E3_MAX0_0_i, E3_MAX0_1_i,
    input wire  [2:0] E3_MAX1_0_i, E3_MAX1_1_i,
    input wire  [5:0] E6_MAX0_0_i, E6_MAX0_1_i,
    input wire  [5:0] E6_MAX1_0_i, E6_MAX1_1_i,

    //握手信号
    //pipe1 -- pipe2
    input  wire valid1_i,
    output wire allow2_in_o,
    //pipe2 -- pipe3
    output reg  valid2_o,
    input  wire allow3_in_i,

    // -------------------------------------------------------
    // 打一拍后的输出：shift_stage 所有输出 + 穿透信号
    // -------------------------------------------------------

    // shift_stage 计算结果（寄存后输出）\
    //解码C
    output reg [7 :0] C_exp_o,
    output reg [31:0] C_sig_o,
    output reg        C_sign_o,

    // EabMAX_choose 输出
    output reg [8:0] Eab_MAX0_o, Eab_MAX1_o,

    // scale 最大指数搜索结果
    output reg [8:0] scale_max_o,

    //输出阈值情况
    output reg threshold_mxintfp4_en_o,
    output reg [9:0] threshold_normal_exp_0_o, threshold_normal_exp_1_o,

    output reg [8:0] scale_exp_s0_o,scale_exp_s1_o,

    // 穿透信号（寄存后输出）
    output reg [5:0]  exp_E6_o [0:`LANE_NUM-1][0:3],

    output reg  [7:0]  bias_o,
    output reg         NAN_o,
    output reg         PINF_o,
    output reg         NINF_o,
    output reg         C_ZERO_o,
    output reg  [8:0]  scale_exp_o   [0:`Scale_NUM - 1],
    output reg  [7:0]  scale_sig_o   [0:`Scale_NUM - 1],
    output reg  [21:0] product_out_o [0:`LANE_NUM-1][0:1],
    output reg  [3:0]  vec4_sign_o   [0:`LANE_NUM-1],
    // output reg  [1:0]  data_type_o,
    output reg         is_High_precision_o,
    output reg         is_low_precision_o,
    output reg         is_MXNVFP4_o,
    output reg         is_INT4_o,
    output reg         is_MXINT4_o,
    output reg         is_INT8_o,
    output reg         scale_type_o
);

//握手逻辑
assign allow2_in_o = (!valid2_o) || allow3_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid2_o    <= 1'd0;
    end
    else if(allow2_in_o) begin
        valid2_o    <= valid1_i;
    end
    else begin
        valid2_o    <= valid2_o;
    end
end

    // =========================================================
    // shift_stage 组合逻辑输出（未寄存）
    // =========================================================

    //解码C
    wire [7 :0] C_exp_w;
    wire [31:0] C_sig_w;
    wire        C_sign_w;
    wire C_NAN_w, C_NINF_w, C_PINF_w, C_ZERO_w;

    // EabMAX_choose 输出
    wire [8:0] Eab_MAX0_w, Eab_MAX1_w;

    // scale 最大指数搜索结果
    wire [8:0] scale_max_w;

    //输出阈值情况
    wire threshold_mxintfp4_en_w;
    wire [9:0] threshold_normal_exp_0_w, threshold_normal_exp_1_w;

    // =========================================================
    // 例化 shift_stage
    // =========================================================

    shift_stage u_shift_stage (
        .scale_exp          (scale_exp_i        ),
        .is_High_precision  (is_High_precision_i),
        .is_INT8            (is_INT8_i          ),
        .is_INT4            (is_INT4_i          ),
        .CAdd_bits          (CAdd_bits_i        ),

        .bias               (bias_i             ),

        .is_threshold_normal     (is_threshold_normal_i   ),
        .is_threshold_mxintfp4   (is_threshold_mxintfp4_i ),

        // .exp_E6             (exp_E6_i           ),
        //block0
        .mask0_0             (mask0_0_i            ),
        .mask0_1             (mask0_1_i            ),
        .remain_data_E3_0_0  (remain_data_E3_0_0_i ),
        .remain_data_E3_0_1  (remain_data_E3_0_1_i ),
        .E3_MAX0_0           (E3_MAX0_0_i          ),
        .E3_MAX0_1           (E3_MAX0_1_i          ),
        .E6_MAX0_0           (E6_MAX0_0_i          ),
        .E6_MAX0_1           (E6_MAX0_1_i          ),
        //block1
        .mask1_0             (mask1_0_i            ),
        .mask1_1             (mask1_1_i            ),
        .remain_data_E3_1_0  (remain_data_E3_1_0_i ),
        .remain_data_E3_1_1  (remain_data_E3_1_1_i ),
        .E3_MAX1_0           (E3_MAX1_0_i          ),
        .E3_MAX1_1           (E3_MAX1_1_i          ),
        .E6_MAX1_0           (E6_MAX1_0_i          ),
        .E6_MAX1_1           (E6_MAX1_1_i          ),

        //输出相关的结果

        //解码C
        .C_exp              (C_exp_w ),
        .C_sig              (C_sig_w ),
        .C_sign             (C_sign_w),
        .C_NAN              (C_NAN_w ), 
        .C_NINF             (C_NINF_w), 
        .C_PINF             (C_PINF_w), 
        .C_ZERO             (C_ZERO_w),

        // EabMAX_choose 输出
        .Eab_MAX0           (Eab_MAX0_w), 
        .Eab_MAX1           (Eab_MAX1_w),

        // scale 最大指数搜索结果
        .scale_max          (scale_max_w),

        //输出阈值情况
        .threshold_mxintfp4_en  (threshold_mxintfp4_en_w),
        .threshold_normal_exp_0 (threshold_normal_exp_0_w),
        .threshold_normal_exp_1 (threshold_normal_exp_1_w)
    );

    // =========================================================
    // 所有输出打一拍
    // =========================================================

    integer i,j;

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            C_exp_o      <= '0;
            C_sig_o      <= '0;
            C_sign_o     <= '0;
            
            Eab_MAX0_o    <= '0;
            Eab_MAX1_o    <= '0;
            scale_max_o   <= '0;

            threshold_mxintfp4_en_o <= 1'b0;
            threshold_normal_exp_0_o <= '0;
            threshold_normal_exp_1_o <= '0;

            scale_exp_s0_o <= '0;
            scale_exp_s1_o <= '0;

            // 穿透信号
            bias_o       <= '0;

            NAN_o        <= '0;
            PINF_o       <= '0;
            NINF_o       <= '0;
            C_ZERO_o     <= '0;
            // data_type_o  <= '0;
            is_High_precision_o <= 1'b0;
            is_low_precision_o  <= 1'b0;
            is_MXNVFP4_o       <= 1'b0;
            is_INT4_o           <= 1'b0;
            is_MXINT4_o         <= 1'b0;
            is_INT8_o           <= 1'b0;
            scale_type_o <= '0;
            for (i = 0; i < `Scale_NUM; i++) begin
                scale_sig_o[i] <= '0;
                scale_exp_o[i] <= '0;
            end
            for (i = 0; i < `LANE_NUM; i++) begin
                vec4_sign_o[i] <= 4'd0;
                for(j = 0; j < 4; j++)begin
                    exp_E6_o[i][j] <= 6'd0;
                end
                for(j = 0; j < 2; j++)begin
                    product_out_o[i][j] <= 22'd0;
                end
            end
        
        end else if(allow2_in_o) begin
            // shift_stage 计算结果寄存
            C_exp_o      <= C_exp_w;
            C_sig_o      <= C_sig_w;
            C_sign_o     <= C_sign_w;

            bias_o       <= bias_i;

            scale_exp_s0_o <= $signed({scale_exp_i[0]}) - 9'sd254;
            scale_exp_s1_o <= $signed({scale_exp_i[`Scale_NUM == 8 ? 2 : 1]}) - 9'sd254;


            Eab_MAX0_o    <= Eab_MAX0_w;
            Eab_MAX1_o    <= Eab_MAX1_w;
            scale_max_o   <= scale_max_w;

            //输出阈值情况
            threshold_mxintfp4_en_o <= threshold_mxintfp4_en_w;
            threshold_normal_exp_0_o <= threshold_normal_exp_0_w;
            threshold_normal_exp_1_o <= threshold_normal_exp_1_w;

            // 穿透信号寄存
            NAN_o        <= (NAN_i || C_NAN_w) || (C_NINF_w && PINF_i) || (C_PINF_w && NINF_i);
            PINF_o       <= (!(NAN_i || C_NAN_w)) && ((PINF_i && !C_NINF_w) || (!NINF_i && C_PINF_w));
            NINF_o       <= (!(NAN_i || C_NAN_w)) && ((!PINF_i && C_NINF_w) || (NINF_i && !C_PINF_w));
            C_ZERO_o     <= C_ZERO_w;
            // data_type_o  <= data_type_i;
            is_High_precision_o <= is_High_precision_i;
            is_low_precision_o  <= is_low_precision_i;
            is_MXNVFP4_o       <= is_MXNVFP4_i;
            is_INT4_o           <= is_INT4_i;
            is_MXINT4_o         <= is_MXINT4_i;
            is_INT8_o           <= is_INT8_i;
            scale_type_o <= scale_type_i;
            for (i = 0; i < `Scale_NUM; i++) begin
                scale_sig_o[i] <= scale_sig_i[i];
                scale_exp_o[i] <= scale_exp_i[i];
            end
            for (i = 0; i < `LANE_NUM; i++) begin
                vec4_sign_o[i] <= vec4_sign_i[i];
                for(j = 0; j < 4; j++)begin
                    exp_E6_o[i][j] <= exp_E6_i[i][j];
                end
                for(j = 0; j < 2; j++)begin
                    product_out_o[i][j] <= product_out_i[i][j];
                end
            end
        end
    end

endmodule
