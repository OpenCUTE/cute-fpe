`include "defines.sv"

module pipe1_stage (
    input wire clock,
    input wire reset,

    // 透传信号
    input wire  [`ResultWidth - 1 : 0]   CAdd_bits, //穿透
    input wire  [`ScaleBusWidth - 1 : 0] AScale_bits,
    input wire  [`ScaleBusWidth - 1 : 0] BScale_bits,

    // 异常标志（decode_stage 已聚合为单 bit）
    // 这个异常直接穿透，打一拍进入下一级
    input wire         NAN_i,
    input wire         PINF_i,
    input wire         NINF_i,

    // opcode
    input wire  [3:0]  opcode,

    input wire  is_High_precision,
    input wire  is_low_precision,
    input wire  is_MXNVFP4,
    input wire  is_INT8,
    input wire  is_INT4,
    input wire  is_MXINT4,
    input wire  is_INTFP4,
    input wire  is_MXINTFP4,

    // 尾数乘法器输入（每通道 2 组）
    input wire  [7:0] a_8 [0 : `LANE_NUM-1][0:1],
    input wire  [3:0] a_4 [0 : `LANE_NUM-1][0:1],
    input wire  [3:0] b_4 [0 : `LANE_NUM-1][0:1],
    input wire  [7:0] b_8 [0 : `LANE_NUM-1][0:1],

    // 指数求和结果
    input wire [5:0] exp_E6 [0:`LANE_NUM-1][0:3],

    input wire [`Scale_NUM-1:0] MXINTFP4_all_zero,
    input wire [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero,

    // 符号位向量
    input wire  [7:0] vec8_sign [0:`LANE_NUM-1],

    //握手信号
    //pipe0 -- pipe1
    input  wire valid0_i,
    output wire allow1_in_o,
    //pipe1 -- pipe2
    output reg  valid1_o,
    input  wire allow2_in_i,

    // -------------------------------------------------------------------------
    // 输出到 pipe2（全部为寄存器输出）
    // -------------------------------------------------------------------------
    // 透传异常标志
    output reg         NAN_o,
    output reg         PINF_o,
    output reg         NINF_o,

    //穿透C信号
    output reg  [`ResultWidth - 1 : 0] CAdd_bits_o,

    // scale 解码结果
    output reg  [8:0]  scale_exp_o [0:`Scale_NUM - 1],
    output reg  [7:0]  scale_sig_o [0:`Scale_NUM - 1],

    // 控制标志
    // output reg  [1:0]  data_type_o,
    output reg         is_High_precision_o,
    output reg         is_low_precision_o,
    output reg         is_MXNVFP4_o,
    output reg         is_INT8_o,
    output reg         is_INT4_o,
    output reg         is_MXINT4_o,
    output reg         scale_type_o,
    output reg         is_threshold_normal_o,
    output reg         is_threshold_mxintfp4_o,

    output reg  [7:0]  bias_o,

    // 尾数乘法结果
    output reg  [21:0] product_out_o [0:`LANE_NUM-1][0:1],

    output reg  [5 :0] exp_E6_o [0:`LANE_NUM-1][0:3],

    // 掩码输出（block0 / block1 各两路，宽度 BLOCK_NUM）
    output reg [`BLOCK_NUM -1: 0]  mask0_0_o, mask0_1_o,
    output reg [`BLOCK_NUM -1: 0]  mask1_0_o, mask1_1_o,
    output reg [`BLOCK_NUM -1: 0]  remain_data_E3_0_0_o [0:2],
    output reg [`BLOCK_NUM -1: 0]  remain_data_E3_0_1_o [0:2],
    output reg [`BLOCK_NUM -1: 0]  remain_data_E3_1_0_o [0:2],
    output reg [`BLOCK_NUM -1: 0]  remain_data_E3_1_1_o [0:2],

    output reg  [3:0] vec4_sign_o [0:`LANE_NUM-1]  ,
 
    output reg [2:0] E3_MAX0_0_o, E3_MAX0_1_o,
    output reg [2:0] E3_MAX1_0_o, E3_MAX1_1_o,
    output reg [5:0] E6_MAX0_0_o, E6_MAX0_1_o,
    output reg [5:0] E6_MAX1_0_o, E6_MAX1_1_o
);

//握手逻辑
assign allow1_in_o = (!valid1_o) || allow2_in_i;
always @(posedge clock or posedge reset) begin
    if(reset)begin
        valid1_o    <= 1'd0;
    end
    else if(allow1_in_o) begin
        valid1_o    <= valid0_i;
    end
    else begin
        valid1_o    <= valid1_o;
    end
end

// =========================================================================
// mul_stage 组合逻辑输出（打拍前的中间线）
// =========================================================================
wire [8 :0]  scale_exp_w [0:`Scale_NUM - 1];
wire [7 :0]  scale_sig_w [0:`Scale_NUM - 1];
wire          scale_NAN_w;

// wire [1:0]   data_type_w;
wire         scale_type_w;
wire         is_threshold_normal_w;
wire         is_threshold_mxintfp4_w;
wire signed [7:0]  bias_w;

wire [21:0] product_out_w [0:`LANE_NUM-1][0:1];
wire [5 :0] exp_E6_o_w    [0:`LANE_NUM-1][0:3];

// 掩码输出（新接口：block0/block1 各两路）
wire [`BLOCK_NUM -1: 0]  mask0_0_w, mask0_1_w;
wire [`BLOCK_NUM -1: 0]  mask1_0_w, mask1_1_w;
wire [`BLOCK_NUM -1: 0]  remain_data_E3_0_0_w [0:2];
wire [`BLOCK_NUM -1: 0]  remain_data_E3_0_1_w [0:2];
wire [`BLOCK_NUM -1: 0]  remain_data_E3_1_0_w [0:2];
wire [`BLOCK_NUM -1: 0]  remain_data_E3_1_1_w [0:2];

wire [2:0] E3_MAX0_0_w, E3_MAX0_1_w;
wire [2:0] E3_MAX1_0_w, E3_MAX1_1_w;
wire [5:0] E6_MAX0_0_w, E6_MAX0_1_w;
wire [5:0] E6_MAX1_0_w, E6_MAX1_1_w;

// =========================================================================
// 例化 mul_stage
// =========================================================================
mul_stage u_mul_stage (
    .AScale_bits  (AScale_bits ),
    .BScale_bits  (BScale_bits ),
    .opcode       (opcode      ),

    .is_High_precision(is_High_precision),
    .is_low_precision (is_low_precision ),
    .is_INT8          (is_INT8          ),
    .is_INTFP4        (is_INTFP4        ),
    .is_MXINTFP4      (is_MXINTFP4      ),

    .a_8          (a_8          ),
    .a_4          (a_4          ),
    .b_4          (b_4          ),
    .b_8          (b_8          ),

    .exp_E6       (exp_E6      ),
    .MXINTFP4_all_zero (MXINTFP4_all_zero),
    .MXFP86_all_zero    (MXFP86_all_zero ),
    .vec8_sign    (vec8_sign   ),

    .scale_exp    (scale_exp_w ),
    .scale_sig    (scale_sig_w ),
    .scale_NAN    (scale_NAN_w ),

    .scale_type   (scale_type_w),

    .is_threshold_normal(is_threshold_normal_w),
    .is_threshold_mxintfp4 (is_threshold_mxintfp4_w),

    .bias         (bias_w      ),

    .product_out  (product_out_w),
    .exp_E6_o     (exp_E6_o_w   ),

    // 新接口：block0/block1 各两路掩码
    .mask0_0              (mask0_0_w            ),
    .mask0_1              (mask0_1_w            ),
    .remain_data_E3_0_0   (remain_data_E3_0_0_w ),
    .remain_data_E3_0_1   (remain_data_E3_0_1_w ),
    .mask1_0              (mask1_0_w            ),
    .mask1_1              (mask1_1_w            ),
    .remain_data_E3_1_0   (remain_data_E3_1_0_w ),
    .remain_data_E3_1_1   (remain_data_E3_1_1_w ),

    .E3_MAX0_0    (E3_MAX0_0_w ),
    .E3_MAX0_1    (E3_MAX0_1_w ),
    .E3_MAX1_0    (E3_MAX1_0_w ),
    .E3_MAX1_1    (E3_MAX1_1_w ),
    .E6_MAX0_0    (E6_MAX0_0_w ),
    .E6_MAX0_1    (E6_MAX0_1_w ),
    .E6_MAX1_0    (E6_MAX1_0_w ),
    .E6_MAX1_1    (E6_MAX1_1_w )
);

    // =========================================================================
    // 时序逻辑：所有信号打一拍
    // =========================================================================
    integer i, j;

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            NAN_o        <= 1'b0;
            PINF_o       <= 1'b0;
            NINF_o       <= 1'b0;
            CAdd_bits_o  <=  '0;

            // data_type_o  <= 2'b0;
            is_High_precision_o <= 1'b0;
            is_low_precision_o  <= 1'b0;
            is_MXNVFP4_o        <= 1'b0;
            is_INT8_o           <= 1'b0;
            is_INT4_o           <= 'b0;
            is_MXINT4_o         <= 'b0;
            scale_type_o <= 1'b0;
            is_threshold_normal_o    <= 1'b0;
            is_threshold_mxintfp4_o  <= 1'b0;
            bias_o       <= 'd0;

            mask0_0_o    <= '0;
            mask0_1_o    <= '0;
            mask1_0_o    <= '0;
            mask1_1_o    <= '0;
            E3_MAX0_0_o  <= 3'b0;
            E3_MAX0_1_o  <= 3'b0;
            E3_MAX1_0_o  <= 3'b0;
            E3_MAX1_1_o  <= 3'b0;
            E6_MAX0_0_o  <= 6'b0;
            E6_MAX0_1_o  <= 6'b0;
            E6_MAX1_0_o  <= 6'b0;
            E6_MAX1_1_o  <= 6'b0;

            for (i = 0; i < 3; i = i + 1) begin
                remain_data_E3_0_0_o[i] <= '0;
                remain_data_E3_0_1_o[i] <= '0;
                remain_data_E3_1_0_o[i] <= '0;
                remain_data_E3_1_1_o[i] <= '0;
            end

            for (i = 0; i < `Scale_NUM; i = i + 1) begin
                scale_exp_o[i] <= 9'b0;
                scale_sig_o[i] <= 8'b0;
            end

            for (i = 0; i < `LANE_NUM; i = i + 1) begin
                for (j = 0; j < 2; j = j + 1) begin
                    product_out_o[i][j] <= 22'b0;
                    exp_E6_o[i][j]      <= 6'd0;
                    exp_E6_o[i][j+2]    <= 6'd0;
                end
                vec4_sign_o[i]   <= 4'd0;
            end

        end else if(allow1_in_o) begin
            // 透传异常标志
            // NAN_o        <= (NAN_i || scale_NAN_w || C_NAN_w) || (C_NINF_w && PINF_i) || (C_PINF_w && NINF_i);
            // PINF_o       <= (!(NAN_i || scale_NAN_w || C_NAN_w)) && ((PINF_i && !C_NINF_w) || (!NINF_i && C_PINF_w));
            // NINF_o       <= (!(NAN_i || scale_NAN_w || C_NAN_w)) && ((!PINF_i && C_NINF_w) || (NINF_i && !C_PINF_w));
            NAN_o        <= (NAN_i || scale_NAN_w) ;
            PINF_o       <= (!scale_NAN_w) && PINF_i;
            NINF_o       <= (!scale_NAN_w) && NINF_i;
            CAdd_bits_o  <= CAdd_bits;

            // scale 解码结果
            for (i = 0; i < `Scale_NUM; i = i + 1) begin
                scale_exp_o[i] <= scale_exp_w[i];
                scale_sig_o[i] <= scale_sig_w[i];
            end

            // 控制标志
            // data_type_o  <= data_type_w;
            is_High_precision_o <= is_High_precision;
            is_low_precision_o  <= is_low_precision;
            is_MXNVFP4_o        <= is_MXNVFP4;
            is_INT8_o           <= is_INT8;
            is_INT4_o           <= is_INT4;
            is_MXINT4_o         <= is_MXINT4;
            scale_type_o <= scale_type_w;
            is_threshold_normal_o   <= is_threshold_normal_w;
            is_threshold_mxintfp4_o <= is_threshold_mxintfp4_w;
            bias_o       <= bias_w;

            // 掩码寄存（block0/block1 各两路）
            mask0_0_o <= mask0_0_w;
            mask0_1_o <= mask0_1_w;
            mask1_0_o <= mask1_0_w;
            mask1_1_o <= mask1_1_w;

            for (i = 0; i < 3; i = i + 1) begin
                remain_data_E3_0_0_o[i] <= remain_data_E3_0_0_w[i];
                remain_data_E3_0_1_o[i] <= remain_data_E3_0_1_w[i];
                remain_data_E3_1_0_o[i] <= remain_data_E3_1_0_w[i];
                remain_data_E3_1_1_o[i] <= remain_data_E3_1_1_w[i];
            end

            E3_MAX0_0_o <= E3_MAX0_0_w;
            E3_MAX0_1_o <= E3_MAX0_1_w;
            E3_MAX1_0_o <= E3_MAX1_0_w;
            E3_MAX1_1_o <= E3_MAX1_1_w;
            E6_MAX0_0_o <= E6_MAX0_0_w;
            E6_MAX0_1_o <= E6_MAX0_1_w;
            E6_MAX1_0_o <= E6_MAX1_0_w;
            E6_MAX1_1_o <= E6_MAX1_1_w;

            // 尾数乘法结果
            for (i = 0; i < `LANE_NUM; i = i + 1) begin
                for (j = 0; j < 2; j = j + 1) begin
                    product_out_o[i][j] <= product_out_w[i][j];
                    exp_E6_o[i][j]      <= exp_E6_o_w[i][j];
                    exp_E6_o[i][j+2]    <= exp_E6_o_w[i][j+2];
                end
                vec4_sign_o[i]  <= vec8_sign[i][3:0];
            end
        end
    end

endmodule
