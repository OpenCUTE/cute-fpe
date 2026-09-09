`include "defines.sv"

module top (
    input  wire        clock,
    input  wire        reset,

    // ===================== 顶层输入（来自外部） =====================

    // A/B 向量输入
    input  wire [`ReduceWidth - 1 : 0]   AVector_bits,
    input  wire                          AVector_valid,

    input  wire [`ReduceWidth - 1 : 0]   BVector_bits,
    input  wire                          BVector_valid,

    // C 累加输入（FP32）
    input  wire [`ResultWidth - 1 : 0]   CAdd_bits,
    input  wire                          CAdd_valid,

    // A/B Scale 输入（MX/NV 格式）
    input  wire [`ScaleBusWidth - 1 : 0] AScale_bits,
    input  wire                          AScale_valid,

    input  wire [`ScaleBusWidth - 1 : 0] BScale_bits,
    input  wire                          BScale_valid,

    // 操作码
    input  wire [4:0]                    opcode,

    // 上游 ready 输出（反压给外部）
    output wire                          AVector_ready,
    output wire                          BVector_ready,
    output wire                          CAdd_ready,
    output wire                          AScale_ready,
    output wire                          BScale_ready,

    // ===================== 顶层握手输出 =====================
    output wire                          valid8_o,     // pipe8 输出有效
    input  wire                          allow9_in_i,  // 下游允许接收

    // ===================== 顶层输出（来自 pipe8_stage） =====================
    output wire [31:0]                   D_bits_o
);

    // =========================================================================
    // 级间握手连线
    // =========================================================================

    wire  allow1_in;   // pipe0 → pipe1 allow
    wire  valid0_o;    // pipe0 → pipe1 valid

    wire  allow2_in;   // pipe1 → pipe2 allow
    wire  valid1_o;    // pipe1 → pipe2 valid

    wire  allow3_in;   // pipe2 → pipe3 allow
    wire  valid2_o;    // pipe2 → pipe3 valid

    wire  allow4_in;   // pipe3 → pipe4 allow
    wire  valid3_o;    // pipe3 → pipe4 valid

    wire  allow5_in;   // pipe4 → pipe5 allow
    wire  valid4_o;    // pipe4 → pipe5 valid

    wire  allow6_in;   // pipe5 → pipe6 allow
    wire  valid5_o;    // pipe5 → pipe6 valid

    wire  allow7_in;   // pipe6 → pipe7 allow
    wire  valid6_o;    // pipe6 → pipe7 valid

    wire  allow8_in;   // pipe7 → pipe8 allow
    wire  valid7_o;    // pipe7 → pipe8 valid

    // =========================================================================
    // pipe0_stage → pipe1_stage 连线
    // =========================================================================

    // 透传信号（打一拍）
    wire [`ResultWidth - 1 : 0]   p0_CAdd_bits_r;
    wire [`ScaleBusWidth - 1 : 0] p0_AScale_bits_r;
    wire [`ScaleBusWidth - 1 : 0] p0_BScale_bits_r;

    // 异常标志
    wire        p0_NAN_o_r;
    wire        p0_PINF_o_r;
    wire        p0_NINF_o_r;

    // opcode 透传
    wire [3:0]  p0_opcode_r;

    // 控制标志
    wire        p0_is_High_precision_r;
    wire        p0_is_low_precision_r;
    wire        p0_is_MXNVFP4_r;
    wire        p0_is_INT8_r;
    wire        p0_is_INT4_r;
    wire        p0_is_MXINT4_r;
    wire        p0_is_INTFP4_r;
    wire        p0_is_MXINTFP4_r;

    // 尾数乘法器输入
    wire [7:0]  p0_a_8_r [0:`LANE_NUM-1][0:1];
    wire [3:0]  p0_a_4_r [0:`LANE_NUM-1][0:1];
    wire [3:0]  p0_b_4_r [0:`LANE_NUM-1][0:1];
    wire [7:0]  p0_b_8_r [0:`LANE_NUM-1][0:1];

    // 指数求和结果
    wire [5:0]  p0_exp_E6_r [0:`LANE_NUM-1][0:3];

    // FP4 全零标志
    wire [`Scale_NUM-1:0] p0_MXINTFP4_all_zero_r;
    wire [`MXFP86_ZERO_GROUP_NUM-1:0] p0_MXFP86_all_zero_r;

    // 符号位向量
    wire [7:0]  p0_vec8_sign_r [0:`LANE_NUM-1];

    // =========================================================================
    // pipe1_stage → pipe2_stage 连线
    // =========================================================================

    // 异常标志
    wire        p1_NAN_o;
    wire        p1_PINF_o;
    wire        p1_NINF_o;

    // C 穿透
    wire [`ResultWidth - 1 : 0] p1_CAdd_bits_o;

    // scale 解码结果
    wire [8:0]  p1_scale_exp_o [0:`Scale_NUM-1];
    wire [7:0]  p1_scale_sig_o [0:`Scale_NUM-1];

    // 控制标志
    // wire [1:0]  p1_data_type_o;
    wire        p1_is_High_precision_o;
    wire        p1_is_low_precision_o;
    wire        p1_is_MXNVFP4_o;
    wire        p1_is_INT8_o;
    wire        p1_is_INT4_o;
    wire        p1_is_MXINT4_o;
    wire        p1_scale_type_o;
    wire        p1_is_threshold_normal_o;
    wire        p1_is_threshold_mxintfp4_o;
    wire signed [7:0]  p1_bias_o;

    // 尾数乘法结果
    wire [21:0] p1_product_out_o [0:`LANE_NUM-1][0:1];
    wire [5:0]  p1_exp_E6_o      [0:`LANE_NUM-1][0:3];

    // 掩码
    wire [`BLOCK_NUM-1:0] p1_mask0_0_o, p1_mask0_1_o, p1_mask1_0_o, p1_mask1_1_o;
    wire [`BLOCK_NUM-1:0] p1_remain_data_E3_0_0_o [0:2];
    wire [`BLOCK_NUM-1:0] p1_remain_data_E3_0_1_o [0:2];
    wire [`BLOCK_NUM-1:0] p1_remain_data_E3_1_0_o [0:2];
    wire [`BLOCK_NUM-1:0] p1_remain_data_E3_1_1_o [0:2];

    // 符号位
    wire [3:0]  p1_vec4_sign_o [0:`LANE_NUM-1];

    // E3/E6 最大值
    wire [2:0]  p1_E3_MAX0_0_o, p1_E3_MAX0_1_o, p1_E3_MAX1_0_o, p1_E3_MAX1_1_o;
    wire [5:0]  p1_E6_MAX0_0_o, p1_E6_MAX0_1_o, p1_E6_MAX1_0_o, p1_E6_MAX1_1_o;

    // =========================================================================
    // 例化 pipe0_stage
    // =========================================================================
    pipe0_stage u_pipe0_stage (
        .clock          (clock          ),
        .reset          (reset          ),

        .AVector_bits   (AVector_bits   ),
        .AVector_valid  (AVector_valid  ),
        .AVector_ready  (AVector_ready  ),

        .BVector_bits   (BVector_bits   ),
        .BVector_valid  (BVector_valid  ),
        .BVector_ready  (BVector_ready  ),

        .CAdd_bits      (CAdd_bits      ),
        .CAdd_valid     (CAdd_valid     ),
        .CAdd_ready     (CAdd_ready     ),

        .AScale_bits    (AScale_bits    ),
        .AScale_valid   (AScale_valid   ),
        .AScale_ready   (AScale_ready   ),

        .BScale_bits    (BScale_bits    ),
        .BScale_valid   (BScale_valid   ),
        .BScale_ready   (BScale_ready   ),

        .opcode         (opcode         ),

        .allow1_in_i    (allow1_in      ),
        .valid0_o       (valid0_o       ),

        .CAdd_bits_r    (p0_CAdd_bits_r ),
        .AScale_bits_r  (p0_AScale_bits_r),
        .BScale_bits_r  (p0_BScale_bits_r),
        .NAN_o_r        (p0_NAN_o_r     ),
        .PINF_o_r       (p0_PINF_o_r    ),
        .NINF_o_r       (p0_NINF_o_r    ),
        .opcode_r       (p0_opcode_r    ),
        .is_High_precision_r(p0_is_High_precision_r),
        .is_low_precision_r (p0_is_low_precision_r ),
        .is_MXNVFP4_r       (p0_is_MXNVFP4_r       ),
        .is_INT8_r          (p0_is_INT8_r          ),
        .is_INT4_r          (p0_is_INT4_r          ),
        .is_MXINT4_r        (p0_is_MXINT4_r        ),
        .is_INTFP4_r        (p0_is_INTFP4_r        ),
        .is_MXINTFP4_r      (p0_is_MXINTFP4_r      ),
        .a_8_r          (p0_a_8_r       ),
        .a_4_r          (p0_a_4_r       ),
        .b_4_r          (p0_b_4_r       ),
        .b_8_r          (p0_b_8_r       ),
        .exp_E6_r       (p0_exp_E6_r    ),
        .MXINTFP4_all_zero_r(p0_MXINTFP4_all_zero_r),
        .MXFP86_all_zero_r(p0_MXFP86_all_zero_r),
        .vec8_sign_r    (p0_vec8_sign_r )
    );

    // =========================================================================
    // 例化 pipe1_stage
    // =========================================================================
    pipe1_stage u_pipe1_stage (
        .clock              (clock              ),
        .reset              (reset              ),

        .CAdd_bits          (p0_CAdd_bits_r     ),
        .AScale_bits        (p0_AScale_bits_r   ),
        .BScale_bits        (p0_BScale_bits_r   ),
        .NAN_i              (p0_NAN_o_r         ),
        .PINF_i             (p0_PINF_o_r        ),
        .NINF_i             (p0_NINF_o_r        ),
        .opcode             (p0_opcode_r        ),
        .is_High_precision  (p0_is_High_precision_r),
        .is_low_precision   (p0_is_low_precision_r ),
        .is_MXNVFP4         (p0_is_MXNVFP4_r       ),
        .is_INT8            (p0_is_INT8_r          ),
        .is_INT4            (p0_is_INT4_r          ),
        .is_MXINT4          (p0_is_MXINT4_r        ),
        .is_INTFP4          (p0_is_INTFP4_r        ),
        .is_MXINTFP4        (p0_is_MXINTFP4_r      ),
        .a_8                (p0_a_8_r           ),
        .a_4                (p0_a_4_r           ),
        .b_4                (p0_b_4_r           ),
        .b_8                (p0_b_8_r           ),
        .exp_E6             (p0_exp_E6_r        ),
        .MXINTFP4_all_zero  (p0_MXINTFP4_all_zero_r),
        .MXFP86_all_zero    (p0_MXFP86_all_zero_r),
        .vec8_sign          (p0_vec8_sign_r     ),

        .valid0_i           (valid0_o           ),
        .allow1_in_o        (allow1_in          ),
        .valid1_o           (valid1_o           ),
        .allow2_in_i        (allow2_in          ),

        .NAN_o              (p1_NAN_o           ),
        .PINF_o             (p1_PINF_o          ),
        .NINF_o             (p1_NINF_o          ),
        .CAdd_bits_o        (p1_CAdd_bits_o     ),
        .scale_exp_o        (p1_scale_exp_o     ),
        .scale_sig_o        (p1_scale_sig_o     ),
        .is_High_precision_o(p1_is_High_precision_o),
        .is_low_precision_o (p1_is_low_precision_o ),
        .is_MXNVFP4_o      (p1_is_MXNVFP4_o      ),
        .is_INT8_o          (p1_is_INT8_o          ),
        .is_INT4_o          (p1_is_INT4_o          ),
        .is_MXINT4_o        (p1_is_MXINT4_o        ),
        .scale_type_o       (p1_scale_type_o    ),
        .is_threshold_normal_o(p1_is_threshold_normal_o),
        .is_threshold_mxintfp4_o(p1_is_threshold_mxintfp4_o),
        .bias_o             (p1_bias_o          ),
        .product_out_o      (p1_product_out_o   ),
        .exp_E6_o           (p1_exp_E6_o        ),
        .mask0_0_o          (p1_mask0_0_o       ),
        .mask0_1_o          (p1_mask0_1_o       ),
        .mask1_0_o          (p1_mask1_0_o       ),
        .mask1_1_o          (p1_mask1_1_o       ),
        .remain_data_E3_0_0_o(p1_remain_data_E3_0_0_o),
        .remain_data_E3_0_1_o(p1_remain_data_E3_0_1_o),
        .remain_data_E3_1_0_o(p1_remain_data_E3_1_0_o),
        .remain_data_E3_1_1_o(p1_remain_data_E3_1_1_o),
        .vec4_sign_o        (p1_vec4_sign_o     ),
        .E3_MAX0_0_o        (p1_E3_MAX0_0_o     ),
        .E3_MAX0_1_o        (p1_E3_MAX0_1_o     ),
        .E3_MAX1_0_o        (p1_E3_MAX1_0_o     ),
        .E3_MAX1_1_o        (p1_E3_MAX1_1_o     ),
        .E6_MAX0_0_o        (p1_E6_MAX0_0_o     ),
        .E6_MAX0_1_o        (p1_E6_MAX0_1_o     ),
        .E6_MAX1_0_o        (p1_E6_MAX1_0_o     ),
        .E6_MAX1_1_o        (p1_E6_MAX1_1_o     )
    );

    // =========================================================================
    // pipe2_stage → pipe3_stage 连线
    // =========================================================================

    // 解码 C 输出
    wire [7:0]  p2_C_exp_o;
    wire [31:0] p2_C_sig_o;
    wire        p2_C_sign_o;

    // EabMAX 输出
    wire [8:0]  p2_Eab_MAX0_o;
    wire [8:0]  p2_Eab_MAX1_o;

    // scale 最大指数
    wire [8:0]  p2_scale_max_o;

    // 阈值处理输出
    wire        p2_threshold_mxintfp4_en_o;
    wire [9:0]  p2_threshold_normal_exp_0_o;
    wire [9:0]  p2_threshold_normal_exp_1_o;

    wire signed [8:0] p2_scale_exp_s0_o,p2_scale_exp_s1_o;

    // 穿透信号
    wire [5:0]  p2_exp_E6_o      [0:`LANE_NUM-1][0:3];
    wire signed [7:0]  p2_bias_o;
    wire        p2_NAN_o;
    wire        p2_PINF_o;
    wire        p2_NINF_o;
    wire        p2_C_ZERO_o;
    wire [8:0]  p2_scale_exp_o   [0:`Scale_NUM-1];
    wire [7:0]  p2_scale_sig_o   [0:`Scale_NUM-1];
    wire [21:0] p2_product_out_o [0:`LANE_NUM-1][0:1];
    wire [3:0]  p2_vec4_sign_o   [0:`LANE_NUM-1];
    // wire [1:0]  p2_data_type_o;
    wire        p2_is_High_precision_o;
    wire        p2_is_low_precision_o;
    wire        p2_is_MXNVFP4_o;
    wire        p2_is_INT8_o;
    wire        p2_is_INT4_o;
    wire        p2_is_MXINT4_o;
    wire        p2_scale_type_o;

    // =========================================================================
    // 例化 pipe2_stage
    // =========================================================================
    pipe2_stage u_pipe2_stage (
        .clock              (clock              ),
        .reset              (reset              ),

        .NAN_i              (p1_NAN_o           ),
        .PINF_i             (p1_PINF_o          ),
        .NINF_i             (p1_NINF_o          ),
        .CAdd_bits_i        (p1_CAdd_bits_o     ),
        .scale_exp_i        (p1_scale_exp_o     ),
        .scale_sig_i        (p1_scale_sig_o     ),
        .is_High_precision_i(p1_is_High_precision_o),
        .is_low_precision_i (p1_is_low_precision_o ),
        .is_MXNVFP4_i      (p1_is_MXNVFP4_o      ),
        .is_INT8_i          (p1_is_INT8_o          ),
        .is_INT4_i          (p1_is_INT4_o          ),
        .is_MXINT4_i        (p1_is_MXINT4_o        ),
        .scale_type_i       (p1_scale_type_o    ),
        .is_threshold_normal_i(p1_is_threshold_normal_o),
        .is_threshold_mxintfp4_i(p1_is_threshold_mxintfp4_o),
        .bias_i             (p1_bias_o          ),
        .product_out_i      (p1_product_out_o   ),
        .exp_E6_i           (p1_exp_E6_o        ),
        .mask0_0_i          (p1_mask0_0_o       ),
        .mask0_1_i          (p1_mask0_1_o       ),
        .mask1_0_i          (p1_mask1_0_o       ),
        .mask1_1_i          (p1_mask1_1_o       ),
        .remain_data_E3_0_0_i(p1_remain_data_E3_0_0_o),
        .remain_data_E3_0_1_i(p1_remain_data_E3_0_1_o),
        .remain_data_E3_1_0_i(p1_remain_data_E3_1_0_o),
        .remain_data_E3_1_1_i(p1_remain_data_E3_1_1_o),
        .vec4_sign_i        (p1_vec4_sign_o     ),
        .E3_MAX0_0_i        (p1_E3_MAX0_0_o     ),
        .E3_MAX0_1_i        (p1_E3_MAX0_1_o     ),
        .E3_MAX1_0_i        (p1_E3_MAX1_0_o     ),
        .E3_MAX1_1_i        (p1_E3_MAX1_1_o     ),
        .E6_MAX0_0_i        (p1_E6_MAX0_0_o     ),
        .E6_MAX0_1_i        (p1_E6_MAX0_1_o     ),
        .E6_MAX1_0_i        (p1_E6_MAX1_0_o     ),
        .E6_MAX1_1_i        (p1_E6_MAX1_1_o     ),

        .valid1_i           (valid1_o           ),
        .allow2_in_o        (allow2_in          ),
        .valid2_o           (valid2_o           ),
        .allow3_in_i        (allow3_in          ),

        .C_exp_o            (p2_C_exp_o         ),
        .C_sig_o            (p2_C_sig_o         ),
        .C_sign_o           (p2_C_sign_o        ),
        .Eab_MAX0_o         (p2_Eab_MAX0_o      ),
        .Eab_MAX1_o         (p2_Eab_MAX1_o      ),
        .scale_max_o        (p2_scale_max_o     ),
        .threshold_mxintfp4_en_o(p2_threshold_mxintfp4_en_o),
        .threshold_normal_exp_0_o(p2_threshold_normal_exp_0_o),
        .threshold_normal_exp_1_o(p2_threshold_normal_exp_1_o),
        .scale_exp_s0_o     (p2_scale_exp_s0_o  ),
        .scale_exp_s1_o     (p2_scale_exp_s1_o  ),
        .exp_E6_o           (p2_exp_E6_o        ),
        .bias_o             (p2_bias_o          ),
        .NAN_o              (p2_NAN_o           ),
        .PINF_o             (p2_PINF_o          ),
        .NINF_o             (p2_NINF_o          ),
        .C_ZERO_o           (p2_C_ZERO_o        ),
        .scale_exp_o        (p2_scale_exp_o     ),
        .scale_sig_o        (p2_scale_sig_o     ),
        .product_out_o      (p2_product_out_o   ),
        .vec4_sign_o        (p2_vec4_sign_o     ),
        .is_High_precision_o(p2_is_High_precision_o),
        .is_low_precision_o (p2_is_low_precision_o ),
        .is_MXNVFP4_o      (p2_is_MXNVFP4_o      ),
        .is_INT8_o          (p2_is_INT8_o          ),
        .is_INT4_o          (p2_is_INT4_o          ),
        .is_MXINT4_o        (p2_is_MXINT4_o        ),
        .scale_type_o       (p2_scale_type_o    )
    );

    // =========================================================================
    // pipe3_stage → pipe4_stage 连线
    // =========================================================================

    wire signed [9:0]  p3_block_max_o;
    wire [5:0]  p3_block_shift_o  [0:`Scale_NUM-1];
    wire [5:0]  p3_nvc_shift_o;
    wire signed [9:0]  p3_E_MAX_o;
    wire [4:0]  p3_c_shift_o;
    wire [4:0]  p3_exp_shift_out_o [0:`LANE_NUM-1][0:3];

    wire        p3_NAN_o;
    wire        p3_PINF_o;
    wire        p3_NINF_o;
    wire [31:0] p3_C_sig_o;
    wire        p3_C_sign_o;
    wire [7:0]  p3_scale_sig_o    [0:`Scale_NUM-1];
    wire [21:0] p3_product_out_o  [0:`LANE_NUM-1][0:1];
    wire [3:0]  p3_vec4_sign_o    [0:`LANE_NUM-1];
    // wire [1:0]  p3_data_type_o;
    wire        p3_is_High_precision_o;
    wire        p3_is_low_precision_o;
    wire        p3_is_MXNVFP4_o;
    wire        p3_is_INT8_o;
    wire        p3_is_INT4_o;
    wire        p3_is_MXINT4_o;
    wire        p3_is_MXINTFP4_o;

    // =========================================================================
    // 例化 pipe3_stage
    // =========================================================================
    pipe3_stage u_pipe3_stage (
        .clock              (clock              ),
        .reset              (reset              ),

        .C_exp_i            (p2_C_exp_o         ),
        .C_sig_i            (p2_C_sig_o         ),
        .C_sign_i           (p2_C_sign_o        ),
        .C_ZERO_i           (p2_C_ZERO_o        ),
        .Eab_MAX0_i         (p2_Eab_MAX0_o      ),
        .Eab_MAX1_i         (p2_Eab_MAX1_o      ),
        .scale_max_i        (p2_scale_max_o     ),
        .threshold_mxintfp4_en_i(p2_threshold_mxintfp4_en_o),
        .threshold_normal_exp_0_i(p2_threshold_normal_exp_0_o),
        .threshold_normal_exp_1_i(p2_threshold_normal_exp_1_o),

        .scale_exp_s0_i     (p2_scale_exp_s0_o  ),
        .scale_exp_s1_i     (p2_scale_exp_s1_o  ),

        .exp_E6_i           (p2_exp_E6_o        ),
        .bias_i             (p2_bias_o          ),
        .scale_exp_i        (p2_scale_exp_o     ),
        .NAN_i              (p2_NAN_o           ),
        .PINF_i             (p2_PINF_o          ),
        .NINF_i             (p2_NINF_o          ),
        .scale_sig_i        (p2_scale_sig_o     ),
        .product_out_i      (p2_product_out_o   ),
        .vec4_sign_i        (p2_vec4_sign_o     ),
        .is_High_precision_i(p2_is_High_precision_o),
        .is_low_precision_i (p2_is_low_precision_o ),
        .is_MXNVFP4_i      (p2_is_MXNVFP4_o      ),
        .is_INT8_i          (p2_is_INT8_o          ),
        .is_INT4_i          (p2_is_INT4_o          ),
        .is_MXINT4_i        (p2_is_MXINT4_o        ),
        .scale_type_i       (p2_scale_type_o    ),

        .valid2_i           (valid2_o           ),
        .allow3_in_o        (allow3_in          ),
        .valid3_o           (valid3_o           ),
        .allow4_in_i        (allow4_in          ),

        .block_max_o        (p3_block_max_o     ),
        .block_shift_o      (p3_block_shift_o   ),
        .nvc_shift_o        (p3_nvc_shift_o     ),
        .E_MAX_o            (p3_E_MAX_o         ),
        .c_shift_o          (p3_c_shift_o       ),
        .exp_shift_out_o    (p3_exp_shift_out_o ),
        .NAN_o              (p3_NAN_o           ),
        .PINF_o             (p3_PINF_o          ),
        .NINF_o             (p3_NINF_o          ),
        .C_sig_o            (p3_C_sig_o         ),
        .C_sign_o           (p3_C_sign_o        ),
        .scale_sig_o        (p3_scale_sig_o     ),
        .product_out_o      (p3_product_out_o   ),
        .vec4_sign_o        (p3_vec4_sign_o     ),
        .is_High_precision_o(p3_is_High_precision_o),
        .is_low_precision_o (p3_is_low_precision_o ),
        .is_MXNVFP4_o      (p3_is_MXNVFP4_o      ),
        .is_INT8_o          (p3_is_INT8_o          ),
        .is_INT4_o          (p3_is_INT4_o          ),
        .is_MXINT4_o        (p3_is_MXINT4_o        ),
        .is_MXINTFP4_o      (p3_is_MXINTFP4_o      )
    );

    // =========================================================================
    // pipe4_stage → pipe5_stage 连线
    // =========================================================================

    wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0]
                      p4_sign_after_add0_o [0:`P4_ADD0_OUT_NUM-1];

    wire signed [9:0]  p4_block_max_o;
    wire [5:0]  p4_block_shift_o [0:`Scale_NUM-1];
    wire [5:0]  p4_nvc_shift_o;
    wire signed [9:0]  p4_E_MAX_o;
    wire [4:0]  p4_c_shift_o;
    wire        p4_NAN_o;
    wire        p4_PINF_o;
    wire        p4_NINF_o;
    wire [31:0] p4_C_sig_o;
    wire        p4_C_sign_o;
    wire [7:0]  p4_scale_sig_o [0:`Scale_NUM-1];
    // wire [1:0]  p4_data_type_o;
    wire        p4_is_MXNVFP4_o;
    wire        p4_is_INT8_o;
    wire        p4_is_INT4_o;
    wire        p4_is_MXINT4_o;

    // =========================================================================
    // 例化 pipe4_stage
    // =========================================================================
    pipe4_stage u_pipe4_stage (
        .clock              (clock              ),
        .reset              (reset              ),

        .block_max_i        (p3_block_max_o     ),
        .block_shift_i      (p3_block_shift_o   ),
        .nvc_shift_i        (p3_nvc_shift_o     ),
        .E_MAX_i            (p3_E_MAX_o         ),
        .c_shift_i          (p3_c_shift_o       ),
        .exp_shift_out_i    (p3_exp_shift_out_o ),
        .product_out_i      (p3_product_out_o   ),
        .vec4_sign_i        (p3_vec4_sign_o     ),
        .NAN_i              (p3_NAN_o           ),
        .PINF_i             (p3_PINF_o          ),
        .NINF_i             (p3_NINF_o          ),
        .C_sig_i            (p3_C_sig_o         ),
        .C_sign_i           (p3_C_sign_o        ),
        .scale_sig_i        (p3_scale_sig_o     ),
        .is_High_precision_i(p3_is_High_precision_o),
        .is_low_precision_i (p3_is_low_precision_o ),
        .is_MXNVFP4_i      (p3_is_MXNVFP4_o      ),
        .is_INT8_i          (p3_is_INT8_o          ),
        .is_INT4_i          (p3_is_INT4_o          ),
        .is_MXINT4_i        (p3_is_MXINT4_o        ),
        .is_MXINTFP4_i      (p3_is_MXINTFP4_o      ),

        .valid3_i           (valid3_o           ),
        .allow4_in_o        (allow4_in          ),
        .valid4_o           (valid4_o           ),
        .allow5_in_i        (allow5_in          ),

        .sign_after_add0_o  (p4_sign_after_add0_o),
        .block_max_o        (p4_block_max_o     ),
        .block_shift_o      (p4_block_shift_o   ),
        .nvc_shift_o        (p4_nvc_shift_o     ),
        .E_MAX_o            (p4_E_MAX_o         ),
        .c_shift_o          (p4_c_shift_o       ),
        .NAN_o              (p4_NAN_o           ),
        .PINF_o             (p4_PINF_o          ),
        .NINF_o             (p4_NINF_o          ),
        .C_sig_o            (p4_C_sig_o         ),
        .C_sign_o           (p4_C_sign_o        ),
        .scale_sig_o        (p4_scale_sig_o     ),
        .is_MXNVFP4_o      (p4_is_MXNVFP4_o      ),
        .is_INT8_o          (p4_is_INT8_o          ),
        .is_INT4_o          (p4_is_INT4_o          ),
        .is_MXINT4_o        (p4_is_MXINT4_o        )
    );

    // =========================================================================
    // pipe5_stage → pipe6_stage 连线
    // =========================================================================

    wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0]
                      p5_sign_after_add1_o  [0:`P4_ADD1_OUT_NUM-1];
    wire signed [`decimal_places1+`block_int_w-1:0]
                      p5_block_sig_af_shift_o [0:`P4_ADD0_OUT_NUM-1];

    wire signed [9:0]  p5_block_max_o;
    wire [5:0]  p5_nvc_shift_o;
    wire signed [9:0]  p5_E_MAX_o;
    wire [4:0]  p5_c_shift_o;
    wire        p5_NAN_o;
    wire        p5_PINF_o;
    wire        p5_NINF_o;
    wire [31:0] p5_C_sig_o;
    wire        p5_C_sign_o;
    // wire [1:0]  p5_data_type_o;
    wire        p5_is_MXINTFP4_o;
    wire        p5_is_INT_o;

    // =========================================================================
    // 例化 pipe5_stage
    // =========================================================================
    pipe5_stage u_pipe5_stage (
        .clock                  (clock                  ),
        .reset                  (reset                  ),

        .sign_after_add0_i      (p4_sign_after_add0_o   ),
        .block_shift_i          (p4_block_shift_o       ),
        .scale_sig_i            (p4_scale_sig_o         ),
        .block_max_i            (p4_block_max_o         ),
        .nvc_shift_i            (p4_nvc_shift_o         ),
        .E_MAX_i                (p4_E_MAX_o             ),
        .c_shift_i              (p4_c_shift_o           ),
        .NAN_i                  (p4_NAN_o               ),
        .PINF_i                 (p4_PINF_o              ),
        .NINF_i                 (p4_NINF_o              ),
        .C_sig_i                (p4_C_sig_o             ),
        .C_sign_i               (p4_C_sign_o            ),
        .is_MXNVFP4_i          (p4_is_MXNVFP4_o       ),
        .is_INT8_i              (p4_is_INT8_o           ),
        .is_INT4_i              (p4_is_INT4_o           ),
        .is_MXINT4_i            (p4_is_MXINT4_o         ),

        .valid4_i               (valid4_o               ),
        .allow5_in_o            (allow5_in              ),
        .valid5_o               (valid5_o               ),
        .allow6_in_i            (allow6_in              ),

        .sign_after_add1_o      (p5_sign_after_add1_o   ),
        .block_sig_af_shift_o   (p5_block_sig_af_shift_o),
        .block_max_o            (p5_block_max_o         ),
        .nvc_shift_o            (p5_nvc_shift_o         ),
        .E_MAX_o                (p5_E_MAX_o             ),
        .c_shift_o              (p5_c_shift_o           ),
        .NAN_o                  (p5_NAN_o               ),
        .PINF_o                 (p5_PINF_o              ),
        .NINF_o                 (p5_NINF_o              ),
        .C_sig_o                (p5_C_sig_o             ),
        .C_sign_o               (p5_C_sign_o            ),
        .is_MXINTFP4_o          (p5_is_MXINTFP4_o      ),
        .is_INT_o               (p5_is_INT_o            )
    );

    // =========================================================================
    // pipe6_stage → pipe7_stage 连线
    // =========================================================================

    wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0]
                      p6_sign_fp8_o;
    wire signed [`decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM):0]
                      p6_nvmxfp4_sig_o;

    wire signed [9:0]  p6_E_MAX_o;
    wire signed [9:0]  p6_block_max_o;
    wire        p6_NAN_o;
    wire        p6_PINF_o;
    wire        p6_NINF_o;
    // wire [1:0]  p6_data_type_o;
    wire        p6_is_MXINTFP4_o;
    wire        p6_is_INT_o;

    // =========================================================================
    // 例化 pipe6_stage
    // =========================================================================
    pipe6_stage u_pipe6_stage (
        .clock                  (clock                  ),
        .reset                  (reset                  ),

        .E_MAX_i                (p5_E_MAX_o             ),
        .block_max_i            (p5_block_max_o         ),
        .nvc_shift_i            (p5_nvc_shift_o         ),
        .c_shift_i              (p5_c_shift_o           ),
        .C_sig_i                (p5_C_sig_o             ),
        .C_sign_i               (p5_C_sign_o            ),
        .NAN_i                  (p5_NAN_o               ),
        .PINF_i                 (p5_PINF_o              ),
        .NINF_i                 (p5_NINF_o              ),
        .is_MXINTFP4_i          (p5_is_MXINTFP4_o      ),
        .is_INT_i               (p5_is_INT_o            ),
        .sign_after_add1_i      (p5_sign_after_add1_o   ),
        .block_sig_af_shift_i   (p5_block_sig_af_shift_o),

        .valid5_i               (valid5_o               ),
        .allow6_in_o            (allow6_in              ),
        .valid6_o               (valid6_o               ),
        .allow7_in_i            (allow7_in              ),

        .sign_fp8_o             (p6_sign_fp8_o          ),
        .nvmxfp4_sig_o          (p6_nvmxfp4_sig_o       ),
        .E_MAX_o                (p6_E_MAX_o             ),
        .block_max_o            (p6_block_max_o         ),
        .NAN_o                  (p6_NAN_o               ),
        .PINF_o                 (p6_PINF_o              ),
        .NINF_o                 (p6_NINF_o              ),
        .is_MXINTFP4_o          (p6_is_MXINTFP4_o      ),
        .is_INT_o               (p6_is_INT_o            )
    );

    // =========================================================================
    // pipe7_stage → pipe8_stage 连线
    // =========================================================================

    wire        p7_NAN_o;
    wire        p7_PINF_o;
    wire        p7_NINF_o;
    // wire [1:0]  p7_data_type_o;
    wire        p7_is_INT_o;
    wire [31:0] p7_INT32_o;
    wire        p7_fp32_sign_o;
    wire [23:0] p7_fp32_sig_o;
    wire signed [9:0]  p7_fp32_exp_o;

    // =========================================================================
    // 例化 pipe7_stage
    // =========================================================================
    pipe7_stage u_pipe7_stage (
        .clock          (clock              ),
        .reset          (reset              ),

        .sign_fp8_i     (p6_sign_fp8_o      ),
        .nvmxfp4_sig_i  (p6_nvmxfp4_sig_o   ),
        .E_MAX_i        (p6_E_MAX_o         ),
        .block_max_i    (p6_block_max_o     ),
        .is_MXINTFP4_i (p6_is_MXINTFP4_o  ),
        .is_INT_i      (p6_is_INT_o        ),
        .NAN_i          (p6_NAN_o           ),
        .PINF_i         (p6_PINF_o          ),
        .NINF_i         (p6_NINF_o          ),

        .valid6_i       (valid6_o           ),
        .allow7_in_o    (allow7_in          ),
        .valid7_o       (valid7_o           ),
        .allow8_in_i    (allow8_in          ),

        .NAN_o          (p7_NAN_o           ),
        .PINF_o         (p7_PINF_o          ),
        .NINF_o         (p7_NINF_o          ),
        .is_INT_o       (p7_is_INT_o        ),
        .INT32_o        (p7_INT32_o         ),
        .fp32_sign_o    (p7_fp32_sign_o     ),
        .fp32_sig_o     (p7_fp32_sig_o      ),
        .fp32_exp_o     (p7_fp32_exp_o      )
    );

    // =========================================================================
    // 例化 pipe8_stage
    // =========================================================================
    pipe8_stage u_pipe8_stage (
        .clock          (clock              ),
        .reset          (reset              ),

        .NAN_i          (p7_NAN_o           ),
        .PINF_i         (p7_PINF_o          ),
        .NINF_i         (p7_NINF_o          ),
        .is_INT_i       (p7_is_INT_o        ),
        .INT32_i        (p7_INT32_o         ),
        .fp32_sign_i    (p7_fp32_sign_o     ),
        .fp32_sig_i     (p7_fp32_sig_o      ),
        .fp32_exp_i     (p7_fp32_exp_o      ),

        .valid7_i       (valid7_o           ),
        .allow8_in_o    (allow8_in          ),
        .valid8_o       (valid8_o           ),
        .allow9_in_i    (allow9_in_i        ),

        .D_bits_o       (D_bits_o           )
    );

endmodule
