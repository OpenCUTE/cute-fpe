`include "defines.sv"

module shift1_stage (
    // ---------- NV-MXF P4 scale 相关 ----------
    input  wire        scale_type,                              // SCALE_UE8M0 / SCALE_E4M3
    input  wire [8:0]  scale_exp  [0:`Scale_NUM - 1],          // 每个 scale block 的指数
    input  wire [8:0]  scale_max,                              // scale 域最大指数

    input  wire  threshold_mxintfp4_en,
    input  wire [9:0]  threshold_normal_exp_0, threshold_normal_exp_1,

    input wire signed [8:0] scale_exp_s0,scale_exp_s1,

    // ---------- C（累加器）指数 ----------
    input  wire [7:0]  C_exp,                                  // FP32 偏置后指数
    input  wire        C_ZERO,
    input  wire signed [7:0]  bias,                                   // 数据类型对应 bias

    // ---------- AB 最大指数（两半） ----------
    input  wire [8:0]  Eab_MAX0,                               // 前半 BLOCK_NUM 个 lane 的 AB 最大指数
    input  wire [8:0]  Eab_MAX1,                               // 后半 lane 的 AB 最大指数

    // ---------- 输入：各 lane 的 6-bit 指数 ----------
    input  wire [5:0]  exp_E6_i   [0:`LANE_NUM - 1][0:3],   // 每个 lane 4 个元素的 6-bit 指数

    // ---------- 数据类型 ----------
    // input  wire [1:0]  data_type_i,
    input  wire        is_High_precision_i,
    input  wire        is_low_precision_i,
    input  wire        is_INT8_i,

    // ---------- 输出：nvmxfp4_shift ----------
    output wire signed [9:0]  block_max,                              // FP32 域最大指数
    output wire [5:0]  block_shift [0:`Scale_NUM - 1],         // 每个 scale block 的移位量
    output wire [5:0]  nvc_shift,                              // C 相对于 scale 最大值的移位量

    // ---------- 输出：expmax ----------
    output wire signed [9:0]  E_MAX,                                  // FP32 域综合最大指数
    output wire [4:0]  c_shift,                                // C 相对于 AB 最大值的移位量

    // ---------- 输出：各 lane 的元素移位量 ----------
    output wire [4:0]  exp_shift_out [0:`LANE_NUM - 1][0:3]
);

    // =========================================================
    // 内部线网：expmax 输出
    // =========================================================
    wire [9:0] E_MAX_shift0;
    wire [9:0] E_MAX_shift1;

    // =========================================================
    // 得到 scale 的移位值
    // =========================================================
    nvmxfp4_shift u_nvmxfp4_shift (
        .scale_type    (scale_type),
        .scale_exp     (scale_exp),

        .threshold_mxintfp4_en_i (threshold_mxintfp4_en),

        .scale_max_exp (scale_max),
        .C_exp         (C_exp),
        .C_ZERO        (C_ZERO),

        .block_max     (block_max),
        .block_shift   (block_shift),
        .c_shift       (nvc_shift)
    );

    // =========================================================
    // 将 AB 最大指数值与 C 的最大指数值相比较，得到对应输出
    // =========================================================
    expmax u_expmax (
        .C_exp         (C_exp),
        .C_ZERO        (C_ZERO),
        .bias          (bias),

        .threshold_normal_exp_0 (threshold_normal_exp_0),
        .threshold_normal_exp_1 (threshold_normal_exp_1),
        
        .scale_exp_s0    (scale_exp_s0),
        .scale_exp_s1    (scale_exp_s1),

        .Eab_MAX0      (Eab_MAX0),
        .Eab_MAX1      (Eab_MAX1),

        .E_MAX         (E_MAX),
        .E_MAX_shift0  (E_MAX_shift0),
        .E_MAX_shift1  (E_MAX_shift1),
        .c_shift       (c_shift)
    );

    // =========================================================
    // 按 lane 计算各元素的移位量（分前后两半）
    // =========================================================
    genvar i;

    generate
        // --- 前半：lane 0 ~ BLOCK_NUM-1，使用 E_MAX_shift0 ---
        for (i = 0; i < `BLOCK_NUM; i = i + 1) begin : GEN_SHIFT_BLOCK0

            wire [4:0] exp_shift_w [0:3];

            exp_shift_unit u_exp_shift_unit (
                .E_MAX_shift (E_MAX_shift0),
                .is_High_precision(is_High_precision_i),
                .is_low_precision (is_low_precision_i ),
                .is_INT8          (is_INT8_i          ),
                .exp_E6      (exp_E6_i[i]),
                .exp_shift   (exp_shift_w)
            );

            // 将结果引出到顶层输出
            assign exp_shift_out[i][0] = exp_shift_w[0];
            assign exp_shift_out[i][1] = exp_shift_w[1];
            assign exp_shift_out[i][2] = exp_shift_w[2];
            assign exp_shift_out[i][3] = exp_shift_w[3];
        end

        // --- 后半：lane BLOCK_NUM ~ LANE_NUM-1，使用 E_MAX_shift1 ---
        for (i = `BLOCK_NUM; i < `LANE_NUM; i = i + 1) begin : GEN_SHIFT_BLOCK1

            wire [4:0] exp_shift_w [0:3];

            exp_shift_unit u_exp_shift_unit (
                .E_MAX_shift (E_MAX_shift1),
                .is_High_precision(is_High_precision_i),
                .is_low_precision (is_low_precision_i ),
                .is_INT8          (is_INT8_i          ),
                .exp_E6      (exp_E6_i[i]),
                .exp_shift   (exp_shift_w)
            );

            assign exp_shift_out[i][0] = exp_shift_w[0];
            assign exp_shift_out[i][1] = exp_shift_w[1];
            assign exp_shift_out[i][2] = exp_shift_w[2];
            assign exp_shift_out[i][3] = exp_shift_w[3];
        end
    endgenerate

endmodule
