`include "defines.sv"

module shift_stage (
    // scale 解码结果
    input wire  [8:0]  scale_exp [0:`Scale_NUM - 1],

    // 控制标志
    // input wire  [1:0]  data_type,
    input wire         is_High_precision,
    input wire         is_INT8, is_INT4,

    input wire  [`ResultWidth - 1 : 0] CAdd_bits,

    input wire signed [7:0] bias,

    input wire         is_threshold_normal,
    input wire         is_threshold_mxintfp4,

    // 掩码输出
    //blcok0
    input wire [`BLOCK_NUM-1:0]  mask0_0,
    input wire [`BLOCK_NUM-1:0]  mask0_1,
    input wire [`BLOCK_NUM-1:0]  remain_data_E3_0_0 [0:2],
    input wire [`BLOCK_NUM-1:0]  remain_data_E3_0_1 [0:2],

    input wire [2:0] E3_MAX0_0,
    input wire [2:0] E3_MAX0_1,
    input wire [5:0] E6_MAX0_0,
    input wire [5:0] E6_MAX0_1,

    //block1
    input wire [`BLOCK_NUM-1:0]  mask1_0,
    input wire [`BLOCK_NUM-1:0]  mask1_1,
    input wire [`BLOCK_NUM-1:0]  remain_data_E3_1_0 [0:2],
    input wire [`BLOCK_NUM-1:0]  remain_data_E3_1_1 [0:2],

    input wire [2:0] E3_MAX1_0,
    input wire [2:0] E3_MAX1_1,
    input wire [5:0] E6_MAX1_0,
    input wire [5:0] E6_MAX1_1,


    //输出相关的结果

    //解码C
    output wire [7 :0] C_exp,
    output wire [31:0] C_sig,
    output wire        C_sign,
    output wire C_NAN, C_NINF, C_PINF, C_ZERO,

    // EabMAX_choose 输出
    output wire [8:0] Eab_MAX0, Eab_MAX1,

    // scale 最大指数搜索结果
    output wire [8:0] scale_max,

    //输出阈值情况
    output wire threshold_mxintfp4_en,
    output wire [9:0] threshold_normal_exp_0, threshold_normal_exp_1
);
    //block0

    // CmpTreeP1 输出：剩余最大指数位
    logic [2:0] E3_later0_0;   // E3_MAX for group0 remaining bits
    logic [2:0] E3_later0_1;   // E3_MAX for group1 remaining bits

    // =========================================================
    // 进行剩下两个 E3 的比较
    // =========================================================

    CmpTreeP1 #(
        .LAYERS   (6),
        .ELEM_NUM (`BLOCK_NUM),
        .EXP_W    (9)
    ) CmpTreeP1_E0_30 (
        .part1 (mask0_0),
        .part2 (remain_data_E3_0_0),
        .out   (E3_later0_0)
    );

    CmpTreeP1 #(
        .LAYERS   (6),
        .ELEM_NUM (`BLOCK_NUM),
        .EXP_W    (9)
    ) CmpTreeP1_E0_31 (
        .part1 (mask0_1),
        .part2 (remain_data_E3_0_1),
        .out   (E3_later0_1)
    );

    // =========================================================
    // 将上一级数据与这一级计算出来的最大指数剩余位合并，
    // 得到 ab 向量的最大指数值
    // =========================================================

    EabMAX_choose u_EabMAX_choose0 (
        .E6_MAX0_in  (E6_MAX0_0),
        .E6_MAX1_in  (E6_MAX0_1),
        .E3_MAX0_in  (E3_MAX0_0),
        .E3_MAX1_in  (E3_MAX0_1),
        .E_MAX_later_0 (E3_later0_0),
        .E_MAX_later_1 (E3_later0_1),
        .is_High_precision(is_High_precision),
        .Eab_MAX     (Eab_MAX0)
    );

    //block1

    // CmpTreeP1 输出：剩余最大指数位
    logic [2:0] E3_later1_0;   // E3_MAX for group0 remaining bits
    logic [2:0] E3_later1_1;   // E3_MAX for group1 remaining bits

    // =========================================================
    // 进行剩下两个 E3 的比较
    // =========================================================

    CmpTreeP1 #(
        .LAYERS   (6),
        .ELEM_NUM (`BLOCK_NUM),
        .EXP_W    (9)
    ) CmpTreeP1_E1_30 (
        .part1 (mask1_0),
        .part2 (remain_data_E3_1_0),
        .out   (E3_later1_0)
    );

    CmpTreeP1 #(
        .LAYERS   (6),
        .ELEM_NUM (`BLOCK_NUM),
        .EXP_W    (9)
    ) CmpTreeP1_E1_31 (
        .part1 (mask1_1),
        .part2 (remain_data_E3_1_1),
        .out   (E3_later1_1)
    );

    // =========================================================
    // 将上一级数据与这一级计算出来的最大指数剩余位合并，
    // 得到 ab 向量的最大指数值
    // =========================================================

    EabMAX_choose u_EabMAX_choose1 (
        .E6_MAX0_in  (E6_MAX1_0),
        .E6_MAX1_in  (E6_MAX1_1),
        .E3_MAX0_in  (E3_MAX1_0),
        .E3_MAX1_in  (E3_MAX1_1),
        .E_MAX_later_0 (E3_later1_0),
        .E_MAX_later_1 (E3_later1_1),
        .is_High_precision(is_High_precision),
        .Eab_MAX     (Eab_MAX1)
    );



    // =========================================================
    // 进行 scale 的指数中寻找最大值
    // =========================================================  

    CmpTreeP63_96 #(
        .LAYERS   (9),
        .ELEM_NUM (`Scale_NUM),
        .EXP_W    (9)
    ) u_CmpTreeP63_96 (
        .in    (scale_exp),
        .part3 (scale_max)
    );

    //解码C
    decode_C u_decode_C (
        .is_INT8 (is_INT8  ),
        .is_INT4 (is_INT4  ),
        .C_bits  (CAdd_bits),
        .C_exp   (C_exp    ),
        .C_sig   (C_sig    ),
        .C_sign  (C_sign   ),
        .C_NAN   (C_NAN    ),
        .C_NINF  (C_NINF   ),
        .C_PINF  (C_PINF   ),
        .C_ZERO  (C_ZERO   )
    );

    threshold_deal u_threshold_deal (
        .is_threshold_normal (is_threshold_normal),
        .is_threshold_mxintfp4  (is_threshold_mxintfp4),

        .C_ZERO              (C_ZERO   ),

        .bias                (bias     ),

        //这个是对于mxfp4的情况而言的，其只与scale_max有关
        // .scale_max           (scale_max),
        //这个是对于normal的情况而言的，其每32个元素共用一个scale_exp
        .scale_exp_0         (scale_exp[0]),
        .scale_exp_1         (scale_exp[`Scale_NUM == 8 ? 2 : 1]),

        //对于mxfp4而言，只需要输出一个使能信号即可（不过如果时序紧张，可以挪移到下一级）
        .threshold_mxintfp4_en  (threshold_mxintfp4_en),
        //对于normal而言，输出临时比较的结果，后续还需要进行比较
        .threshold_normal_exp_0 (threshold_normal_exp_0),
        .threshold_normal_exp_1 (threshold_normal_exp_1)
    );

endmodule
