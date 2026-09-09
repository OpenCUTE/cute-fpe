`include "defines.sv"

module mul_stage (
    input wire  [`ScaleBusWidth - 1 : 0] AScale_bits,
    input wire  [`ScaleBusWidth - 1 : 0] BScale_bits,

    input wire  [3:0]  opcode,

    input wire  is_High_precision,
    input wire  is_low_precision,
    input wire  is_INT8,
    input wire  is_INTFP4,
    input wire  is_MXINTFP4,

    // 尾数乘法器输入（每通道 2 组）
    input wire  [7:0] a_8 [0 : `LANE_NUM-1][0:1],
    input wire  [3:0] a_4 [0 : `LANE_NUM-1][0:1],
    input wire  [3:0] b_4 [0 : `LANE_NUM-1][0:1],
    input wire  [7:0] b_8 [0 : `LANE_NUM-1][0:1],


    // 指数求和结果
    input wire  [5:0] exp_E6 [0:`LANE_NUM-1][0:3],

    input wire [`Scale_NUM-1:0] MXINTFP4_all_zero,
    input wire [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero,

    // 符号位向量
    input wire  [7:0] vec8_sign [0:`LANE_NUM-1],

    //输出信号
    //解码scale
    output wire  [8:0] scale_exp [0:`Scale_NUM - 1] ,
    output wire  [7:0] scale_sig [0:`Scale_NUM - 1] ,
    output wire        scale_NAN       ,

    //标志位输出
    // output wire [1:0] data_type ,
    output wire       scale_type,
    output wire       is_threshold_normal,
    output wire       is_threshold_mxintfp4,
    output wire signed [7:0] bias,

    output wire [21:0] product_out [0:`LANE_NUM-1][0:1],
    output wire [5 :0] exp_E6_o    [0:`LANE_NUM-1][0:3],

    //掩码输出
    output logic [`BLOCK_NUM -1: 0]  mask0_0, mask0_1        ,
    output logic [`BLOCK_NUM -1: 0]  remain_data_E3_0_0 [0:2],
    output logic [`BLOCK_NUM -1: 0]  remain_data_E3_0_1 [0:2],
    output logic [`BLOCK_NUM -1: 0]  mask1_0, mask1_1        ,
    output logic [`BLOCK_NUM -1: 0]  remain_data_E3_1_0 [0:2],
    output logic [`BLOCK_NUM -1: 0]  remain_data_E3_1_1 [0:2],

    output wire  [2:0] E3_MAX0_0, E3_MAX0_1,
    output wire  [5:0] E6_MAX0_0, E6_MAX0_1,
    output wire  [2:0] E3_MAX1_0, E3_MAX1_1,
    output wire  [5:0] E6_MAX1_0, E6_MAX1_1
);

// =========================================================================
// 内部连线声明
// =========================================================================
//opcode 输出
wire scale_en;

logic [`BLOCK_NUM -1: 0]  mask_E3_0_0, mask_E3_0_1, mask_E6_0_0, mask_E6_0_1;
logic [`BLOCK_NUM -1: 0]  mask_E3_1_0, mask_E3_1_1, mask_E6_1_0, mask_E6_1_1;

//现在我们将这个比较树分成两半，一半是block0，一半是block1
// CmpTree 输入：需要把二维数组的某一维度拆出来
// exp_E9[lane][0] 和 exp_E9[lane][1] 分别喂给两棵 E9 树
// block0
wire [5:0] exp_E6_ch0_0 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch0_1 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch0_2 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch0_3 [0:`BLOCK_NUM-1];

//block1
wire [5:0] exp_E6_ch1_0 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch1_1 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch1_2 [0:`BLOCK_NUM-1];
wire [5:0] exp_E6_ch1_3 [0:`BLOCK_NUM-1];

genvar k;// 用于 generate
generate
    for (k = 0; k < `BLOCK_NUM; k = k + 1) begin : gen_exp_slice_0
        assign exp_E6_ch0_0[k] = exp_E6[k][0];
        assign exp_E6_ch0_1[k] = exp_E6[k][1];
        assign exp_E6_ch0_2[k] = exp_E6[k][2];
        assign exp_E6_ch0_3[k] = exp_E6[k][3];
    end
    for(k = `BLOCK_NUM; k < `LANE_NUM; k = k + 1)begin: gen_exp_slice_1
        assign exp_E6_ch1_0[k-`BLOCK_NUM] = exp_E6[k][0];
        assign exp_E6_ch1_1[k-`BLOCK_NUM] = exp_E6[k][1];
        assign exp_E6_ch1_2[k-`BLOCK_NUM] = exp_E6[k][2];
        assign exp_E6_ch1_3[k-`BLOCK_NUM] = exp_E6[k][3];   
    end
endgenerate


// =========================================================================
// 模块例化
// =========================================================================


// 指数比较树：从 BLOCK_NUM 个通道里找各自最大指数
//E6_1-E6_0中取前三层（其可能是低精度的前三层，也可能是高精度时的0）

//block0
CmpTreeP6_3 #(
    .LAYERS   (3        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E3_0_0 (
    .in     (exp_E6_ch0_0),
    .part1  (mask_E3_0_0),
    .part2  (remain_data_E3_0_0),
    .part3  (E3_MAX0_0)
);

CmpTreeP6_3 #(
    .LAYERS   (3        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E3_0_1 (
    .in     (exp_E6_ch0_1),
    .part1  (mask_E3_0_1),
    .part2  (remain_data_E3_0_1),
    .part3  (E3_MAX0_1)
);

//对于低精度而言，直接是E6_3/E6_2的最终结果，对于高精度而言，是E9的高六位的结果
CmpTreeP9_6 #(
    .LAYERS   (6        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E6_0_0 (
    .in     (exp_E6_ch0_2),
    .part1  (mask_E6_0_0),
    .part3  (E6_MAX0_0)
);

CmpTreeP9_6 #(
    .LAYERS   (6        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E6_0_1 (
    .in     (exp_E6_ch0_3),
    .part1  (mask_E6_0_1),
    .part3  (E6_MAX0_1)
);

// 分别从E3和E6中挑选最大值
expmax_choose u_expmax_choose0 (
    // .data_type(data_type[0]),
    .is_High_precision(is_High_precision),

    .mask_E3_0 (mask_E3_0_0),
    .mask_E3_1 (mask_E3_0_1),
    .mask_E6_0 (mask_E6_0_0),
    .mask_E6_1 (mask_E6_0_1),

    .mask_0  (mask0_0),
    .mask_1  (mask0_1)
);

//block1
CmpTreeP6_3 #(
    .LAYERS   (3        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E3_1_2 (
    .in     (exp_E6_ch1_0),
    .part1  (mask_E3_1_0),
    .part2  (remain_data_E3_1_0),
    .part3  (E3_MAX1_0)
);

CmpTreeP6_3 #(
    .LAYERS   (3        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E3_1_3 (
    .in     (exp_E6_ch1_1),
    .part1  (mask_E3_1_1),
    .part2  (remain_data_E3_1_1),
    .part3  (E3_MAX1_1)
);

CmpTreeP9_6 #(
    .LAYERS   (6        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E6_1_2 (
    .in     (exp_E6_ch1_2),
    .part1  (mask_E6_1_0),
    .part3  (E6_MAX1_0)
);

CmpTreeP9_6 #(
    .LAYERS   (6        ),
    .ELEM_NUM (`BLOCK_NUM),
    .EXP_W    (6        )
) u_CmpTree_E6_1_3 (
    .in     (exp_E6_ch1_3),
    .part1  (mask_E6_1_1),
    .part3  (E6_MAX1_1)
);

expmax_choose u_expmax_choose1 (
    // .data_type(data_type[0]),
    .is_High_precision(is_High_precision),
    
    .mask_E3_0 (mask_E3_1_0),
    .mask_E3_1 (mask_E3_1_1),
    .mask_E6_0 (mask_E6_1_0),
    .mask_E6_1 (mask_E6_1_1),

    .mask_0  (mask1_0),
    .mask_1  (mask1_1)
);

// 操作码解码
decode_opcode1 u_decode_opcode1 (
    .opcode     (opcode    ),
    .is_mxintfp4 (is_MXINTFP4),
    // .data_type  (data_type ),
    .scale_type (scale_type),
    .scale_en   (scale_en  ),

    .is_threshold_normal    (is_threshold_normal),
    .is_threshold_mxintfp4  (is_threshold_mxintfp4),

    .bias       (bias      )
);

//解码scale（A和B）
scale_deal u_scale_deal (
    .AScale_bits(AScale_bits),
    .BScale_bits(BScale_bits),

    .scale_type (scale_type ),
    .scale_en   (scale_en   ),

    .is_MXINTFP4 (is_MXINTFP4),

    .MXINTFP4_all_zero(MXINTFP4_all_zero),
    .MXFP86_all_zero  (MXFP86_all_zero  ),

    .scale_exp  (scale_exp  ),
    .scale_sig  (scale_sig  ),
    .scale_NAN  (scale_NAN  )
);


// 尾数处理单元：generate 重复 LANE_NUM 次
genvar lane;
generate
    for (lane = 0; lane < `LANE_NUM; lane = lane + 1) begin : gen_sig_deal

        wire [7 :0] a_84[0:1], a_88[0:1];
        wire [3 :0] a_48[0:1], a_44[0:1];
        wire [3 :0] b_84[0:1], b_44[0:1];
        wire [7 :0] b_48[0:1], b_88[0:1];

        //解码尾数寄存器
        decode_sig u_decode_sig (
            .is_High_precision (is_High_precision),
            .is_low_precision  (is_low_precision ),
            .is_INTFP4         (is_INTFP4        ),

            .a_8        (a_8[lane]          ),
            .a_4        (a_4[lane]          ),
            .b_4        (b_4[lane]          ),
            .b_8        (b_8[lane]          ),

            .exp_E6     (exp_E6[lane]       ),

            .a_84        (a_84),
            .a_88        (a_88),
            .a_48        (a_48),
            .a_44        (a_44),
            .b_84        (b_84),
            .b_44        (b_44),
            .b_48        (b_48),
            .b_88        (b_88)
        );


        sig_deal_unit u_sig_deal_unit (
            .a_84        (a_84),
            .a_88        (a_88),
            .a_48        (a_48),
            .a_44        (a_44),
            .b_84        (b_84),
            .b_44        (b_44),
            .b_48        (b_48),
            .b_88        (b_88),
            .sign        (vec8_sign  [lane]),
            // .data_type   (data_type       ),
            .is_High_precision (is_High_precision),
            // .is_low_precision  (is_low_precision ),
            .is_INTFP4         (is_INTFP4        ),
            .is_INT8           (is_INT8          ),

            .exp_E6      (exp_E6[lane]       ),
            .product_out (product_out[lane]),
            .exp_E6_o   (exp_E6_o[lane]       )
        );
    end
endgenerate

    
endmodule
