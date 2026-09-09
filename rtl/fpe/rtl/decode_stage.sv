`include "defines.sv"

module decode_stage (

    // A 向量输入
    input  wire [`ReduceWidth - 1 : 0]  AVector_bits,

    // B 向量输入
    input  wire [`ReduceWidth - 1 : 0]  BVector_bits,
    input  wire [4:0]                   opcode      ,

    // -------- 输出（每通道独立，以数组形式输出）--------

    // 异常标志
    output wire       NAN ,
    output wire       PINF,
    output wire       NINF,

    output wire [3:0] opcode_o,

    output wire       is_High_precision_o,
    output wire       is_low_precision_o,
    output wire       is_MXNVFP4_o,
    output wire       is_INT8_o,
    output wire       is_INT4_o,
    output wire       is_MXINT4_o,
    output wire       is_INTFP4_o,
    output wire       is_MXINTFP4_o,
    output wire       is_MXFP8_6_o,

    // 尾数乘法器输入（每通道 2 组）
    output wire [7:0] a_8 [0: `LANE_NUM-1][0:1],
    output wire [3:0] a_4 [0: `LANE_NUM-1][0:1],
    output wire [3:0] b_4 [0: `LANE_NUM-1][0:1],
    output wire [7:0] b_8 [0: `LANE_NUM-1][0:1],

    // 指数求和结果
    output wire [5:0] exp_E6 [0:`LANE_NUM-1][0:3],

    //fp4全为0的标志
    output wire [`Scale_NUM-1:0] MXINTFP4_all_zero,
    //mxfp8/fp6全是0的标志
    output wire [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero,

    // 符号位向量
    output wire [7:0] vec8_sign [0:`LANE_NUM-1]
);

    // 操作码解码信号（由上层 decode_opcode 产生，广播给所有通道）
    wire is_TF32, is_BF16, is_FP16 ;
    wire is_E5M2, is_E3M2, is_E4M3 ;
    wire is_E2,   is_Int8, is_E8   ;
    wire is_fp16_bf16 ;
    wire is_mxnvfp4                ;
    wire is_Int4                   ;
    wire is_int                    ;
    wire is_intfp4                 ;
    wire is_mxint4                 ;
    wire is_mxfp8_6, is_mxfp8      ;
    wire AS_have, BS_have          ;

    // 异常标志
    wire [`LANE_NUM-1:0] NAN_o      ;
    wire [`LANE_NUM-1:0] PINF_o     ;
    wire [`LANE_NUM-1:0] NINF_o     ;

    assign is_MXINTFP4_o = is_mxint4 || is_mxnvfp4;
    assign is_MXFP8_6_o  = is_mxfp8_6;

    // FP4 全是零的标志
    wire [`LANE_NUM-1:0] MXINTFP4_all_zero_temp;

    //每两个MXINTFP4_all_zero_temp合并一下
    genvar fp4_zero_idx;
    generate
        for (fp4_zero_idx = 0; fp4_zero_idx < `Scale_NUM; fp4_zero_idx = fp4_zero_idx + 1) begin : gen_mxintfp4_all_zero
            assign MXINTFP4_all_zero[fp4_zero_idx] = MXINTFP4_all_zero_temp[fp4_zero_idx*2] &
                                                     MXINTFP4_all_zero_temp[fp4_zero_idx*2 + 1];
        end
    endgenerate

    // mxfp8/fp6全是零点标志
    wire [`LANE_NUM-1:0] MXFP86_all_zero_temp;

    genvar mxfp86_zero_idx;
    generate
        for (mxfp86_zero_idx = 0;
             mxfp86_zero_idx < `MXFP86_ZERO_GROUP_NUM;
             mxfp86_zero_idx = mxfp86_zero_idx + 1) begin : gen_mxfp86_all_zero
            assign MXFP86_all_zero[mxfp86_zero_idx] = is_mxfp8_6 ? &MXFP86_all_zero_temp[ mxfp86_zero_idx * `MXFP86_ZERO_GROUP_SIZE +:`MXFP86_ZERO_GROUP_SIZE] :
                                                                   1'd0;
        end
    endgenerate

    ABexcep_summ u_ABexcep_summ(
        .NAN_IN     (NAN_o  ),
        .PINF_IN    (PINF_o ),
        .NINF_IN    (NINF_o ),
        .NAN        (NAN    ),
        .PINF       (PINF   ),
        .NINF       (NINF   )
    );

    decode_opcode u_decode_opcode(
        .opcode        (opcode      ),

        .opcode_o      (opcode_o    ),

        .is_TF32       (is_TF32     ), 
        .is_BF16       (is_BF16     ), 
        .is_FP16       (is_FP16     ), 
        .is_E5M2       (is_E5M2     ), 
        .is_E3M2       (is_E3M2     ), 
        .is_E4M3       (is_E4M3     ), 
        .is_E2         (is_E2       ), 
        .is_Int8       (is_Int8     ), 
        .is_E8         (is_E8       ),
        .is_fp16_bf16  (is_fp16_bf16),
        .is_mxnvfp4    (is_mxnvfp4  ), 
        .is_Int4       (is_Int4     ),
        .is_int        (is_int      ),
        .is_intfp4     (is_intfp4   ),
        .is_mxint4     (is_mxint4   ),
        .is_mxfp8_6    (is_mxfp8_6  ),
        .is_mxfp8      (is_mxfp8    ),

        .is_High_precision_o (is_High_precision_o),
        .is_low_precision_o  (is_low_precision_o ),
        .is_MXNVFP4_o        (is_MXNVFP4_o       ),
        .is_INT8_o           (is_INT8_o          ),
        .is_INT4_o           (is_INT4_o          ),
        .is_MXINT4_o         (is_MXINT4_o        ),
        .is_INTFP4_o         (is_INTFP4_o        ),

        .AS_have       (AS_have     ), 
        .BS_have       (BS_have     )// 1: s8/s4; 0: u8/u4
    );


    // ----------------------------------------------------------------
    // generate：并行例化 LANE_NUM 个 decode_data_unit
    // 每个实例处理 32bit 的 A 切片和 32bit 的 B 切片
    // ----------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < `LANE_NUM; i = i + 1) begin : g_decode_lane

            decode_data_unit u_decode_data_unit (
                // 向量切片：每次取 32bit
                .AVector_bits   (AVector_bits[i*32 +: 32]  ),
                .BVector_bits   (BVector_bits[i*32 +: 32]  ),

                // 操作码控制信号广播
                .is_TF32        (is_TF32                    ),
                .is_BF16        (is_BF16                    ),
                .is_FP16        (is_FP16                    ),
                .is_E5M2        (is_E5M2                    ),
                .is_E3M2        (is_E3M2                    ),
                .is_E4M3        (is_E4M3                    ),
                .is_E2          (is_E2                      ),
                .is_Int8        (is_Int8                    ),
                .is_int         (is_int                     ),
                .is_E8          (is_E8                      ),
                .is_fp16_bf16   (is_fp16_bf16               ),
                .is_mxnvfp4     (is_mxnvfp4                 ),
                .is_Int4        (is_Int4                    ),
                .is_mxint4      (is_mxint4                  ),
                .is_intfp4      (is_intfp4                  ),
                .is_mxfp8       (is_mxfp8                   ),

                .AS_have        (AS_have                    ),
                .BS_have        (BS_have                    ),

                // 异常输出
                .NAN_o          (NAN_o [i]                  ),
                .PINF_o         (PINF_o[i]                  ),
                .NINF_o         (NINF_o[i]                  ),

                // 尾数输出
                .a_8            (a_8[i]                     ),
                .a_4            (a_4[i]                     ),
                .b_4            (b_4[i]                     ),
                .b_8            (b_8[i]                     ),

                //该32bit的FP4是否全为0，如果是全为0，则输出FP4_all_zero为1，否则为0
                .MXINTFP4_all_zero   (MXINTFP4_all_zero_temp[i]           ),
                .MXFP86_all_zero     (MXFP86_all_zero_temp[i]             ),

                // 指数输出
                .exp_E6         (exp_E6[i]                  ),

                // 符号位
                .vec8_sign      (vec8_sign[i]               )
            );

        end
    endgenerate

endmodule
