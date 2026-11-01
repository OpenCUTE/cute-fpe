`include "defines.sv"

module decode_data_unit (
    // from input
    input wire [31 : 0] AVector_bits,
    input wire [31 : 0] BVector_bits,

    // from decode_opcode
    input wire is_TF32, is_BF16, is_FP16,
    input wire is_E5M2, is_E3M2, is_E4M3,
    input wire is_E2  , is_Int8, is_E8,
    input wire is_fp16_bf16,
    input wire is_mxnvfp4,
    input wire is_Int4,
    input wire is_int,
    input wire is_mxint4,
    input wire is_intfp4,
    input wire is_mxfp8,

    input wire AS_have, BS_have,

    // to pipe1_stage
    output wire NAN_o, PINF_o, NINF_o,

    // to sig_mul
    output wire [7 :0] a_8[0:1],
    output wire [3 :0] a_4[0:1],
    output wire [3 :0] b_4[0:1],
    output wire [7 :0] b_8[0:1],

    //判断该组fp4/int4是否全为0，如果是全为0，则输出FP4_all_zero为1，否则为0
    output wire        MXINTFP4_all_zero,
    //mxfp8/fp6
    output wire        MXFP86_all_zero,

    // to exp_compare
    output wire [5:0] exp_E6 [0 : 3],

    // sign vector
    output wire [7:0] vec8_sign
);

    /*=========================================================
     * 内部连线声明
     *=========================================================*/

    // --- decode_Adata 输出 ---
    wire [7:0] fp_a_8[0:1];
    wire [3:0] fp_a_4[0:1];
    wire signed [4:0] Aexp_E5[0:3];
    wire [7:0] A_vec8_sign;
    wire [3:0] Azero;
    wire [3:0] nz_E5_AFP8, nz_E3_AFP6;

    // --- decode_Bdata 输出 ---
    wire [7:0] fp_b_8[0:1];
    wire [3:0] fp_b_4[0:1];
    wire signed [4:0] Bexp_E5[0:3];
    wire [7:0] B_vec8_sign;
    wire [3:0] Bzero;
    wire [3:0] nz_E5_BFP8, nz_E3_BFP6;

    // --- decode_int8 输出 ---
    wire [7:0] int8_a_8[0:1];
    wire [3:0] int8_a_4[0:1];
    wire [3:0] int8_b_4[0:1];
    wire [7:0] int8_b_8[0:1];
    wire [5:0] exp_E6_int[0:3];

    // --- FP4_to_int 输出 ---
    wire [31:0] AFP4_int;
    wire [31:0] BFP4_int;

    // --- int4 decode outputs ---
    wire [31:0] int4_Avector;
    wire [31:0] int4_Bvector;

    // --- assign_FP4 输出 （fp4/int4）---
    wire [7:0] intfp4_a_8[0:1];
    wire [3:0] intfp4_a_4[0:1];
    wire [3:0] intfp4_b_4[0:1];
    wire [7:0] intfp4_b_8[0:1];
    wire [5:0] exp_E6_intfp4[0:3];

    // --- exp_sum 输出 ---
    wire [5:0] exp_E6_fp[0:3];

    // --- exception_hand 输出 ---
    wire [3:0] A_NAN, A_INF, A_ZERO;
    wire [3:0] B_NAN, B_INF, B_ZERO;

    /*=========================================================
     * 符号位向量：A和B的符号位进行异或操作即可
     *=========================================================*/
    assign vec8_sign = A_vec8_sign ^ B_vec8_sign;

    /*=========================================================
     * 1. decode_Adata：解码 A 向量浮点数据
     *=========================================================*/
    decode_Adata u_decode_Adata (
        .Vector_bits    (AVector_bits   ),
        .is_TF32        (is_TF32        ),
        .is_BF16        (is_BF16        ),
        .is_FP16        (is_FP16        ),
        .is_E5M2        (is_E5M2        ),
        .is_E3M2        (is_E3M2        ),
        .is_E4M3        (is_E4M3        ),
        .is_E2          (is_E2          ),
        .is_Int8        (is_Int8        ),
        .is_mxint4      (is_mxint4      ),
        .is_Int4        (is_Int4        ),
        .is_int         (is_int         ),
        .is_E8          (is_E8          ),
        .is_fp16_bf16   (is_fp16_bf16   ),
        .is_intfp4      (is_intfp4     ),
        .AS_have        (AS_have        ),
        .a_8            (fp_a_8         ),
        .a_4            (fp_a_4         ),
        .Azero          (Azero          ),
        .vec8_sign      (A_vec8_sign    ),
        .nz_E5_FP8      (nz_E5_AFP8     ),
        .nz_E3_FP6      (nz_E3_AFP6     ),
        .exp_E5         (Aexp_E5        )
    );

    /*=========================================================
     * 2. decode_Bdata：解码 B 向量浮点数据
     *=========================================================*/
    decode_Bdata u_decode_Bdata (
        .Vector_bits    (BVector_bits   ),
        .is_TF32        (is_TF32        ),
        .is_BF16        (is_BF16        ),
        .is_FP16        (is_FP16        ),
        .is_E5M2        (is_E5M2        ),
        .is_E3M2        (is_E3M2        ),
        .is_E4M3        (is_E4M3        ),
        .is_E2          (is_E2          ),
        .is_Int8        (is_Int8        ),
        .is_mxint4      (is_mxint4      ),
        .is_Int4        (is_Int4        ),
        .is_int         (is_int         ),
        .is_E8          (is_E8          ),
        .is_fp16_bf16   (is_fp16_bf16   ),
        .is_intfp4      (is_intfp4     ),
        .BS_have        (BS_have        ),
        .b_8            (fp_b_8         ),
        .b_4            (fp_b_4         ),
        .Bzero          (Bzero          ),
        .vec8_sign      (B_vec8_sign    ),
        .nz_E5_FP8      (nz_E5_BFP8     ),
        .nz_E3_FP6      (nz_E3_BFP6     ),
        .exp_E5         (Bexp_E5        )
    );

    /*=========================================================
     * 3. decode_int8：解码 INT8 路径的尾数和指数
     *=========================================================*/
    decode_int8 u_decode_int8 (
        .AVector    (AVector_bits   ),
        .BVector    (BVector_bits   ),
        .AS_have    (AS_have        ),
        .BS_have    (BS_have        ),
        .a_8        (int8_a_8       ),
        .a_4        (int8_a_4       ),
        .b_4        (int8_b_4       ),
        .b_8        (int8_b_8       ),
        .exp_E6     (exp_E6_int     )
    );

    /*=========================================================
     * 4. FP4toint x 2：将 A/B 的 FP4 向量转换为定点整数
     *=========================================================*/
    FP4toint u_FP4toint_A (
        .Vector         (AVector_bits   ),
        .FP4_int        (AFP4_int       )
    );

    FP4toint u_FP4toint_B (
        .Vector         (BVector_bits   ),
        .FP4_int        (BFP4_int       )
    );

    //将int4向量进行解码
    decode_int4 u_decode_int4 (
        .AVector_bits   (AVector_bits), // 8x s4/u4
        .BVector_bits   (BVector_bits),

        .AS_have        (AS_have), 
        .BS_have        (BS_have),  // 1: s4; 0: u4

        .int4_Avector   (int4_Avector),
        .int4_Bvector   (int4_Bvector)
    );

    zero_detection u_zero_detection (
        .A_Vector           (AVector_bits), 
        .B_Vector           (BVector_bits),
        .is_mxint4          (is_mxint4   ),
        .is_mxfp8           (is_mxfp8    ),

        .nz_E5_AFP8         (nz_E5_AFP8  ),
        .nz_E3_AFP6         (nz_E3_AFP6  ),
        .nz_E5_BFP8         (nz_E5_BFP8  ),
        .nz_E3_BFP6         (nz_E3_BFP6  ),

        .MXINTFP4_all_zero  (MXINTFP4_all_zero),
        .MXFP86_all_zero    (MXFP86_all_zero  )
    );

    /*=========================================================
     * 5. assign_INTFP4：将 FP4/INT4 整数分配到乘法器输入
     *=========================================================*/
    assign_INTFP4 u_assign_INTFP4 (
        .AFP4_int   (AFP4_int       ),
        .BFP4_int   (BFP4_int       ),

        .int4_Avector(int4_Avector  ),
        .int4_Bvector(int4_Bvector  ),

        .is_mxnvfp4  (is_mxnvfp4    ),

        .a_8        (intfp4_a_8    ),
        .a_4        (intfp4_a_4    ),
        .b_4        (intfp4_b_4    ),
        .b_8        (intfp4_b_8    ),
        .exp_E6     (exp_E6_intfp4 )
    );

    /*=========================================================
     * 6. multiplier_choose：三路选择（FP / INTFP4 / INT8）
     *=========================================================*/
    multiplier_choose u_multiplier_choose (
        // FP 路径
        .fp_a_8     (fp_a_8         ),
        .fp_a_4     (fp_a_4         ),
        .fp_b_8     (fp_b_8         ),
        .fp_b_4     (fp_b_4         ),

        // NVFP4/int4 路径
        .intfp4_a_8  (intfp4_a_8      ),
        .intfp4_a_4  (intfp4_a_4      ),
        .intfp4_b_8  (intfp4_b_8      ),
        .intfp4_b_4  (intfp4_b_4      ),

        // INT8 路径
        .int8_a_8   (int8_a_8       ),
        .int8_a_4   (int8_a_4       ),
        .int8_b_8   (int8_b_8       ),
        .int8_b_4   (int8_b_4       ),

        // 选择控制
        .is_intfp4  (is_intfp4      ),
        .is_Int8    (is_Int8        ),

        // 输出
        .a_8        (a_8     ),
        .a_4        (a_4     ),
        .b_8        (b_8     ),
        .b_4        (b_4     )
    );

    /*=========================================================
     * 7. 指数路径选择
     *=========================================================*/
    exp_sum u_exp_sum (
        .is_E8      (is_E8          ),
        .is_FP16    (is_FP16        ),
        .Azero      (Azero          ),
        .Bzero      (Bzero          ),
        .Aexp_E5    (Aexp_E5        ),
        .Bexp_E5    (Bexp_E5        ),
        .exp_E6     (exp_E6_fp      )
    );

    exp_choose u_exp_choose (
        .exp_E6_fp  (exp_E6_fp  ),
        .exp_E6_int (exp_E6_int ),
        .exp_E6_intfp4 (exp_E6_intfp4 ),
        .is_intfp4  (is_intfp4 ),
        .is_Int8    (is_Int8    ),
        .exp_E6     (exp_E6     )
    );

    /*=========================================================
     * 8. decode_excep x 2：A/B 异常检测
     *=========================================================*/
    decode_excep u_excep_A (
        .Vector_bits    (AVector_bits   ),
        .is_TF32        (is_TF32        ),
        .is_BF16        (is_BF16        ),
        .is_FP16        (is_FP16        ),
        .is_E5M2        (is_E5M2        ),
        .is_E4M3        (is_E4M3        ),
        .NAN            (A_NAN          ),
        .INF            (A_INF          ),
        .ZERO           (A_ZERO         )
    );

    decode_excep u_excep_B (
        .Vector_bits    (BVector_bits   ),
        .is_TF32        (is_TF32        ),
        .is_BF16        (is_BF16        ),
        .is_FP16        (is_FP16        ),
        .is_E5M2        (is_E5M2        ),
        .is_E4M3        (is_E4M3        ),
        .NAN            (B_NAN          ),
        .INF            (B_INF          ),
        .ZERO           (B_ZERO         )
    );

    /*=========================================================
     * 9. excep_mul：合并 A/B 异常，判断乘法结果异常
     *=========================================================*/
    excep_mul u_excep_mul (
        .sign       (vec8_sign[3:0] ),
        .A_NAN      (A_NAN          ),
        .A_INF      (A_INF          ),
        .A_ZERO     (A_ZERO         ),
        .B_NAN      (B_NAN          ),
        .B_INF      (B_INF          ),
        .B_ZERO     (B_ZERO         ),
        .NAN_o      (NAN_o          ),
        .PINF_o     (PINF_o         ),
        .NINF_o     (NINF_o         )
    );

endmodule
