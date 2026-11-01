`include "defines.sv"

module sig_deal_unit (
    input  wire [7 :0] a_84[0:1], a_88[0:1],
    input  wire [3 :0] a_48[0:1], a_44[0:1],
    input  wire [3 :0] b_84[0:1], b_44[0:1],
    input  wire [7 :0] b_48[0:1], b_88[0:1],

    input  wire [7 :0] sign,
    // input  wire [1 :0] data_type,
    input  wire        is_High_precision,
    // input  wire        is_low_precision,
    input  wire        is_INTFP4,
    input  wire        is_INT8,

    input wire  [5:0] exp_E6 [0:3],

    //存储除了int8之外的尾数部分，然后存储int8的低11位（0-10）
    output wire [21:0] product_out [0:1],
    //存储int8的高5位（12-15），或者存储浮点数据类型的指数部分
    output wire [5:0] exp_E6_o [0:3]
);

    // -------------------------------------------------------------------------
    // 中间连线：sig_mul 的输出 → product_sum 的输入
    // -------------------------------------------------------------------------
    wire [11:0] product0_0, product0_1;
    wire [15:0] product1_0;
    wire [7 :0] product1_1;

    wire [11:0] product2_0, product2_1;
    wire [15:0] product3_0;
    wire [7 :0] product3_1;

    // -------------------------------------------------------------------------
    // 尾数乘法
    // -------------------------------------------------------------------------
    sig_mul u_sig_mul (
        .a_84       (a_84      ),
        .a_88       (a_88      ),
        .a_48       (a_48      ),
        .a_44       (a_44      ),
        .b_84       (b_84      ),
        .b_44       (b_44      ),
        .b_48       (b_48      ),
        .b_88       (b_88      ),

        .product0_0 (product0_0),
        .product0_1 (product0_1),
        .product1_0 (product1_0),
        .product1_1 (product1_1),
        .product2_0 (product2_0),
        .product2_1 (product2_1),
        .product3_0 (product3_0),
        .product3_1 (product3_1)
    );

    // -------------------------------------------------------------------------
    // 乘积汇总求和
    // -------------------------------------------------------------------------
    product_sum u_product_sum (
        .product0_0  (product0_0  ),
        .product0_1  (product0_1  ),
        .product1_0  (product1_0  ),
        .product1_1  (product1_1  ),

        .product2_0  (product2_0  ),
        .product2_1  (product2_1  ),
        .product3_0  (product3_0  ),
        .product3_1  (product3_1  ),

        .sign        (sign        ),
        // .data_type   (data_type   ),
        .is_High_precision (is_High_precision),
        // .is_low_precision  (is_low_precision ),
        .is_INTFP4         (is_INTFP4      ),
        .is_INT8           (is_INT8          ),

        .exp_E6     (exp_E6         ),

        .product_out (product_out ),
        .exp_E6_o   (exp_E6_o       )
    );

endmodule
