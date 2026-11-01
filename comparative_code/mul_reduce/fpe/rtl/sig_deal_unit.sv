`include "defines.sv"

module sig_deal_unit (
    input  wire [7 :0] a_84[0:1], a_88[0:1],
    input  wire [3 :0] a_48[0:1], a_44[0:1],
    input  wire [3 :0] b_84[0:1], b_44[0:1],
    input  wire [7 :0] b_48[0:1], b_88[0:1],

    input  wire [7 :0] sign,
    input  wire        is_High_precision,
    input  wire        is_INTFP4,
    input  wire        is_INT8,

    input  wire [5:0] exp_E6 [0:3],

    // 保持当前流水线接口：完整 22 位高精度乘积，或两个 11 位低精度结果。
    output wire [21:0] product_out [0:1],
    // INT8 乘积的高 5 位 [15:11]，其余类型透传指数。
    output wire [5:0] exp_E6_o [0:3]
);

    wire [21:0] product11_0, product11_1;
    wire [7 :0] product4_0, product4_1, product4_2, product4_3;
    wire [15:0] product8_0, product8_1;

    sig_mul u_sig_mul (
        .a_84       (a_84),
        .a_88       (a_88),
        .a_48       (a_48),
        .a_44       (a_44),
        .b_84       (b_84),
        .b_44       (b_44),
        .b_48       (b_48),
        .b_88       (b_88),

        .is_High_precision (is_High_precision),
        .is_INTFP4         (is_INTFP4),
        .is_INT8           (is_INT8),

        .product11_0 (product11_0),
        .product11_1 (product11_1),
        .product4_0  (product4_0),
        .product4_1  (product4_1),
        .product4_2  (product4_2),
        .product4_3  (product4_3),
        .product8_0  (product8_0),
        .product8_1  (product8_1)
    );

    product_sum u_product_sum (
        .product11_0 (product11_0),
        .product11_1 (product11_1),
        .product4_0  (product4_0),
        .product4_1  (product4_1),
        .product4_2  (product4_2),
        .product4_3  (product4_3),
        .product8_0  (product8_0),
        .product8_1  (product8_1),

        .sign              (sign),
        .is_High_precision (is_High_precision),
        .is_INTFP4         (is_INTFP4),
        .is_INT8           (is_INT8),

        .exp_E6      (exp_E6),
        .product_out (product_out),
        .exp_E6_o    (exp_E6_o)
    );

endmodule
