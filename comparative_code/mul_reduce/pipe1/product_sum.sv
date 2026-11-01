`include "defines.sv"

module product_sum (
    input  wire [21:0] product11_0, product11_1,
    input  wire [7 :0] product4_0, product4_1, product4_2, product4_3,
    input  wire [15:0] product8_0, product8_1,

    input  wire [7 :0] sign,
    input  wire        is_High_precision,
    input  wire        is_INTFP4,
    input  wire        is_INT8,

    input  wire [5:0] exp_E6 [0:3],

    output wire [21:0] product_out [0:1],
    output wire [5:0] exp_E6_o [0:3]
);

    // 四位类型（INT4、NVFP4、MXFP4、MXINT4）先按符号转为补码，再两两求和。
    wire signed [8:0] INTFP4_0_0 = sign[0] ? -$signed({1'b0, product11_0[7:0]}) : $signed({1'b0, product11_0[7:0]});
    wire signed [8:0] INTFP4_0_1 = sign[4] ? -$signed({1'b0, product11_1[7:0]}) : $signed({1'b0, product11_1[7:0]});
    wire signed [8:0] INTFP4_1_0 = sign[1] ? -$signed({1'b0, product4_0}) : $signed({1'b0, product4_0});
    wire signed [8:0] INTFP4_1_1 = sign[5] ? -$signed({1'b0, product4_1}) : $signed({1'b0, product4_1});
    wire signed [8:0] INTFP4_2_0 = sign[2] ? -$signed({1'b0, product4_2}) : $signed({1'b0, product4_2});
    wire signed [8:0] INTFP4_2_1 = sign[6] ? -$signed({1'b0, product4_3}) : $signed({1'b0, product4_3});
    wire signed [8:0] INTFP4_3_0 = sign[3] ? -$signed({1'b0, product8_0[7:0]}) : $signed({1'b0, product8_0[7:0]});
    wire signed [8:0] INTFP4_3_1 = sign[7] ? -$signed({1'b0, product8_1[7:0]}) : $signed({1'b0, product8_1[7:0]});

    wire signed [9:0] INTFP4_sum0_0 = INTFP4_0_0 + INTFP4_1_0;
    wire signed [9:0] INTFP4_sum0_1 = INTFP4_2_0 + INTFP4_3_0;
    wire signed [9:0] INTFP4_sum1_0 = INTFP4_0_1 + INTFP4_1_1;
    wire signed [9:0] INTFP4_sum1_1 = INTFP4_2_1 + INTFP4_3_1;

    // 当前下游读取 11 位有符号和，负数必须从 10 位符号扩展。
    wire [10:0] sam_l0 = is_INTFP4 ? {INTFP4_sum0_0[9], INTFP4_sum0_0} : product11_0[10:0];
    wire [10:0] sam_l1 = is_INTFP4 ? {INTFP4_sum0_1[9], INTFP4_sum0_1} :
                        is_INT8   ? product8_0[10:0] : {3'd0, product4_0};
    wire [10:0] sam_l2 = is_INTFP4 ? {INTFP4_sum1_0[9], INTFP4_sum1_0} : product11_1[10:0];
    wire [10:0] sam_l3 = is_INTFP4 ? {INTFP4_sum1_1[9], INTFP4_sum1_1} :
                        is_INT8   ? product8_1[10:0] : {3'd0, product4_1};

    // INT8 的完整 16 位幅值由 exp_E6_o 的低 5 位和 sam_l 的 11 位重组。
    assign exp_E6_o[0] = is_INT8 ? {1'b0, product11_0[15:11]} : exp_E6[0];
    assign exp_E6_o[1] = is_INT8 ? {1'b0, product8_0[15:11]} : exp_E6[1];
    assign exp_E6_o[2] = is_INT8 ? {1'b0, product11_1[15:11]} : exp_E6[2];
    assign exp_E6_o[3] = is_INT8 ? {1'b0, product8_1[15:11]} : exp_E6[3];

    // 高精度不再重组部分积，直接使用 11x11 的完整乘积。
    assign product_out[0] = is_High_precision ? product11_0 : {sam_l1, sam_l0};
    assign product_out[1] = is_High_precision ? product11_1 : {sam_l3, sam_l2};

endmodule
