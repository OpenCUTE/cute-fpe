`include "defines.sv"

module sig_mul (
    input  wire [7 :0] a_84[0:1], a_88[0:1],
    input  wire [3 :0] a_48[0:1], a_44[0:1],
    input  wire [3 :0] b_84[0:1], b_44[0:1],
    input  wire [7 :0] b_48[0:1], b_88[0:1],

    input  wire        is_High_precision,
    input  wire        is_INTFP4,
    input  wire        is_INT8,

    output wire [21:0] product11_0, product11_1,
    output wire [7 :0] product4_0, product4_1, product4_2, product4_3,
    output wire [15:0] product8_0, product8_1
);

    // 与 mul_private 相同：2 对 11x11、4 对 4x4、2 对 8x8 乘法器。
    // 输入仍沿用当前 decode_sig 的布局，不能直接套用旧版 mul_private 的选线。
    wire is_low_precision = !(is_High_precision || is_INTFP4 || is_INT8);

    // 高精度重组完整 11 位尾数；INT8 的第一个 B 分散在 b_84 和 a_48。
    // 低精度的第一个尾数在 a_48/b_48，四位数据的第一个尾数在 a_84/b_84。
    wire [10:0] a_11_0 = is_High_precision ? {a_84[0][6:0], a_48[0]} :
                         is_INT8           ? {3'd0, a_84[0]} :
                         is_INTFP4         ? {7'd0, a_84[0][3:0]} :
                                             {7'd0, a_48[0]};
    wire [10:0] a_11_1 = is_High_precision ? {a_84[1][6:0], a_48[1]} :
                         is_INT8           ? {3'd0, a_84[1]} :
                         is_INTFP4         ? {7'd0, a_84[1][3:0]} :
                                             {7'd0, a_48[1]};

    wire [10:0] b_11_0 = is_High_precision ? {b_48[0][6:0], b_84[0]} :
                         is_INT8           ? {3'd0, b_84[0], a_48[0]} :
                         is_INTFP4         ? {7'd0, b_84[0]} :
                                             {7'd0, b_48[0][3:0]};
    wire [10:0] b_11_1 = is_High_precision ? {b_48[1][6:0], b_84[1]} :
                         is_INT8           ? {3'd0, b_84[1], a_48[1]} :
                         is_INTFP4         ? {7'd0, b_84[1]} :
                                             {7'd0, b_48[1][3:0]};

    // 第一对 4x4：低精度的第二个乘积，或四位数据的第二个乘积。
    wire [3:0] a_44_0 = is_low_precision ? a_88[0][3:0] :
                        is_INTFP4       ? a_48[0] : 4'd0;
    wire [3:0] a_44_1 = is_low_precision ? a_88[1][3:0] :
                        is_INTFP4       ? a_48[1] : 4'd0;
    wire [3:0] b_44_0 = is_low_precision ? b_88[0][3:0] :
                        is_INTFP4       ? b_48[0][3:0] : 4'd0;
    wire [3:0] b_44_1 = is_low_precision ? b_88[1][3:0] :
                        is_INTFP4       ? b_48[1][3:0] : 4'd0;

    // 第二对 4x4：四位数据的第三个乘积。
    wire [3:0] a_44_2 = is_INTFP4 ? a_88[0][3:0] : 4'd0;
    wire [3:0] a_44_3 = is_INTFP4 ? a_88[1][3:0] : 4'd0;
    wire [3:0] b_44_2 = is_INTFP4 ? b_88[0][3:0] : 4'd0;
    wire [3:0] b_44_3 = is_INTFP4 ? b_88[1][3:0] : 4'd0;

    // 8x8：INT8 的第二个乘积，或四位数据的第四个乘积。
    wire [7:0] a_88_0 = is_INT8   ? a_88[0] :
                        is_INTFP4 ? {4'd0, a_44[0]} : 8'd0;
    wire [7:0] a_88_1 = is_INT8   ? a_88[1] :
                        is_INTFP4 ? {4'd0, a_44[1]} : 8'd0;
    wire [7:0] b_88_0 = is_INT8   ? b_88[0] :
                        is_INTFP4 ? {4'd0, b_44[0]} : 8'd0;
    wire [7:0] b_88_1 = is_INT8   ? b_88[1] :
                        is_INTFP4 ? {4'd0, b_44[1]} : 8'd0;

    assign product11_0 = a_11_0 * b_11_0;
    assign product11_1 = a_11_1 * b_11_1;
    assign product4_0  = a_44_0 * b_44_0;
    assign product4_1  = a_44_1 * b_44_1;
    assign product4_2  = a_44_2 * b_44_2;
    assign product4_3  = a_44_3 * b_44_3;
    assign product8_0  = a_88_0 * b_88_0;
    assign product8_1  = a_88_1 * b_88_1;

endmodule
