`include "defines.sv"

module multiplier_choose(
    //来自除了fp4之外的
    input wire [7 :0] fp_a_84[0:1], fp_a_88[0:1],
    input wire [3 :0] fp_a_48[0:1], fp_a_44[0:1],
    input wire [7 :0] fp_b_48[0:1], fp_b_88[0:1],
    input wire [3 :0] fp_b_84[0:1], fp_b_44[0:1],
    // input wire [7 :0] fp_a_8[0:1],
    // input wire [3 :0] fp_a_4[0:1],
    // input wire [7 :0] fp_b_8[0:1],
    // input wire [3 :0] fp_b_4[0:1],

    //来自intnvfp4
    input wire [7 :0] intfp4_a_84[0:1], intfp4_a_88[0:1],
    input wire [3 :0] intfp4_a_48[0:1], intfp4_a_44[0:1],
    input wire [7 :0] intfp4_b_48[0:1], intfp4_b_88[0:1],
    input wire [3 :0] intfp4_b_84[0:1], intfp4_b_44[0:1],
    // input wire [7 :0] intfp4_a_8[0:1],
    // input wire [3 :0] intfp4_a_4[0:1],
    // input wire [7 :0] intfp4_b_8[0:1],
    // input wire [3 :0] intfp4_b_4[0:1],

    //来自int8
    input wire [7 :0] int8_a_84[0:1], int8_a_88[0:1],
    input wire [3 :0] int8_a_48[0:1], int8_a_44[0:1],
    input wire [7 :0] int8_b_48[0:1], int8_b_88[0:1],
    input wire [3 :0] int8_b_84[0:1], int8_b_44[0:1],
    // input wire [7 :0] int8_a_8[0:1],
    // input wire [3 :0] int8_a_4[0:1],
    // input wire [7 :0] int8_b_8[0:1],
    // input wire [3 :0] int8_b_4[0:1],


    //如果这两个都不是则为fp，反之则为对应的数据
    input wire is_intfp4, is_Int8,

    output wire [7 :0] a_84[0:1], a_88[0:1],
    output wire [3 :0] a_48[0:1], a_44[0:1],
    output wire [7 :0] b_48[0:1], b_88[0:1],
    output wire [3 :0] b_84[0:1], b_44[0:1]
    // output wire [7 :0] a_8[0:1],
    // output wire [3 :0] a_4[0:1],
    // output wire [7 :0] b_8[0:1],
    // output wire [3 :0] b_4[0:1]
);

    // a_84
    assign a_84[0] = is_Int8 ? int8_a_84[0] : (is_intfp4 ? intfp4_a_84[0] : fp_a_84[0]);
    assign a_84[1] = is_Int8 ? int8_a_84[1] : (is_intfp4 ? intfp4_a_84[1] : fp_a_84[1]);

    // a_88
    assign a_88[0] = is_Int8 ? int8_a_88[0] : (is_intfp4 ? intfp4_a_88[0] : fp_a_88[0]);
    assign a_88[1] = is_Int8 ? int8_a_88[1] : (is_intfp4 ? intfp4_a_88[1] : fp_a_88[1]);

    // a_48
    assign a_48[0] = is_Int8 ? int8_a_48[0] : (is_intfp4 ? intfp4_a_48[0] : fp_a_48[0]);
    assign a_48[1] = is_Int8 ? int8_a_48[1] : (is_intfp4 ? intfp4_a_48[1] : fp_a_48[1]);

    // a_44
    assign a_44[0] = is_Int8 ? int8_a_44[0] : (is_intfp4 ? intfp4_a_44[0] : fp_a_44[0]);
    assign a_44[1] = is_Int8 ? int8_a_44[1] : (is_intfp4 ? intfp4_a_44[1] : fp_a_44[1]);

    // b_84
    assign b_84[0] = is_Int8 ? int8_b_84[0] : (is_intfp4 ? intfp4_b_84[0] : fp_b_84[0]);
    assign b_84[1] = is_Int8 ? int8_b_84[1] : (is_intfp4 ? intfp4_b_84[1] : fp_b_84[1]);

    // b_88
    assign b_88[0] = is_Int8 ? int8_b_88[0] : (is_intfp4 ? intfp4_b_88[0] : fp_b_88[0]);
    assign b_88[1] = is_Int8 ? int8_b_88[1] : (is_intfp4 ? intfp4_b_88[1] : fp_b_88[1]);

    // b_48
    assign b_48[0] = is_Int8 ? int8_b_48[0] : (is_intfp4 ? intfp4_b_48[0] : fp_b_48[0]);
    assign b_48[1] = is_Int8 ? int8_b_48[1] : (is_intfp4 ? intfp4_b_48[1] : fp_b_48[1]);

    // b_44
    assign b_44[0] = is_Int8 ? int8_b_44[0] : (is_intfp4 ? intfp4_b_44[0] : fp_b_44[0]);
    assign b_44[1] = is_Int8 ? int8_b_44[1] : (is_intfp4 ? intfp4_b_44[1] : fp_b_44[1]);


    // assign a_8[0] = is_Int8 ? int8_a_8[0] : (is_intfp4 ? intfp4_a_8[0] : fp_a_8[0]);
    // assign a_8[1] = is_Int8 ? int8_a_8[1] : (is_intfp4 ? intfp4_a_8[1] : fp_a_8[1]);
    // assign a_4[0] = is_Int8 ? int8_a_4[0] : (is_intfp4 ? intfp4_a_4[0] : fp_a_4[0]);
    // assign a_4[1] = is_Int8 ? int8_a_4[1] : (is_intfp4 ? intfp4_a_4[1] : fp_a_4[1]);

    // assign b_8[0] = is_Int8 ? int8_b_8[0] : (is_intfp4 ? intfp4_b_8[0] : fp_b_8[0]);
    // assign b_8[1] = is_Int8 ? int8_b_8[1] : (is_intfp4 ? intfp4_b_8[1] : fp_b_8[1]);
    // assign b_4[0] = is_Int8 ? int8_b_4[0] : (is_intfp4 ? intfp4_b_4[0] : fp_b_4[0]);
    // assign b_4[1] = is_Int8 ? int8_b_4[1] : (is_intfp4 ? intfp4_b_4[1] : fp_b_4[1]);

endmodule
