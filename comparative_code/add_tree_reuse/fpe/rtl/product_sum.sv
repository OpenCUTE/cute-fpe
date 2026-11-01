

`include "defines.sv"

module product_sum (
    input  wire [11:0] product0_0, product0_1,
    input  wire [15:0] product1_0,
    input  wire [7 :0] product1_1,

    input  wire [11:0] product2_0, product2_1,
    input  wire [15:0] product3_0,
    input  wire [7 :0] product3_1,

    input  wire [7 :0] sign,
    // input  wire [1 :0] data_type,
    input  wire        is_High_precision,
    // input  wire        is_low_precision,
    input  wire        is_INTFP4,
    input  wire        is_INT8,

    input wire  [5:0] exp_E6 [0:3],

    //存储除了int8之外的尾数部分，然后存储int8的低11位（0-10）
    output wire [21:0] product_out [0:1],
    //存储int8的高5位（11-15），或者存储浮点数据类型的指数部分
    output wire [5:0] exp_E6_o [0:3]
);
    //首先根据符号位，将INTFP4全部取反
    wire signed [8:0] INTFP4_0_0 = (sign[0]) ? -$signed({1'd0, product0_0[7:0]}) : $signed({1'd0, product0_0[7:0]});
    wire signed [8:0] INTFP4_0_1 = (sign[1]) ? -$signed({1'd0, product0_1[7:0]}) : $signed({1'd0, product0_1[7:0]});
    wire signed [8:0] INTFP4_1_0 = (sign[2]) ? -$signed({1'd0, product1_0[7:0]}) : $signed({1'd0, product1_0[7:0]});
    wire signed [8:0] INTFP4_1_1 = (sign[3]) ? -$signed({1'd0, product1_1[7:0]}) : $signed({1'd0, product1_1[7:0]});

    wire signed [8:0] INTFP4_2_0 = (sign[4]) ? -$signed({1'd0, product2_0[7:0]}) : $signed({1'd0, product2_0[7:0]});
    wire signed [8:0] INTFP4_2_1 = (sign[5]) ? -$signed({1'd0, product2_1[7:0]}) : $signed({1'd0, product2_1[7:0]});
    wire signed [8:0] INTFP4_3_0 = (sign[6]) ? -$signed({1'd0, product3_0[7:0]}) : $signed({1'd0, product3_0[7:0]});
    wire signed [8:0] INTFP4_3_1 = (sign[7]) ? -$signed({1'd0, product3_1[7:0]}) : $signed({1'd0, product3_1[7:0]});

    //sam0-1的处理
    //sam0_0：将这两个数据加起来,此时只需要一个12bit+11bit的加法器
    wire [11:0] sam0_0 = is_INTFP4  ? {{3{INTFP4_0_0[8]}}, INTFP4_0_0}  : product0_0;
    wire [10:0] sam0_1 = is_INTFP4  ? {{2{INTFP4_0_1[8]}}, INTFP4_0_1}  :
                         is_INT8    ? {3'd0, product0_1[11:4]}          : product0_1[10:0];
    
    wire [11:0] sam0_temp = sam0_0 + sam0_1;//注意这里的第13bit永远为0

    //除去高精度的
    wire signed [10:0] sam_l0 = is_INT8 ? {sam0_temp[6:0],product0_1[3:0]}: $signed(sam0_temp[10:0]);
    assign exp_E6_o[0] = is_INT8 ? {1'd0, sam0_temp[11:7]} : exp_E6[0];

    //sam0_1：高精度靠拼起来，intfp4才需要加起来，其余直接填充
    wire [23:0] samh0_1 = {product1_0,product1_1};

    wire signed [9:0] INTFP4_1 = INTFP4_1_0 + INTFP4_1_1;
    //除去高精度的
    wire signed [10:0] sam_l1 = is_INTFP4 ? {{1{INTFP4_1[9]}}, INTFP4_1} : product1_0[10:0];
    assign exp_E6_o[1] = is_INT8 ? {1'd0, product1_0[15:11]} : exp_E6[1];
    
    //计算高精度
    wire [19:0] samh_0_temp = sam0_temp + samh0_1[23:4];
    wire [21:0] sam_h0 = {samh_0_temp, samh0_1[3:0]};


    //sam2-3的处理
    //sam2_0：将这两个数据加起来,此时只需要一个12bit+11bit的加法器
    wire [11:0] sam2_0 = is_INTFP4 ? {{3{INTFP4_2_0[8]}}, INTFP4_2_0} : product2_0;
    wire [10:0] sam2_1 = is_INTFP4 ? {{2{INTFP4_2_1[8]}}, INTFP4_2_1} :
                         is_INT8    ? {3'd0, product2_1[11:4]}           : product2_1[10:0];
    
    wire [11:0] sam2_temp = sam2_0 + sam2_1;//注意这里的第13bit永远为0

    //除去高精度的
    wire signed [10:0] sam_l2 = is_INT8 ? {sam2_temp[6:0],product2_1[3:0]}: $signed(sam2_temp[10:0]);
    assign exp_E6_o[2] = is_INT8 ? {1'd0, sam2_temp[11:7]} : exp_E6[2];
    
    //sam0_1：高精度靠拼起来，intfp4才需要加起来，其余直接填充
    wire [23:0] samh1_1 = {product3_0,product3_1};

    wire signed [9:0] INTFP4_3 = INTFP4_3_0 + INTFP4_3_1;
    //除去高精度的
    wire signed [10:0] sam_l3 = is_INTFP4 ? {{1{INTFP4_3[9]}}, INTFP4_3} : product3_0[10:0];
    assign exp_E6_o[3] = is_INT8 ? {1'd0, product3_0[15:11]} : exp_E6[3];
    
    //计算高精度
    wire [19:0] samh_1_temp = sam2_temp + samh1_1[23:4];
    wire [21:0] sam_h1 = {samh_1_temp, samh1_1[3:0]};

    // =========================================================================
    // 输出拼接
    // =========================================================================
    // assign product_out[0] = (data_type == `DATA_High_precision) ? {5'd0, sam_h0, 5'd0} : {sam_l1, sam_l0};
    // assign product_out[1] = (data_type == `DATA_High_precision) ? {5'd0, sam_h1, 5'd0} : {sam_l3, sam_l2};
    // assign product_out[0] = is_High_precision ? {5'd0, sam_h0, 5'd0} : {sam_l1, sam_l0};
    // assign product_out[1] = is_High_precision ? {5'd0, sam_h1, 5'd0} : {sam_l3, sam_l2};
    // assign product_out[0] = is_High_precision ? {sam_h0, 2'd0} : {sam_l1, sam_l0};
    // assign product_out[1] = is_High_precision ? {sam_h1, 2'd0} : {sam_l3, sam_l2};
    assign product_out[0] = is_High_precision ? {sam_h0} : {sam_l1, sam_l0};
    assign product_out[1] = is_High_precision ? {sam_h1} : {sam_l3, sam_l2};

endmodule
