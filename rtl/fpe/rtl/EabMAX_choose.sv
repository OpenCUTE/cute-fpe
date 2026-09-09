`include "defines.sv"

module EabMAX_choose (
    input [5:0] E6_MAX0_in, E6_MAX1_in,

    input [2:0] E3_MAX0_in, E3_MAX1_in,

    input [2:0] E_MAX_later_0,E_MAX_later_1,

    // input [1:0] data_type ,
    input is_High_precision,

    output [8:0] Eab_MAX
);
//后面三级中的最大值
//先比较两个E6之间谁大
wire [5:0] E6_01_MAX = (E6_MAX0_in > E6_MAX1_in) ? E6_MAX0_in : E6_MAX1_in;

//找出E6_2,E6_3
wire [5:0] E6_MAX2 = {E3_MAX0_in, E_MAX_later_0};
wire [5:0] E6_MAX3 = {E3_MAX1_in, E_MAX_later_1};
wire [5:0] E6_23_MAX  = (E6_MAX2 > E6_MAX3) ? E6_MAX2 : E6_MAX3;

//E9的最大值暂存
wire [8:0] E9_MAX0_temp = {E6_MAX0_in, E_MAX_later_0};
wire [8:0] E9_MAX1_temp = {E6_MAX1_in, E_MAX_later_1};
wire [8:0] E9_MAX       = (E9_MAX0_temp > E9_MAX1_temp) ? E9_MAX0_temp : E9_MAX1_temp;

assign Eab_MAX = (is_High_precision) ? E9_MAX    : //高精度
                 (E6_23_MAX > E6_01_MAX        ) ? {3'd0, E6_23_MAX} : //其余为低精度
                                                   {3'd0, E6_01_MAX} ;

    
endmodule
