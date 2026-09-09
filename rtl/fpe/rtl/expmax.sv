`include "defines.sv"
//我们在这里进行如下处理
//1:首先在计算移位值的时候，需要考虑C为0，及其阈值的情况
//2:其次在计算E_MAX的时候，其作为FP32域的最大指数值，按照现有逻辑进行计算即可（即不考虑C为0的情况）
//3:最后当C=0的时候，如果ab的最大指数值小于-126，则我们按照计算一下当前最大指数值相较于-126小了多少。
//对于情况3，我们在最后计算先导0的个数的时候，直接将计算出来的FP32域的最大指数值减去当前的偏移值即可。
module expmax (
    input wire [7:0] C_exp,
    input wire       C_ZERO,
    input wire signed [7:0] bias,
    
    //normal阈值情况
    input wire [9:0] threshold_normal_exp_0, threshold_normal_exp_1,

    input wire signed [8:0] scale_exp_s0,scale_exp_s1,

    input wire [8:0] Eab_MAX0,
    input wire [8:0] Eab_MAX1,

    //可能出现负数，因为我们将其按照FP32域的指数来输出，但是其最低指数值会低于-126，所以我们需要使用signed类型来进行输出
    output wire signed [9:0] E_MAX,

    output wire [9:0] E_MAX_shift0, E_MAX_shift1,
    output wire [4:0] c_shift
);
    //阈值判断
    wire threshold_normal_en = ($signed(threshold_normal_exp_0) > $signed({1'd0, Eab_MAX0}) && $signed(threshold_normal_exp_1) > $signed({1'd0, Eab_MAX1})) ? 1'b1 : 1'b0;

    // wire [9:0] E_MAX_temp;
    wire signed [9:0] E_MAX_shift;
    // wire signed [9:0] scale_exp_s0, scale_exp_s1;

    // assign scale_exp_s0 = $signed({1'd0,scale_exp0}) - 10'sd254;
    // assign scale_exp_s1 = $signed({1'd0,scale_exp1}) - 10'sd254;

    wire signed [9:0] Eab_MAX_S0 = $signed({1'd0, Eab_MAX0}) + scale_exp_s0;
    wire signed [9:0] Eab_MAX_S1 = $signed({1'd0, Eab_MAX1}) + scale_exp_s1;

    wire signed [9:0] Eab_MAX_S  = (Eab_MAX_S0 > Eab_MAX_S1) ? Eab_MAX_S0 : Eab_MAX_S1;

    // 将C的指数调整到可以与ab的指数对齐
    wire signed [9:0] C_temp;
    assign C_temp = C_ZERO ? 10'h200 : ($signed({2'b0, C_exp}) + $signed({{2{bias[7]}}, bias})); // $signed(bias) 自动符号扩展到10位

    // 用于移位的最大值（shift域）
    assign E_MAX_shift  = (C_temp < Eab_MAX_S || threshold_normal_en) ? Eab_MAX_S : C_temp;
    assign E_MAX_shift0 = (threshold_normal_en)? {threshold_normal_exp_0} : $unsigned(E_MAX_shift - scale_exp_s0);
    assign E_MAX_shift1 = (threshold_normal_en)? {threshold_normal_exp_1} : $unsigned(E_MAX_shift - scale_exp_s1);

    // c_shift：C相对于最大值需要右移多少位
    wire [9:0] c_shift_temp;

    assign c_shift_temp = (C_temp > Eab_MAX_S) ? 10'd0
                        : ($unsigned(Eab_MAX_S) - $unsigned(C_temp));

    assign c_shift = (|c_shift_temp[9:5]) ? 5'h1F : c_shift_temp[4:0];

    //输出原始的
    assign E_MAX = (threshold_normal_en) ? -10'sd6 : E_MAX_shift - $signed({{2{bias[7]}}, bias});

    // // Eab在FP32域的真实指数
    // wire signed [9:0] Eab_MAX_temp;
    // assign Eab_MAX_temp = Eab_MAX_S - $signed(bias);

    // // E_MAX：FP32域的最大指数
    // assign E_MAX_temp = ($signed({2'd0, C_exp}) > Eab_MAX_temp) ? {2'd0, C_exp} : Eab_MAX_temp;
    // assign E_MAX      = E_MAX_temp[9] ? 9'b111111111 : E_MAX_temp[8:0] ;

endmodule
