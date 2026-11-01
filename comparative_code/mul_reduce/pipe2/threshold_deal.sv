`include "defines.sv"

module threshold_deal (
    input wire is_threshold_normal,
    input wire is_threshold_mxintfp4,

    input wire C_ZERO,

    input wire [7:0] bias,
    
    //这个是对于mxfp4的情况而言的，其只与scale_max有关
    // input wire [8:0] scale_max,
    //这个是对于normal的情况而言的，其每32个元素共用一个scale_exp
    input wire [8:0] scale_exp_0, scale_exp_1,

    //对于mxfp4而言，只需要输出一个使能信号即可（不过如果时序紧张，可以挪移到下一级）
    output wire threshold_mxintfp4_en,
    //对于normal而言，输出临时比较的结果，后续还需要进行比较
    output wire [9:0] threshold_normal_exp_0, threshold_normal_exp_1
);
// parameter THRESHOLD_mxfp4  = `MIN_EXP_THRESHOLD1 + 2*127;
parameter THRESHOLD_normal = `MIN_EXP_THRESHOLD0 + 3*127;

//因为时序原因，这里我们将比较逻辑后移动
// //对于mxfp4的阈值处理
// assign threshold_mxfp4_en = is_threshold_mxintfp4 && C_ZERO && (scale_max < THRESHOLD_mxfp4);
assign threshold_mxintfp4_en = is_threshold_mxintfp4 && C_ZERO;

//对于normal的阈值处理
assign threshold_normal_exp_0 = (is_threshold_normal && C_ZERO) ? (THRESHOLD_normal + {{2{bias[7]}}, bias} - scale_exp_0) : 10'd0;
assign threshold_normal_exp_1 = (is_threshold_normal && C_ZERO) ? (THRESHOLD_normal + {{2{bias[7]}}, bias} - scale_exp_1) : 10'd0;

endmodule
