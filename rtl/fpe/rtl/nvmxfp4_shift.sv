`include "defines.sv"

module nvmxfp4_shift (
    input  wire       scale_type,
    input  wire [8:0] scale_exp [0:`Scale_NUM - 1],

    //mxfp4的阈值情况
    input  wire threshold_mxintfp4_en_i,

    //这个是对于mxfp4的情况而言的，其只与scale_max有关

    input  wire [8:0] scale_max_exp,
    input  wire [7:0] C_exp,
    input  wire C_ZERO,

    output wire signed [9:0] block_max,

    output wire [5:0] block_shift [0:`Scale_NUM - 1],
    output wire [5:0] c_shift
);
parameter THRESHOLD_mxfp4  = `MIN_EXP_THRESHOLD1 + 2*127;

wire threshold_mxintfp4_en = threshold_mxintfp4_en_i && (scale_max_exp < THRESHOLD_mxfp4);

wire [6:0] scale_bias = (scale_type == `SCALE_UE8M0) ? 7'd127 : 7'd7;

wire signed [9:0] C_exp_temp = (C_ZERO) ? 10'h200 : $signed({2'd0, C_exp}) - 10'sd127 + $signed({2'd0,scale_bias,1'd0});

wire signed [9:0] scale_max_temp = {1'd0,scale_max_exp} - $signed({2'd0,scale_bias,1'd0}) + 10'sd127;
//1:scale_max_exp 0:C_exp

assign block_max =  (threshold_mxintfp4_en)                        ?  -10'sd12        :
                    (C_exp_temp > $signed({1'd0,scale_max_exp}) || !(|scale_max_exp)) ? {2'd0,C_exp} : scale_max_temp;//作为FP32的原始指数输出即可


wire signed [8:0] shift_temp =  (threshold_mxintfp4_en)                                           ? 9'sd115:
                                (C_exp_temp > $signed({1'd0,scale_max_exp}) || !(|scale_max_exp)) ? C_exp_temp : scale_max_exp;

genvar i;
generate
    for (i = 0; i < `Scale_NUM; i = i + 1) begin : gen_block_shift
        wire [8:0] block_shift_temp = shift_temp - scale_exp[i];
        assign block_shift[i] = (|block_shift_temp[8:6]) ? 6'b11_1111 : block_shift_temp[5:0];
    end
endgenerate

wire [8:0] c_shift_temp = shift_temp - C_exp_temp;

assign c_shift = |c_shift_temp[8:6] ? 6'b11_1111 : c_shift_temp[5:0];
    
endmodule
