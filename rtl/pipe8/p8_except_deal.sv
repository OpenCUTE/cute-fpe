`include "defines.sv"

module p8_except_deal (
    input wire NAN_i, NINF_i, PINF_i,

    input wire fp32_sign,
    input wire [23:0]  fp32_sig,
    input wire signed [9 :0]  fp32_exp,

    output wire [31:0] fp32
);

wire [31:0] fp32_temp;

assign fp32_temp = NAN_i  ? 32'h7FFFFFFF :
                   NINF_i ? 32'hFF800000 :
                   PINF_i ? 32'h7F800000 : 
                            32'h0        ;

wire [7:0] fp32_exp_temp;
assign fp32_exp_temp = (fp32_exp[9] || (!(|fp32_sig))) ? 8'h0 : //非规格化树或者该数据为0(当最大指数被减成负数了，或者最大指数直接就是0)
                       (fp32_exp[8]) ? 8'hFF   : //无穷
                                       fp32_exp[7:0] ;

wire [22:0]  fp32_sig_temp;
assign fp32_sig_temp = (fp32_exp[9] || !(|fp32_exp)) ? fp32_sig >> (1 - fp32_exp) : //非规格化数(当最大指数被减成负数了，或者最大指数直接就是0)
                       (fp32_exp[8] || &fp32_exp[7:0]) ? 23'd0    : //无穷
                                                        fp32_sig[22:0];


assign fp32 = (NAN_i || NINF_i || PINF_i)? fp32_temp : {fp32_sign, fp32_exp_temp, fp32_sig_temp};

endmodule
