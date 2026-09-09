`include "defines.sv"

module p7_shift (
    input wire [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2) - 1:0] sign_fp8,
    input wire [`decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM) - 1:0] nvmxfp4_sig,

    input wire signed [9:0] E_MAX_i,
    input wire signed [9:0] block_max_i,

    input wire  [6:0] nvmxfp4_clz,
    input wire  [5:0] fp8_clz    ,

    // input wire  [1:0]  data_type_i,
    input wire         is_MXINTFP4_i,

    output wire [23:0]  fp32_sig,
    output wire  signed [9 :0]  fp32_exp
);
localparam fp8_w = 28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2);
localparam fp4_w = `decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM);

wire [`decimal_places0 : 0] fp8_sig_temp;
wire [`decimal_places1 : 0] fp4_sig_temp;

assign fp8_sig_temp = (fp8_w - fp8_clz > 26) ? sign_fp8 >> (fp8_w - fp8_clz -26) :  sign_fp8 << (fp8_clz + 26 - fp8_w);
assign fp4_sig_temp = (fp4_w - nvmxfp4_clz > 36) ? nvmxfp4_sig >> (fp4_w - nvmxfp4_clz - 36) : nvmxfp4_sig << (nvmxfp4_clz + 36 - fp4_w);

assign fp32_sig = (is_MXINTFP4_i) ? fp4_sig_temp[`decimal_places1: `decimal_places1 - 23]:
                                   fp8_sig_temp[`decimal_places0: `decimal_places0 - 23];

wire signed [9:0] E_MAX_temp, block_max_temp;

assign E_MAX_temp =  E_MAX_i + (fp8_w - fp8_clz -26);
// assign block_max_temp = (fp4_w - nvmxfp4_clz > 36) ? block_max_i + (fp4_w - nvmxfp4_clz - 36) : block_max_i + (fp4_w - nvmxfp4_clz - 36);
assign block_max_temp = $signed(block_max_i) + (fp4_w - nvmxfp4_clz - 36);

assign fp32_exp =  (is_MXINTFP4_i) ? block_max_temp : E_MAX_temp;
    
endmodule
