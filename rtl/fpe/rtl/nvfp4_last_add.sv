`include "defines.sv"

module nvfp4_last_add (
    input  wire signed [`decimal_places1+1:0]                                         nvmxfp4_c_sig_i,
    input wire  signed [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM):0]   nvmxfp4_sig_i,

    output wire signed [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig
);

assign nvmxfp4_sig = nvmxfp4_c_sig_i + nvmxfp4_sig_i;

endmodule
