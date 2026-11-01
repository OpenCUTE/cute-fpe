`include "defines.sv"
//尾数需要先移动，后面再进行符号位的判断，否则会出现负数移动完之后，本来应该是0，却变成了FFF，变成了-1

module c_shift (
    input   wire [31:0] C_sig_i,
    input   wire        C_sign_i,
    input   wire [5:0]  nvc_shift_i,
    input   wire [4:0]  c_shift_i,

    output  wire signed [`decimal_places0 + 1 : 0] fp8_c_sig,
    output  wire signed [`decimal_places1 + 1 : 0] nvmxfp4_c_sig
);

wire [`decimal_places0:0] C_sig_temp = {C_sig_i[23:0], 2'd0};
wire [`decimal_places1:0] nvmxfp4_c_sig_temp = {C_sig_i[23:0], 12'd0};

wire [`decimal_places0:0] C_sig_temp_shift = C_sig_temp >> c_shift_i;
wire [`decimal_places1:0] nvmxfp4_c_sig_temp_shift = nvmxfp4_c_sig_temp >> nvc_shift_i;

wire signed [`decimal_places0 + 1:0] C_sig_temp_s;
wire signed [`decimal_places1 + 1:0] nvmxfp4_c_sig_temp_s;

assign C_sig_temp_s = (C_sign_i) ? -$signed({1'b0,C_sig_temp_shift}) : $signed({1'b0,C_sig_temp_shift});
assign nvmxfp4_c_sig_temp_s = (C_sign_i) ? -$signed({1'b0, nvmxfp4_c_sig_temp_shift}) : $signed({1'b0, nvmxfp4_c_sig_temp_shift});

assign fp8_c_sig  = C_sig_temp_s;
assign nvmxfp4_c_sig = nvmxfp4_c_sig_temp_s;
    
endmodule
