`include "defines.sv"

module p6_add (
    input  wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0] sign_after_add1_i  [0:`P4_ADD1_OUT_NUM-1],
    input  wire signed [`decimal_places0 + 1: 0] fp8_c_sig,
    input  wire signed [31:0] c_int32,

    // input  wire [1:0] data_type,
    input  wire        is_INT,

    output wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0] sign_fp8
);

assign sign_fp8 = (is_INT) ? sign_after_add1_i[0] + sign_after_add1_i[1] + c_int32   :
                             sign_after_add1_i[0] + sign_after_add1_i[1] + fp8_c_sig ;
    
endmodule
