`include "defines.sv"

module p4_add0 (
    input  wire  signed [27+$clog2(`ADD_NUM0):0]                     sign_after_add_i [0:`ADD_NUM1-1],
    output wire  signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0] sign_after_add0
);

generate
    if (`ADD_NUM1 == 4) begin : ADD4
        wire signed [28+$clog2(`ADD_NUM0):0] sum0, sum1;

        assign sum0 = sign_after_add_i[0] + sign_after_add_i[1];
        assign sum1 = sign_after_add_i[2] + sign_after_add_i[3];
        assign sign_after_add0 = sum0 + sum1;
    end
    else if (`ADD_NUM1 == 2) begin : ADD2
        assign sign_after_add0 = sign_after_add_i[0] + sign_after_add_i[1];
    end
endgenerate

endmodule
