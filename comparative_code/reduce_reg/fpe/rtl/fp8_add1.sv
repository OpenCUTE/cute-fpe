`include "defines.sv"

module fp8_add1 (
    input  wire  signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0]                   sign_after_add0 [0:`ADD_NUM2-1],
    output wire  signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0] sign_after_add1
);

generate
    if (`ADD_NUM2 == 4) begin : ADD4
        wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1):0] sum0, sum1;

        assign sum0 = sign_after_add0[0] + sign_after_add0[1];
        assign sum1 = sign_after_add0[2] + sign_after_add0[3];
        assign sign_after_add1 = sum0 + sum1;
        // assign sign_after_add1 = sign_after_add0[0] + sign_after_add0[1] + sign_after_add0[2] + sign_after_add0[3];
    end
    else if (`ADD_NUM2 == 2) begin : ADD2
        assign sign_after_add1 = sign_after_add0[0] + sign_after_add0[1];
    end
    else if (`ADD_NUM2 == 1) begin : ADD2
        assign sign_after_add1 = sign_after_add0[0];
    end
endgenerate

endmodule
