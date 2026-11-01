`include "defines.sv"

module sign_add(
    input  wire signed [27:0] sign_sig [0:`ADD_NUM0-1],
    output wire signed [27+$clog2(`ADD_NUM0):0] sign_after_add
);

generate
    if (`ADD_NUM0 == 4) begin : ADD4
        // wire signed [28:0] sum0, sum1;

        // assign sum0 = sign_sig[0] + sign_sig[1];
        // assign sum1 = sign_sig[2] + sign_sig[3];
        // assign sign_after_add = sum0 + sum1;
        assign sign_after_add = sign_sig[0] + sign_sig[1] + sign_sig[2] + sign_sig[3];
    end
    else if (`ADD_NUM0 == 2) begin : ADD2
        assign sign_after_add = sign_sig[0] + sign_sig[1];
    end
    else if (`ADD_NUM0 == 1) begin : ADD2
        assign sign_after_add = sign_sig[0];
    end
endgenerate

endmodule
