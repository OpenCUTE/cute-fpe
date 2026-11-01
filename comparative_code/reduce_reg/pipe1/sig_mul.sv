`include "defines.sv"

module sig_mul (
    input  wire [7 :0] a_84[0:1], a_88[0:1],
    input  wire [3 :0] a_48[0:1], a_44[0:1],
    input  wire [3 :0] b_84[0:1], b_44[0:1],
    input  wire [7 :0] b_48[0:1], b_88[0:1],

    output wire [11:0] product0_0, product0_1,
    output wire [15:0] product1_0,
    output wire [7 :0] product1_1,

    output wire [11:0] product2_0, product2_1,
    output wire [15:0] product3_0,
    output wire [7 :0] product3_1
);

assign product0_0 = a_84[0] * b_84[0];
assign product0_1 = a_48[0] * b_48[0];
assign product1_0 = a_88[0] * b_88[0];
assign product1_1 = a_44[0] * b_44[0];

assign product2_0 = a_84[1] * b_84[1];
assign product2_1 = a_48[1] * b_48[1];
assign product3_0 = a_88[1] * b_88[1];
assign product3_1 = a_44[1] * b_44[1];
    
endmodule
