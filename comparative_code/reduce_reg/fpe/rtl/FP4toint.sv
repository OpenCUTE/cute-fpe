`include "defines.sv"

// FP4 to fixed-point integer converter
// Input:  4-bit FP4 unsigned representation [sign(1), exp(2), mantissa(1)]
// Output: 4-bit signed integer per chunk (8 output chunks)

module FP4toint (
    input  wire [31:0] Vector,
    output wire [31:0] FP4_int
);

    function automatic signed [3:0] decode_fp4;
        input [3:0] fp;
        reg [1:0] exp;
        reg mantissa;
        reg nz;
        reg [3:0] sig;
        begin
            exp = fp[2:1];
            mantissa = fp[0];
            nz = |exp;
            sig = {2'b00, nz, mantissa};

            case (exp)
                2'd2: decode_fp4 = sig << 1;
                2'd3: decode_fp4 = sig << 2;
                default: decode_fp4 = sig;
            endcase
        end
    endfunction

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : decode_loop
            assign FP4_int[i*4 +: 4] = decode_fp4(Vector[i*4 +: 4]);
        end
    endgenerate

endmodule
