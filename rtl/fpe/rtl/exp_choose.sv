module exp_choose (
    input wire [5:0] exp_E6_fp  [0:3],
    input wire [5:0] exp_E6_int [0:3],
    input wire [5:0] exp_E6_intfp4 [0:3],

    input wire is_intfp4, is_Int8,

    output wire [5:0] exp_E6 [0:3]
);

    assign exp_E6[0] = is_Int8 ? exp_E6_int[0] : (is_intfp4 ? exp_E6_intfp4[0] : exp_E6_fp[0]);
    assign exp_E6[1] = is_Int8 ? exp_E6_int[1] : (is_intfp4 ? exp_E6_intfp4[1] : exp_E6_fp[1]);
    assign exp_E6[2] = is_Int8 ? exp_E6_int[2] : (is_intfp4 ? exp_E6_intfp4[2] : exp_E6_fp[2]);
    assign exp_E6[3] = is_Int8 ? exp_E6_int[3] : (is_intfp4 ? exp_E6_intfp4[3] : exp_E6_fp[3]);
    
endmodule
