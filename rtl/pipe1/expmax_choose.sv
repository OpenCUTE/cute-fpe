`include "defines.sv"

module expmax_choose (
    //因为我们只看是否为高精度还是低精度，所以我们只需要传入data_type的最低位就可以了
    // input data_type,
    input is_High_precision,

    input  logic [`BLOCK_NUM -1: 0]  mask_E3_0, mask_E3_1, mask_E6_0, mask_E6_1,

    output logic [`BLOCK_NUM -1: 0]  mask_0, mask_1
    
);

assign mask_0 = (is_High_precision) ? mask_E6_0 : mask_E3_0;
assign mask_1 = (is_High_precision) ? mask_E6_1 : mask_E3_1;


endmodule
