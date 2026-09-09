`include "defines.sv"

module p6_block_add (
    input  wire signed [`decimal_places1 + `block_int_w -1:0] block_sig_af_shift_i [0:`P4_ADD0_OUT_NUM-1],
    output wire signed [`decimal_places1 + `block_int_w -1 + $clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig
);

generate
    if (`P4_ADD0_OUT_NUM == 8) begin : GEN_ADD8
        wire signed [`decimal_places1 + `block_int_w:0]   add_l1 [0:3];
        wire signed [`decimal_places1 + `block_int_w+1:0] add_l2 [0:1];
        wire signed [`decimal_places1 + `block_int_w+2:0] add_l3;

        assign add_l1[0] = block_sig_af_shift_i[0] + block_sig_af_shift_i[1];
        assign add_l1[1] = block_sig_af_shift_i[2] + block_sig_af_shift_i[3];
        assign add_l1[2] = block_sig_af_shift_i[4] + block_sig_af_shift_i[5];
        assign add_l1[3] = block_sig_af_shift_i[6] + block_sig_af_shift_i[7];

        assign add_l2[0] = add_l1[0] + add_l1[1];
        assign add_l2[1] = add_l1[2] + add_l1[3];

        assign add_l3 = add_l2[0] + add_l2[1];
        assign nvmxfp4_sig = add_l3;
    end
    else if (`P4_ADD0_OUT_NUM == 4) begin : GEN_ADD4
        wire signed [`decimal_places1 + `block_int_w:0]   add_l1 [0:1];
        wire signed [`decimal_places1 + `block_int_w+1:0] add_l2;

        assign add_l1[0] = block_sig_af_shift_i[0] + block_sig_af_shift_i[1];
        assign add_l1[1] = block_sig_af_shift_i[2] + block_sig_af_shift_i[3];

        assign add_l2 = add_l1[0] + add_l1[1];
        assign nvmxfp4_sig = add_l2;
    end
    else if (`P4_ADD0_OUT_NUM == 2) begin : GEN_ADD2
        wire signed [`decimal_places1 + `block_int_w:0] add_l1;

        assign add_l1 = block_sig_af_shift_i[0] + block_sig_af_shift_i[1];
        assign nvmxfp4_sig = add_l1;
    end
    else begin : GEN_ADD_DEFAULT
        assign nvmxfp4_sig = '0;
    end
endgenerate

endmodule
