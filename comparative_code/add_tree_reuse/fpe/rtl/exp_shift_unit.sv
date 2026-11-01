`include "defines.sv"

module exp_shift_unit (
    input  wire [9:0] E_MAX_shift,
    // input  wire [1:0] data_type,
    input  wire       is_High_precision,
    input  wire       is_low_precision,
    input  wire       is_INT8,
    input  wire [5:0] exp_E6 [0:3],

    output wire [4:0] exp_shift [0:3]
);

    wire [8:0] diff_hp_0 = E_MAX_shift[8:0] - {exp_E6[2], exp_E6[0][2:0]};
    wire [8:0] diff_hp_1 = E_MAX_shift[8:0] - {exp_E6[3], exp_E6[1][2:0]};
    wire [6:0] diff_lp_0 = (|E_MAX_shift[9:7]) ? 7'd31 : E_MAX_shift[6:0] - {1'd0, exp_E6[0]};
    wire [6:0] diff_lp_1 = (|E_MAX_shift[9:7]) ? 7'd31 : E_MAX_shift[6:0] - {1'd0, exp_E6[1]};
    wire [6:0] diff_lp_2 = (|E_MAX_shift[9:7]) ? 7'd31 : E_MAX_shift[6:0] - {1'd0, exp_E6[2]};
    wire [6:0] diff_lp_3 = (|E_MAX_shift[9:7]) ? 7'd31 : E_MAX_shift[6:0] - {1'd0, exp_E6[3]};

    wire [4:0] diff_hp_0_temp = |diff_hp_0[8:5] ? 5'b11111 : diff_hp_0[4:0];
    wire [4:0] diff_hp_1_temp = |diff_hp_1[8:5] ? 5'b11111 : diff_hp_1[4:0];
    wire [4:0] diff_lp_0_temp = |diff_lp_0[6:5] ? 5'b11111 : diff_lp_0[4:0];
    wire [4:0] diff_lp_1_temp = |diff_lp_1[6:5] ? 5'b11111 : diff_lp_1[4:0];
    wire [4:0] diff_lp_2_temp = |diff_lp_2[6:5] ? 5'b11111 : diff_lp_2[4:0];
    wire [4:0] diff_lp_3_temp = |diff_lp_3[6:5] ? 5'b11111 : diff_lp_3[4:0];

    assign exp_shift[0] = (is_INT8)           ? exp_E6[0][4:0] :
                          (is_High_precision) ? diff_hp_0_temp : 
                          (is_low_precision ) ? diff_lp_0_temp : 5'd0;

    assign exp_shift[1] = (is_INT8)           ? exp_E6[1][4:0] : 
                          (is_High_precision) ? diff_hp_1_temp : 
                          (is_low_precision ) ? diff_lp_1_temp : 5'd0;

    assign exp_shift[2] = (is_INT8)           ? exp_E6[2][4:0] : 
                          (is_low_precision ) ? diff_lp_2_temp : 5'd0;

    assign exp_shift[3] = (is_INT8)           ? exp_E6[3][4:0] : 
                          (is_low_precision ) ? diff_lp_3_temp : 5'd0;

endmodule
