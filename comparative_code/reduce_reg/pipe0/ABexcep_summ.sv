`include "defines.sv"

module ABexcep_summ(
    input  wire [`LANE_NUM-1 : 0] NAN_IN,
    input  wire [`LANE_NUM-1 : 0] PINF_IN,
    input  wire [`LANE_NUM-1 : 0] NINF_IN,
    output wire NAN,
    output wire PINF,
    output wire NINF
);
    // Rule (1): any NaN, or both +inf and -inf exist => NaN
    wire any_nan  = |NAN_IN;
    wire any_pinf = |PINF_IN;
    wire any_ninf = |NINF_IN;
    wire both_inf = any_pinf & any_ninf;

    assign NAN  = any_nan | both_inf;

    // Rule (2): no NaN, and any infinity => that infinity
    assign PINF = ~NAN & any_pinf;
    assign NINF = ~NAN & any_ninf;

endmodule
