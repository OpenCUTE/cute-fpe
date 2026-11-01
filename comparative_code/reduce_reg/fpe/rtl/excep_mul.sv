`include "defines.sv"

module excep_mul(
    input wire [3:0] sign,
    input wire [3:0] A_NAN, A_INF, A_ZERO,
    input wire [3:0] B_NAN, B_INF, B_ZERO,

    output wire NAN_o, PINF_o, NINF_o
);
    wire [3:0] NAN, PINF, NINF;
    //向量按位操作（如 | & ^）会自动对每个对应位进行运算：
    // Rule 1: any NaN input => NaN (per bit)
    wire [3:0] any_nan = A_NAN | B_NAN;

    // Check whether each operand has infinity (per bit)
    wire [3:0] a_has_inf = A_INF;
    wire [3:0] b_has_inf = B_INF;

    // Rule 1 (extra): 0 * infinity => NaN (per bit)
    wire [3:0] zero_times_inf = (A_ZERO & b_has_inf) | (B_ZERO & a_has_inf);

    // NaN output: NaN input or 0*infinity (per bit)
    assign NAN = any_nan | zero_times_inf;

    // Check if any infinity exists (per bit)
    wire [3:0] any_inf = a_has_inf | b_has_inf;

    // Rule 2/3: if infinity exists (and not NaN), sign depends on count of -inf
    // Odd count of -inf => -inf, even count => +inf (per bit)
    wire [3:0] result_negative = sign;

    // +inf: any infinity, positive result, and not NaN (per bit)
    assign PINF = any_inf & ~result_negative & ~NAN;

    // -inf: any infinity, negative result, and not NaN (per bit)
    assign NINF = any_inf & result_negative & ~NAN;
    

    // Rule (1): any NaN, or both +inf and -inf exist => NaN
    wire any_nan_o  = |NAN;
    wire any_pinf_o = |PINF;
    wire any_ninf_o = |NINF;
    wire both_inf_o = any_pinf_o & any_ninf_o;

    assign NAN_o  = any_nan_o | both_inf_o;

    // Rule (2): no NaN, and any infinity => that infinity
    assign PINF_o = ~NAN_o & any_pinf_o;
    assign NINF_o = ~NAN_o & any_ninf_o;

endmodule
