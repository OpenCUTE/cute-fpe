`include "defines.sv"

//C按照FP32来进行解码
module decode_C (
    // input  wire [1 :0] data_type,
    input  wire        is_INT8,
    input  wire        is_INT4,
    input  wire [31:0] C_bits,

    output wire [7 :0] C_exp,
    output wire [31:0] C_sig,
    output wire        C_sign,
    output wire C_NAN, C_NINF, C_PINF , C_ZERO
);

    // =========================
    // 基本字段拆分
    // =========================
    assign C_sign = C_bits[31];
    wire nz_c = |C_bits[30:23];
    assign C_exp  = (nz_c) ? C_bits[30:23] : 8'd1; // 非规格化数的exp为1

    // 尾数：补隐藏位
    // 正常数：1.xxx
    // 非规格化数：0.xxx
    assign C_sig = (is_INT8 || is_INT4) ? C_bits               :
                   (nz_c   ) ? {8'd0, 1'b1, C_bits[22:0]} :
                               {8'd0, 1'b0, C_bits[22:0]} ;

    // =========================
    // 特殊值判断
    // =========================
    wire nz_c_sig = |C_bits[22:0];

    // NaN: exp全1 且 frac非0
    assign C_NAN  = (&C_bits[30:23]) && nz_c_sig;

    // +INF: sign=0, exp全1, frac全0
    assign C_PINF = (!C_sign) && (&C_bits[30:23]) && (!nz_c_sig);

    // -INF: sign=1, exp全1, frac全0
    assign C_NINF = (C_sign) && (&C_bits[30:23]) && (!nz_c_sig);

    assign C_ZERO = !(nz_c || nz_c_sig);
    
endmodule
