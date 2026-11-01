`include "defines.sv"

module add2_stage (
    input  wire [5:0]  nvc_shift_i,
    input  wire [4:0]  c_shift_i,

    input  wire [31:0] C_sig_i,
    input  wire        C_sign_i,

    // input  wire [1 :0] data_type_i,
    input  wire        is_INT_i,

    input  wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)-1:0] sign_after_add1_i [0:`P4_ADD1_OUT_NUM-1],
    input  wire signed [`decimal_places1+ `block_int_w -1:0]                          block_sig_af_shift_i [0:`P4_ADD0_OUT_NUM-1],

    //输出
    output wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0]   sign_fp8,
    output wire signed [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig
);

// -------------------------------------------------------
// 中间信号
// -------------------------------------------------------
wire signed [`decimal_places0 + 1:0]  fp8_c_sig;


wire signed [`decimal_places1 + 1:0]  nvmxfp4_c_sig;
wire signed [`decimal_places1+ `block_int_w -1+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig_i;

// -------------------------------------------------------
// 例化 c_shift
// -------------------------------------------------------
c_shift u_c_shift (
    .C_sig_i        (C_sig_i),
    .C_sign_i       (C_sign_i),
    .nvc_shift_i    (nvc_shift_i),
    .c_shift_i      (c_shift_i),

    .fp8_c_sig      (fp8_c_sig),
    .nvmxfp4_c_sig  (nvmxfp4_c_sig)
);

// -------------------------------------------------------
// 例化 p6_add
// -------------------------------------------------------
p6_add u_p6_add (
    .sign_after_add1_i  (sign_after_add1_i),
    .fp8_c_sig          (fp8_c_sig),
    .c_int32            (C_sig_i  ),
    .is_INT             (is_INT_i ),

    .sign_fp8           (sign_fp8)
);

// -------------------------------------------------------
// 例化 p5_block_add
// -------------------------------------------------------
p6_block_add u_p5_block_add (
    .block_sig_af_shift_i  (block_sig_af_shift_i),

    .nvmxfp4_sig           (nvmxfp4_sig_i)
);

// -------------------------------------------------------
// 例化 nvfp4_last_add
// 将 nvmxfp4_c_sig 与 nvmxfp4_sig 相加，得到最终 nvmxfp4 尾数
// -------------------------------------------------------
nvfp4_last_add u_nvfp4_last_add (
    .nvmxfp4_c_sig_i  (nvmxfp4_c_sig),
    .nvmxfp4_sig_i    (nvmxfp4_sig_i),

    .nvmxfp4_sig      (nvmxfp4_sig)
);


endmodule
