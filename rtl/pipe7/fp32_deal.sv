`include "defines.sv"

module fp32_deal (
    input wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2):0] sign_fp8_i,
    input wire signed [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM):0] nvmxfp4_sig,

    // input wire [1:0]  data_type_i,
    input wire         is_MXINTFP4_i,

    input wire signed [9:0]  E_MAX_i,
    input wire signed [9:0]  block_max_i,

    output wire fp32_sign,
    output wire [23:0]  fp32_sig,
    output wire [9 :0]  fp32_exp
);

wire [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2) - 1:0] sign_fp8;

wire [`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM) - 1:0] nvmxfp4_sig_temp;

// assign fp32_sign = (data_type_i == `DATA_MXNVFP4) ? nvmxfp4_sig[`decimal_places1+`block_int_w+$clog2(`P4_ADD0_OUT_NUM) ] :
//                                                     sign_fp8_i[28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)] ;
assign fp32_sign = is_MXINTFP4_i ? nvmxfp4_sig[`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM) ] :
                                   sign_fp8_i[28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)] ;

assign sign_fp8 = (sign_fp8_i[28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)+$clog2(`ADD_NUM2)]) ? -sign_fp8_i : sign_fp8_i;

assign nvmxfp4_sig_temp = (nvmxfp4_sig[`decimal_places1+`block_int_w-1+$clog2(`P4_ADD0_OUT_NUM)]) ? -nvmxfp4_sig : nvmxfp4_sig;

wire [6:0] nvmxfp4_clz; 
wire [5:0]fp8_clz_temp;
wire [5:0] fp8_clz;

localparam P4_PAD_W = 64 - (`decimal_places1 + `block_int_w - 1 + $clog2(`P4_ADD0_OUT_NUM));

//检查nvmxfp4_sig的前面的零的个数
CLZ #(
    .LEN(64)
)fp4_sig_CLZ(
    .in({nvmxfp4_sig_temp, {P4_PAD_W{1'b0}}}),
    .out(nvmxfp4_clz)
);

generate
    if (`P4_ADD0_OUT_NUM == 8) begin : gen_fp32_width_8
        //检查fp8_sig的前面的零的个数
        CLZ #(
            .LEN(32)
        )fp8_sig_CLZ(
            .in (sign_fp8[31:0]),
            .out(fp8_clz_temp )
        );

        assign fp8_clz = sign_fp8[32] ? 6'd0 : (fp8_clz_temp + 6'd1);
    end
    else if (`P4_ADD0_OUT_NUM == 4) begin : gen_fp32_width_4
        //检查fp8_sig的前面的零的个数
        CLZ #(
            .LEN(32)
        )fp8_sig_CLZ(
            .in ({sign_fp8[31:0]}),
            .out(fp8_clz_temp )
        );
        assign fp8_clz = fp8_clz_temp;
    end
    else if (`P4_ADD0_OUT_NUM == 2) begin : gen_fp32_width_2
        //检查fp8_sig的前面的零的个数
        CLZ #(
            .LEN(32)
        )fp8_sig_CLZ(
            .in ({sign_fp8[30:0],1'd0}),
            .out(fp8_clz_temp )
        );
        assign fp8_clz = fp8_clz_temp;
    end
endgenerate

p7_shift u_p7_shift (
    .sign_fp8    (sign_fp8),
    .nvmxfp4_sig (nvmxfp4_sig_temp),

    .E_MAX_i     (E_MAX_i),
    .block_max_i (block_max_i),

    .nvmxfp4_clz (nvmxfp4_clz),
    .fp8_clz     (fp8_clz),

    // .data_type_i (data_type_i),
    .is_MXINTFP4_i (is_MXINTFP4_i),

    .fp32_sig    (fp32_sig),
    .fp32_exp    (fp32_exp)
);
    
endmodule
