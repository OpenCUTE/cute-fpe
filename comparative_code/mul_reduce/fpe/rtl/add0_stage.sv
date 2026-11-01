`include "defines.sv"

module add0_stage (
    // 上一级寄存结果
    input wire [4:0]  exp_shift_out_i [0:`LANE_NUM - 1][0:3],

    // 穿透信号
    input  wire [21:0] product_out_i [0:`LANE_NUM-1][0:1],

    input  wire [3:0]  vec4_sign_i   [0:`LANE_NUM-1],
    // input  wire [1:0]  data_type_i,
    input  wire        is_High_precision_i,
    input  wire        is_low_precision_i,
    input  wire        is_MXINTFP4_i,
    // input  wire        is_INT8_i,

    // 本级输出
    output wire signed [28+$clog2(`ADD_NUM0)+$clog2(`ADD_NUM1)-1:0] sign_after_add0 [0:`P4_ADD0_OUT_NUM-1]
);
wire signed [27+$clog2(`ADD_NUM0):0] sign_after_add [0:`LANE_NUM-1];

//先移位后进行一次加法
genvar i;

generate
    for(i = 0; i < `LANE_NUM; i = i + 1) begin
        wire signed [27:0] sign_sig_w [0:3];

        sign_shift u_sign_shift (
            .product_i   (product_out_i[i]),
            .is_High_precision(is_High_precision_i),
            .is_low_precision (is_low_precision_i ),
            .is_MXINTFP4      (is_MXINTFP4_i      ),
            // .is_INT8          (is_INT8_i          ),
            .exp_shift (exp_shift_out_i[i]),
            .vec4_sign (vec4_sign_i[i]),
            .sign_sig  (sign_sig_w)
        );

        sign_add u_sign_add (
            .sign_sig       (sign_sig_w),
            .sign_after_add (sign_after_add[i])
        );

    end
endgenerate

// -------------------------------------------------------
// 重复 LANE_NUM/ADD_NUM1 次：p4_add0 例化
// 每个实例消耗 ADD_NUM1 路输入，产生 1 路输出
// -------------------------------------------------------
generate
    for (i = 0; i < `P4_ADD0_OUT_NUM; i++) begin : gen_p4_add0
        p4_add0 u_p4_add0 (
            // 从 sign_after_add_i 中切出连续的 ADD_NUM1 路
            .sign_after_add_i (sign_after_add[i*`ADD_NUM1 : i*`ADD_NUM1+`ADD_NUM1-1]),
            .sign_after_add0  (sign_after_add0[i])
        );
    end
endgenerate

endmodule
