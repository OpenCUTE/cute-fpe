`include "defines.sv"
//本模块，我们对分组后得到的定点数进行与scale的尾数乘法，然后再根据前面的移位值来进行尾数的移位。

// 对于mxnvfp4而言，其只有两个小数点，而对于mxint而言，其本身存在四个小数点
// 对于e4m3而言，存在3个小数点，对于ue8m0,我们在前面也让他为1000，从而存在三个小数点
// 两个scale相乘之后就存在了六个小数点

//除此之外，在进行scale与最后加法值的乘法之前，我们需要将其都转化为正数，
//一方面可以使得乘法器位宽变小，另一方面输入的时候将小位宽的有符号数变成绝对值，比乘完之后的大位宽的变成绝对值，ppa表现更好

module block_mul_shift (
    input  wire signed [12:0] sign_after_add0 [0 : `P4_ADD0_OUT_NUM-1],//对于16个元素一组而言，有效数据是1bit符号位+12bit数据
    input  wire        [7:0] scale_sig_i [0:`Scale_NUM - 1],
    input  wire        is_mxint4,

    input  wire         [5:0]  block_shift_i [0:`Scale_NUM - 1],

    output wire signed [`decimal_places1+ `block_int_w -1 :0] block_sig_af_shift [0: `P4_ADD0_OUT_NUM-1]
);
//先得到sign_after_add0的无符号的绝对值
wire        sign_after_add0_sign [0:`P4_ADD0_OUT_NUM-1];
wire [12:0] sign_after_add0_abs [0:`P4_ADD0_OUT_NUM-1];//其实这里可以用12位来存储，因为按照我们现在的编码，这个地方不可能为最小值-4096

// 输入位宽                                       
localparam MUL_W  = 21;
wire [MUL_W-1:0] mul_full [0:`P4_ADD0_OUT_NUM-1]; // 全精度乘法结果（无符号）

wire [`decimal_places1 + `block_int_w -1 :0] FP4_block_sig [0: `P4_ADD0_OUT_NUM-1];

genvar i;
generate
    for (i = 0; i < `P4_ADD0_OUT_NUM; i = i + 1) begin : GEN_ABS
        assign sign_after_add0_sign[i] = sign_after_add0[i][12];

        assign sign_after_add0_abs[i] = sign_after_add0[i][12] ? $unsigned(-$signed(sign_after_add0[i][12:0]))
                                                               : $unsigned( sign_after_add0[i][12:0]);

        //计算绝对值的尾数乘积
        assign mul_full[i] = scale_sig_i[i] * sign_after_add0_abs[i];
        assign FP4_block_sig[i] = is_mxint4 ? {1'd0, sign_after_add0_abs[i], 31'd0} : {mul_full[i], 27'd0};

        // wire FP4_sign = FP4_block_sig[i][`decimal_places1+ `block_int_w -1];

        // wire [`decimal_places1+ `block_int_w -1 :0] FP4_block_sig_temp = FP4_sign ? -$signed(FP4_block_sig[i]) : $signed(FP4_block_sig[i]);

        wire [`decimal_places1+ `block_int_w -2 :0] block_sig_af_shift_temp = FP4_block_sig[i][`decimal_places1+ `block_int_w -2 :0] >> block_shift_i[i];
        assign block_sig_af_shift[i] = sign_after_add0_sign[i] ? -$signed({1'd0, block_sig_af_shift_temp}) : $signed({1'd0, block_sig_af_shift_temp});
 
    end
endgenerate

    
endmodule
