`include "defines.sv"

module scale_deal (
    input  wire  [`ScaleBusWidth - 1 : 0] AScale_bits,
    input  wire  [`ScaleBusWidth - 1 : 0] BScale_bits,

    input  wire  scale_type,
    input  wire  scale_en  ,

    input  wire is_MXINTFP4,

    input wire [`Scale_NUM-1:0] MXINTFP4_all_zero,
    input wire [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero,

    output wire  [8:0] scale_exp [0:`Scale_NUM - 1],
    output wire  [7:0] scale_sig [0:`Scale_NUM - 1],
    output wire        scale_NAN
);

//A输出
wire [7:0]  Ascale_exp  [0:`Scale_NUM - 1] ;
wire [3:0]  Ascale_sig  [0:`Scale_NUM - 1] ;
wire        Ascale_ZERO [0:`Scale_NUM - 1] ; 
wire        Ascale_NAN       ;
 
//B输出 
wire [7:0]  Bscale_exp  [0:`Scale_NUM - 1] ;
wire [3:0]  Bscale_sig  [0:`Scale_NUM - 1] ;
wire        Bscale_ZERO [0:`Scale_NUM - 1] ;
wire        Bscale_NAN       ;


decode_scale u_decode_scale_A (
    .scale_bit    (AScale_bits),
    .scale_opcode (scale_type ),
    .scale_en     (scale_en   ),
    .scale_exp    (Ascale_exp  ),
    .scale_sig    (Ascale_sig  ),
    .scale_ZERO   (Ascale_ZERO ),
    .scale_NAN    (Ascale_NAN  )
);

decode_scale u_decode_scale_B (
    .scale_bit    (BScale_bits),
    .scale_opcode (scale_type ),
    .scale_en     (scale_en   ),
    .scale_exp    (Bscale_exp  ),
    .scale_sig    (Bscale_sig  ),
    .scale_ZERO   (Bscale_ZERO ),
    .scale_NAN    (Bscale_NAN  )
);

assign scale_NAN = scale_en ? (Ascale_NAN || Bscale_NAN) : 1'd0;

//将尾数和指数都相加起来
wire [8:0] scale_exp_raw [0:`Scale_NUM - 1];
wire [7:0] scale_sig_raw [0:`Scale_NUM - 1];

genvar i, j;
generate
    for (i = 0; i < `Scale_NUM; i = i + 1) begin : SCALE_RAW
        wire zero_mxfp86;
        if (i < `MXFP86_ZERO_GROUP_NUM) begin : GEN_MXFP86_ZERO
            assign zero_mxfp86 = MXFP86_all_zero[i];
        end else begin : GEN_UNUSED_MXFP86_ZERO
            assign zero_mxfp86 = 1'b0;
        end

        // MXFP8/FP6沿用原零掩码；FP4/INT4在scale映射到16元素block后处理。
        assign scale_exp_raw[i] = (!is_MXINTFP4 && zero_mxfp86)
            ? 9'd0 : {1'b0, Ascale_exp[i]} + {1'b0, Bscale_exp[i]};
        assign scale_sig_raw[i] = Ascale_sig[i] * Bscale_sig[i];
    end

    for (j = 0; j < `Scale_NUM; j = j + 1) begin : SCALE_REARRANGE
        // UE8M0每32元素共用一个scale；UE4M3每16元素使用一个scale。
        wire [8:0] mapped_exp = (scale_type == `SCALE_UE8M0)
            ? scale_exp_raw[j / 2] : scale_exp_raw[j];
        wire mapped_zero = (scale_type == `SCALE_UE8M0)
            ? (Ascale_ZERO[j / 2] || Bscale_ZERO[j / 2])
            : (Ascale_ZERO[j] || Bscale_ZERO[j]);

        // 零标志按目标block索引，不能先屏蔽raw scale再广播到两个block。
        assign scale_exp[j] = (is_MXINTFP4 && (MXINTFP4_all_zero[j] || mapped_zero))
            ? 9'd0 : mapped_exp;
        assign scale_sig[j] = (scale_type == `SCALE_UE8M0)
            ? scale_sig_raw[j / 2] : scale_sig_raw[j];
    end
endgenerate

//如果mxfp4跟nvfp4一样，每16个元素共用一个scale
// //现在有一个问题：如果是按照2x的情况下，我们就需要将 scale_exp_raw 和 scale_sig_raw 的值进行重排，才能得到最终的 scale_exp 和 scale_sig。具体来说，对于 SCALE_UE8M0 模式下，我们需要将每两个相邻的 raw 值合并为一个输出值，而对于 SCALE_UE4M3 模式下，我们直接使用 raw 值即可。
// genvar j;
// generate
//     for (j = 0; j < `Scale_NUM; j = j + 1) begin : SCALE_REARRANGE
//         assign scale_exp[j] = (scale_type == `SCALE_UE8M0 && !is_MXINTFP4) ? scale_exp_raw[j / 2]
//                                                              : scale_exp_raw[j];
//         assign scale_sig[j] = (scale_type == `SCALE_UE8M0 && !is_MXINTFP4) ? scale_sig_raw[j / 2]
//                                                              : scale_sig_raw[j];
//     end
// endgenerate

// //如果按照4x的情况下，我们只需要顺序直连就行了
// genvar j;
// generate
//     for (j = 0; j < `Scale_NUM; j = j + 1) begin : SCALE_REARRANGE
//         assign scale_exp[j] = scale_exp_raw[j];
//         assign scale_sig[j] = scale_sig_raw[j];
//     end
// endgenerate

endmodule
