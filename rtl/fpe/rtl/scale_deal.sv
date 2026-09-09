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
    for (i = 0; i < `MXFP86_ZERO_GROUP_NUM; i = i + 1) begin : SCALE_ADD_MXFP86
        assign scale_exp_raw[i] =
            (((Ascale_ZERO[i] || Bscale_ZERO[i] || MXINTFP4_all_zero[i]) &&
              is_MXINTFP4) || MXFP86_all_zero[i])
            ? 9'd0
            : {1'b0, Ascale_exp[i]} + {1'b0, Bscale_exp[i]};

        assign scale_sig_raw[i] = Ascale_sig[i] * Bscale_sig[i];
    end

    for (j = `MXFP86_ZERO_GROUP_NUM; j < `Scale_NUM; j = j + 1) begin : SCALE_ADD_REST
        assign scale_exp_raw[j] =
            ((Ascale_ZERO[j] || Bscale_ZERO[j] || MXINTFP4_all_zero[j]) &&
             is_MXINTFP4)
            ? 9'd0
            : {1'b0, Ascale_exp[j]} + {1'b0, Bscale_exp[j]};

        assign scale_sig_raw[j] = Ascale_sig[j] * Bscale_sig[j];
    end
endgenerate

// SCALE_UE8M0 模式下对 scale_exp 进行位置重排
// 低位存储规则：每个有效 scale 对应连续两个通道（每个值广播到相邻两槽）
// Scale_NUM==4: 存储 00ba → 输出 bbaa，即 out[i] = raw[i/2]
// Scale_NUM==8: 存储 0000dcba → 输出 ddccbbaa，即 out[i] = raw[i/2]

//mxfp4每32个元素共用1个scale，因为按照道理来说
//现在有一个问题：如果是按照2x的情况下，我们就需要将 scale_exp_raw 和 scale_sig_raw 的值进行重排，才能得到最终的 scale_exp 和 scale_sig。具体来说，对于 SCALE_UE8M0 模式下，我们需要将每两个相邻的 raw 值合并为一个输出值，而对于 SCALE_UE4M3 模式下，我们直接使用 raw 值即可。
generate
    for (j = 0; j < `Scale_NUM; j = j + 1) begin : SCALE_REARRANGE
        assign scale_exp[j] = (scale_type == `SCALE_UE8M0) ? scale_exp_raw[j / 2]
                                                            : scale_exp_raw[j];
        assign scale_sig[j] = (scale_type == `SCALE_UE8M0) ? scale_sig_raw[j / 2]
                                                            : scale_sig_raw[j];
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
