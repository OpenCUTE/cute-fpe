`include "defines.sv"
// 注意对于UE4M3存在非规格化数和NAN，而UE8M0则只存在NAN，不存在非规格化数

module decode_scale (
    input  wire [`ScaleBusWidth - 1:0] scale_bit,
    input  wire        scale_opcode, // 0: UE8M0, 1: UE4M3
    input  wire        scale_en,

    output wire [7:0]  scale_exp  [0:`Scale_NUM - 1],
    output wire [3:0]  scale_sig  [0:`Scale_NUM - 1],
    output wire        scale_ZERO [0:`Scale_NUM - 1],
    output wire        scale_NAN
);

// ── 每8bit处理一个scale（共 Scale_NUM 个）──────────────────────
genvar i;
generate
    for (i = 0; i < `Scale_NUM; i = i + 1) begin : GEN_SCALE_DECODE

        wire [7:0] byte_data;
        wire nz_E4;
        wire zero;

        assign byte_data = scale_bit[i*8 +: 8];
        assign nz_E4     = |byte_data[6:3];
        assign zero      = !(|byte_data[6:0]);

        assign scale_ZERO[i] = scale_opcode && zero;

        // UE8M0: 每8bit直接作为指数，无尾数
        // UE4M3: 1bit隐藏位 + 3bit fraction
        assign scale_exp[i] = (!scale_en                   ) ? 8'd127 :
                              (scale_opcode == `SCALE_UE8M0) ? byte_data :
                              (nz_E4  || zero              ) ? {4'b0000, byte_data[6:3]} :
                                                               8'd1;

        assign scale_sig[i] = (scale_opcode == `SCALE_UE8M0) ?
                                4'b1000 :
                                {nz_E4, byte_data[2:0]};

    end
endgenerate

// ── NAN检测：任意一个8bit全1即为NAN ────────────────────────────
// 用 generate 替代手动展开，自动适配32/64bit
wire [`Scale_NUM - 1:0] nan_vec;
genvar j;
generate
    for (j = 0; j < `Scale_NUM; j = j + 1) begin : GEN_NAN
        //对于scale_opcode为1的情况，UE4M3的NAN是111_1111（不管符号位），而UE8M0的NAN是1111_1111
        assign nan_vec[j] = scale_opcode ? &scale_bit[j*8 +: 7] : &scale_bit[j*8 +: 8];
    end
endgenerate

assign scale_NAN = |nan_vec;

endmodule
