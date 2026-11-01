// -----------------------------------------------------------------------------
// CmpTreeP0: 处理高 LAYERS 位
// -----------------------------------------------------------------------------
module CmpTreeP63_96 #(
    parameter integer LAYERS   = 6,
    parameter integer ELEM_NUM = 16,
    parameter integer EXP_W    = 9
)(
    input  logic [EXP_W-1:0]              in     [0:ELEM_NUM-1],
    output logic [LAYERS-1:0]             part3
);
 
    // -------------------------------------------------------------------------
    // Step 1: 数据转置
    //   inVec0[j][i] = in[i][EXP_W-1-j]
    //   即把"按元素"存储转成"按 bit 位"存储
    // -------------------------------------------------------------------------
    logic [ELEM_NUM-1:0] inVec0 [0:EXP_W-1];
 
    always_comb begin
        for (int j = 0; j < EXP_W; j++) begin
            for (int i = 0; i < ELEM_NUM; i++) begin
                inVec0[j][i] = in[i][EXP_W-1-j];
            end
        end
    end
 
    // -------------------------------------------------------------------------
    // Step 2: 逐 bit 筛选候选集合 mask
    //   mask[k] = 经过前 k+1 位筛选后，仍可能是最大值的元素集合
    // -------------------------------------------------------------------------
    logic [ELEM_NUM-1:0] mask [0:LAYERS-1];
 
    always_comb begin
        // 第 0 层：看最高位，有 1 则只保留有 1 的，全 0 则全保留
        mask[0] = (inVec0[0] == '0) ? '1 : inVec0[0];
 
        for (int k = 1; k < LAYERS; k++) begin
            logic [ELEM_NUM-1:0] hit;
            hit = mask[k-1] & inVec0[k];
            // 候选中有人这位是 1 → 收窄；否则保持候选不变
            mask[k] = (hit == '0) ? mask[k-1] : hit;
        end
    end
 
    // -------------------------------------------------------------------------
    // Step 5: part3 —— 已确定的高位最大指数值
    //   对每一位 k，用最终候选 mask 与该位数据相与，非零则该位为 1
    // -------------------------------------------------------------------------
    always_comb begin
        for (int k = 0; k < LAYERS; k++) begin
            part3[LAYERS-1-k] = |(mask[LAYERS-1] & inVec0[k]);
        end
    end
 
endmodule
