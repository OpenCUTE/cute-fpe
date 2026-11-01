// -----------------------------------------------------------------------------
// CmpTreeP1: 处理低 (EXP_W - LAYERS) 位，拼出完整最大指数
// -----------------------------------------------------------------------------
module CmpTreeP1 #(
    parameter integer LAYERS   = 6,
    parameter integer ELEM_NUM = 16,
    parameter integer EXP_W    = 9
)(
    input  logic [ELEM_NUM-1:0]  part1,
    input  logic [ELEM_NUM-1:0]  part2  [0:EXP_W-LAYERS-1],

    output logic [EXP_W - LAYERS -1:0]   out
);
 
    localparam integer LOW_BITS = EXP_W - LAYERS;
 
    // -------------------------------------------------------------------------
    // Step 1: 继续低位 mask 筛选，初始候选接续 P0 的 part1
    // -------------------------------------------------------------------------
    logic [ELEM_NUM-1:0] mask [0:LOW_BITS-1];
 
    always_comb begin
        logic [ELEM_NUM-1:0] hit0;
        hit0   = part1 & part2[0];
        mask[0] = (hit0 == '0) ? part1 : hit0;
 
        for (int k = 1; k < LOW_BITS; k++) begin
            logic [ELEM_NUM-1:0] hit;
            hit     = mask[k-1] & part2[k];
            mask[k] = (hit == '0) ? mask[k-1] : hit;
        end
    end
 
    // -------------------------------------------------------------------------
    // Step 2: 生成低位最大指数值
    // -------------------------------------------------------------------------
    logic [LOW_BITS-1:0] res_low;
 
    always_comb begin
        for (int k = 0; k < LOW_BITS; k++) begin
            res_low[LOW_BITS-1-k] = |(mask[LOW_BITS-1] & part2[k]);
        end
    end
 
    // -------------------------------------------------------------------------
    // Step 3: 拼接高位 part3 + 低位 res_low → 完整最大指数
    // -------------------------------------------------------------------------
    assign out = res_low;
 
endmodule
