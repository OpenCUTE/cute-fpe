`include "defines.sv"

// 独立的四位数据组内归约树：每个 block 对应连续 16 个输入元素。
// 每 lane 含 8 个元素；product_out[lane][0/1] 各有两个 10 位有符号部分和，
// 存在 11 位槽的 [9:0] 和 [20:11]，槽的最高位为符号扩展位。
// block b 使用 lane 2*b 和 2*b+1 的两路 product，与 scale_sig[b] 对齐。
module p2_nvfp4_add (
    input  wire [21:0] product_out [0:`LANE_NUM-1][0:1],
    output wire signed [`BLOCK_SUM_W-1:0] block_sum [0:`Scale_NUM-1]
);
    localparam int LEVELS = $clog2(`FP4P1AddNum);

    for (genvar block_idx = 0; block_idx < `Scale_NUM; block_idx++) begin : gen_block
        for (genvar level = 0; level <= LEVELS; level++) begin : gen_level
            wire signed [`FP4_PAIR_SUM_W+level-1:0] sum [0:(`FP4P1AddNum >> level)-1];
            for (genvar node = 0; node < (`FP4P1AddNum >> level); node++) begin : gen_node
                if (level == 0) begin : gen_leaf
                    localparam int LANE = block_idx * (`FP4P1AddNum / 4) + node / 4;
                    localparam int PROD = (node % 4) / 2;
                    localparam int LSB = (node % 2) * 11;
                    assign sum[node] = $signed(product_out[LANE][PROD][LSB +: `FP4_PAIR_SUM_W]);
                end else begin : gen_add
                    assign sum[node] = $signed(gen_level[level-1].sum[2*node]) +
                                       $signed(gen_level[level-1].sum[2*node+1]);
                end
            end
        end
        assign block_sum[block_idx] = gen_level[LEVELS].sum[0];
    end
endmodule
