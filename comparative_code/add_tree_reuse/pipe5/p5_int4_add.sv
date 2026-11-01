`include "defines.sv"

// INT4 无 scale：独立归约各 block，避免重新进入浮点/INT8 归约树。
module p5_int4_add (
    input  wire signed [`BLOCK_SUM_W-1:0] block_sum_i [0:`Scale_NUM-1],
    output wire signed [`INT4_DOT_W-1:0] int4_dot
);
    localparam int LEVELS = $clog2(`Scale_NUM);
    for (genvar level = 0; level <= LEVELS; level++) begin : gen_level
        wire signed [`BLOCK_SUM_W+level-1:0] sum [0:(`Scale_NUM >> level)-1];
        for (genvar node = 0; node < (`Scale_NUM >> level); node++) begin : gen_node
            if (level == 0) begin : gen_leaf
                assign sum[node] = block_sum_i[node];
            end else begin : gen_add
                assign sum[node] = $signed(gen_level[level-1].sum[2*node]) +
                                   $signed(gen_level[level-1].sum[2*node+1]);
            end
        end
    end
    assign int4_dot = gen_level[LEVELS].sum[0];
endmodule
