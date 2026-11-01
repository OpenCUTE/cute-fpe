`include "defines.sv"

module exp_sum (
    //传入opcode
    input wire is_E8, is_FP16,
    //判断是否为0
    input wire [3 : 0] Azero, Bzero,

    //指数组
    input wire signed [4:0] Aexp_E5 [0 :3],
    input wire signed [4:0] Bexp_E5 [0 :3],

    output wire[5:0] exp_E6 [0 :3]
);
//判断这两个里面是否存在0
wire [3:0] zero = Azero | Bzero;

wire [8:0] E9_0 = zero[0] ? 9'd0 : {Aexp_E5[2], Aexp_E5[0][2:0]} + {Bexp_E5[2], Bexp_E5[0][2:0]};
wire [8:0] E9_1 = zero[1] ? 9'd0 : {Aexp_E5[3], Aexp_E5[1][2:0]} + {Bexp_E5[3], Bexp_E5[1][2:0]};

wire [5:0] E6_0 = zero[0] ? 6'd0 : (Aexp_E5[0] + Bexp_E5[0]);
wire [5:0] E6_1 = zero[1] ? 6'd0 : (Aexp_E5[1] + Bexp_E5[1]);
wire [5:0] E6_2 = zero[2] ? 6'd0 : (Aexp_E5[2] + Bexp_E5[2]);
wire [5:0] E6_3 = zero[3] ? 6'd0 : (Aexp_E5[3] + Bexp_E5[3]);

//将两个E9的高6位存放在E6_3与E6_2，将低3位存放在E6_1-E6_0（放置在低位，高位直接补0即可）
assign exp_E6[0] = (is_E8 || is_FP16) ? {3'd0, E9_0[2:0]} : E6_0 ;
assign exp_E6[1] = (is_E8 || is_FP16) ? {3'd0, E9_1[2:0]} : E6_1 ;

assign exp_E6[2] = (is_E8 || is_FP16) ? E9_0[8:3] : E6_2 ;
assign exp_E6[3] = (is_E8 || is_FP16) ? E9_1[8:3] : E6_3 ;

endmodule
