`include "defines.sv"

module decode_int8(
    input  wire [31:0] AVector, // 4x s8/u8
    input  wire [31:0] BVector,
    input  wire AS_have, BS_have,  // 1: s8; 0: u8
    //输出寄存器
    output wire [7 :0] a_8[0:1],
    output wire [3 :0] a_4[0:1],
    
    output wire [3 :0] b_4[0:1],
    output wire [7 :0] b_8[0:1],

    output wire [5:0] exp_E6 [0:3]
);

wire [7:0] INT8_A [0:3];
wire [7:0] INT8_B [0:3];

// 为每个字节提取和处理
// 如果S_have=1，取S8的绝对值；否则直接使用U8
genvar i;
generate
    for (i = 0; i < 4; i = i + 1) begin : Extract_Byte
        wire [7:0] Abyte_in = AVector[i*8+7:i*8];
        wire Ais_negative = Abyte_in[7] & AS_have;
        
        // 如果是负的S8，取绝对值（取反加1）；否则直接使用
        assign INT8_A[i] = Ais_negative ? (~Abyte_in + 1'b1) : Abyte_in;

        wire [7:0] Bbyte_in = BVector[i*8+7:i*8];
        wire Bis_negative = Bbyte_in[7] & BS_have;
        
        // 如果是负的S8，取绝对值（取反加1）；否则直接使用
        assign INT8_B[i] = Bis_negative ? (~Bbyte_in + 1'b1) : Bbyte_in;
    end
endgenerate

//每32个分配对应的寄存器
//第一个8b
assign a_8[0] = INT8_A[0];
assign b_8[0] = INT8_B[0];

//第二个8b
//高四位
assign a_4[0] = INT8_A[1][7:4];
assign b_4[0] = INT8_B[1][7:4];
//低四位
assign exp_E6[0] = {2'd0, INT8_A[1][3:0]};
assign exp_E6[2] = {2'd0, INT8_B[1][3:0]};

//第三个8b
assign a_8[1] = INT8_A[2];
assign b_8[1] = INT8_B[2];

//第四个8b
//高四位
assign a_4[1] = INT8_A[3][7:4];
assign b_4[1] = INT8_B[3][7:4];
//低四位
assign exp_E6[1] = {2'd0, INT8_A[3][3:0]};
assign exp_E6[3] = {2'd0, INT8_B[3][3:0]};

// //第一个8*8
// assign a_84[0] = INT8_A[0];
// assign b_84[0] = INT8_B[0][7:4];
// assign a_48[0] = INT8_B[0][3:0];
// assign b_48[0] = INT8_A[0];

// //第二个8*8
// assign a_88[0] = INT8_A[1];
// assign b_88[0] = INT8_B[1];
// assign a_44[0] = 4'd0;
// assign b_44[0] = 4'd0;

// //第三个8*8
// assign a_84[1] = INT8_A[2];
// assign b_84[1] = INT8_B[2][7:4];
// assign a_48[1] = INT8_B[2][3:0];
// assign b_48[1] = INT8_A[2];

// //第四个8*8
// assign a_88[1] = INT8_A[3];
// assign b_88[1] = INT8_B[3];
// assign a_44[1] = 4'd0;
// assign b_44[1] = 4'd0;


endmodule
