`include "defines.sv"

module decode_int4 (
    input wire [31:0]  AVector_bits, // 8x s4/u4
    input wire [31:0]  BVector_bits,
    
    input wire         AS_have, BS_have,  // 1: s4; 0: u4

    output wire [31:0] int4_Avector,
    output wire [31:0] int4_Bvector
    
    // //输出寄存器
    // output wire [7 :0] a_8[0:1],
    // output wire [3 :0] a_4[0:1],

    // output wire [3 :0] b_4[0:1],
    // output wire [7 :0] b_8[0:1],

    // output wire [5:0] exp_E6 [0:3]
);

// wire [3:0] INT4_A [0:7];
// wire [3:0] INT4_B [0:7];

// 为每个4位提取和处理
// 如果S_have=1，取S4的绝对值；否则直接使用U4
genvar i;
generate
    for (i = 0; i < 8; i = i + 1) begin : Extract_4bit
        wire [3:0] A4bit_in = AVector_bits[i*4+3:i*4];
        wire Ais_negative = A4bit_in[3] & AS_have;

        // 如果是负的S4，取绝对值（取反加1）；否则直接使用
        assign int4_Avector[i*4+3:i*4] = Ais_negative ? (~A4bit_in + 1'b1) : A4bit_in;

        wire [3:0] B4bit_in = BVector_bits[i*4+3:i*4];
        wire Bis_negative = B4bit_in[3] & BS_have;

        // 如果是负的S4，取绝对值（取反加1）；否则直接使用
        assign int4_Bvector[i*4+3:i*4] = Bis_negative ? (~B4bit_in + 1'b1) : B4bit_in;
    end
endgenerate

// //每32个分配对应的寄存器
// //第1个和第2个4b
// assign a_8[0] = {INT4_A[1], INT4_A[0]};
// assign b_8[0] = {INT4_B[1], INT4_B[0]};

// //第3个4b
// assign a_4[0] = INT4_A[2];
// assign b_4[0] = INT4_B[2];

// //第4个4b
// assign exp_E6[0] = {2'd0, INT4_A[3]};
// assign exp_E6[2] = {2'd0, INT4_B[3]};

// //第5个和第6个4b
// assign a_8[1] = {INT4_A[5], INT4_A[4]};
// assign b_8[1] = {INT4_B[5], INT4_B[4]};

// //第7个4b
// assign a_4[1] = INT4_A[6];
// assign b_4[1] = INT4_B[6];

// //第8个4b
// assign exp_E6[1] = {2'd0, INT4_A[7]};
// assign exp_E6[3] = {2'd0, INT4_B[7]};
    
endmodule
