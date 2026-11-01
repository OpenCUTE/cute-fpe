`include "defines.sv"

//用于重新分配FP4/INT4的尾数寄存器
module assign_INTFP4(
    //fp4
    input  wire [31:0] AFP4_int, BFP4_int,

    //int4
    input  wire [31:0] int4_Avector, int4_Bvector,

    input  wire is_mxnvfp4,
    //输出寄存器
    output wire [7 :0] a_8[0:1],
    output wire [3 :0] a_4[0:1],

    output wire [3 :0] b_4[0:1],
    output wire [7 :0] b_8[0:1],

    output wire [5:0] exp_E6 [0:3]
);

// 第1个和第2个4b
assign a_8[0] = is_mxnvfp4 ?  {AFP4_int[7:4], AFP4_int[3:0]} : {int4_Avector[7:4], int4_Avector[3:0]};
assign b_8[0] = is_mxnvfp4 ?  {BFP4_int[7:4], BFP4_int[3:0]} : {int4_Bvector[7:4], int4_Bvector[3:0]};

//第3个4b
assign a_4[0] = is_mxnvfp4 ? AFP4_int[11:8] : int4_Avector[11:8];
assign b_4[0] = is_mxnvfp4 ? BFP4_int[11:8] : int4_Bvector[11:8];

//第4个4b
assign exp_E6[0] = is_mxnvfp4 ? {2'd0, AFP4_int[15:12]} : {2'd0, int4_Avector[15:12]};
assign exp_E6[2] = is_mxnvfp4 ? {2'd0, BFP4_int[15:12]} : {2'd0, int4_Bvector[15:12]};

//第5个和第6个4b
assign a_8[1] = is_mxnvfp4 ? {AFP4_int[23:20], AFP4_int[19:16]} : {int4_Avector[23:20], int4_Avector[19:16]};
assign b_8[1] = is_mxnvfp4 ? {BFP4_int[23:20], BFP4_int[19:16]} : {int4_Bvector[23:20], int4_Bvector[19:16]};

//第7个4b
assign a_4[1] = is_mxnvfp4 ? AFP4_int[27:24] : int4_Avector[27:24];
assign b_4[1] = is_mxnvfp4 ? BFP4_int[27:24] : int4_Bvector[27:24];

//第8个4b
assign exp_E6[1] = is_mxnvfp4 ? {2'd0, AFP4_int[31:28]} : {2'd0, int4_Avector[31:28]};
assign exp_E6[3] = is_mxnvfp4 ? {2'd0, BFP4_int[31:28]} : {2'd0, int4_Bvector[31:28]};

//第1个和第2个4b
// assign a_8[0] = is_Int4 ? {int4_Avector[7:4], int4_Avector[3:0]} : {AFP4_int[7:4], AFP4_int[3:0]};
// assign b_8[0] = is_Int4 ? {int4_Bvector[7:4], int4_Bvector[3:0]} : {BFP4_int[7:4], BFP4_int[3:0]};

// //第3个4b
// assign a_4[0] = is_Int4 ? int4_Avector[11:8] : AFP4_int[11:8];
// assign b_4[0] = is_Int4 ? int4_Bvector[11:8] : BFP4_int[11:8];

// //第4个4b
// assign exp_E6[0] = is_Int4 ? {2'd0, int4_Avector[15:12]} : {2'd0, AFP4_int[15:12]};
// assign exp_E6[2] = is_Int4 ? {2'd0, int4_Bvector[15:12]} : {2'd0, BFP4_int[15:12]};

// //第5个和第6个4b
// assign a_8[1] = is_Int4 ? {int4_Avector[23:20], int4_Avector[19:16]} : {AFP4_int[23:20], AFP4_int[19:16]};
// assign b_8[1] = is_Int4 ? {int4_Bvector[23:20], int4_Bvector[19:16]} : {BFP4_int[23:20], BFP4_int[19:16]};

// //第7个4b
// assign a_4[1] = is_Int4 ? int4_Avector[27:24] : AFP4_int[27:24];
// assign b_4[1] = is_Int4 ? int4_Bvector[27:24] : BFP4_int[27:24];

// //第8个4b
// assign exp_E6[1] = is_Int4 ? {2'd0, int4_Avector[31:28]} : {2'd0, AFP4_int[31:28]};
// assign exp_E6[3] = is_Int4 ? {2'd0, int4_Bvector[31:28]} : {2'd0, BFP4_int[31:28]};


endmodule
