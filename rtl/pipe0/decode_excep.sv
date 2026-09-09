`include "defines.sv"

//这里我们检测四种异常情况,分别为正负无穷,NAN,0
//检测完这四种异常之后,我们再将四种放在一起处理输出一组情况
module decode_excep(
    input  wire [31 : 0] Vector_bits ,
    input  wire is_TF32, is_BF16, 
    input  wire is_FP16, is_E5M2, is_E4M3,
    output wire [3:0] NAN, INF, ZERO
);

//tf32
wire [7:0] TF32_exp  = Vector_bits[30:23];
wire [9:0] TF32_sig  = Vector_bits[22:13];
//bf16
wire [7:0] BF16_exp_0  = Vector_bits[14:7];
wire [6:0] BF16_sig_0  = Vector_bits[6:0];

wire [7:0] BF16_exp_1  = Vector_bits[30:23]; 
wire [6:0] BF16_sig_1  = Vector_bits[22:16];

//FP16
wire [4:0] FP16_exp_0  = Vector_bits[14:10];
wire [9:0] FP16_sig_0  = Vector_bits[9:0];

wire [4:0] FP16_exp_1  = Vector_bits[30:26];
wire [9:0] FP16_sig_1  = Vector_bits[25:16] ;

//E5
wire [4:0] E5M2_exp_0  = Vector_bits[6:2];
wire [1:0] E5M2_sig_0  = Vector_bits[1:0];

wire [4:0] E5M2_exp_1  = Vector_bits[14:10];
wire [1:0] E5M2_sig_1  = Vector_bits[9:8];

wire [4:0] E5M2_exp_2  = Vector_bits[22:18];
wire [1:0] E5M2_sig_2  = Vector_bits[17:16] ;

wire [4:0] E5M2_exp_3  = Vector_bits[30:26];
wire [1:0] E5M2_sig_3  = Vector_bits[25:24] ;  // 2-bit

//e4m3 (4个8-bit值)
wire [3:0] E4M3_exp_0  = Vector_bits[6:3];
wire [2:0] E4M3_sig_0  = Vector_bits[2:0];

wire [3:0] E4M3_exp_1  = Vector_bits[14:11];
wire [2:0] E4M3_sig_1  = Vector_bits[10:8];

wire [3:0] E4M3_exp_2  = Vector_bits[22:19];
wire [2:0] E4M3_sig_2  = Vector_bits[18:16] ;

wire [3:0] E4M3_exp_3  = Vector_bits[30:27];
wire [2:0] E4M3_sig_3  = Vector_bits[26:24] ;

//TF32
wire TF32_NAN  = &TF32_exp && |TF32_sig;
wire TF32_INF  = &TF32_exp && !(|TF32_sig);
wire TF32_ZERO = !(|TF32_exp || |TF32_sig);

//BF16
wire BF16_NAN_0  = &BF16_exp_0 && |BF16_sig_0;
wire BF16_INF_0  = &BF16_exp_0 && !(|BF16_sig_0);
wire BF16_ZERO_0 = !(|BF16_exp_0 || |BF16_sig_0);

wire BF16_NAN_1  = &BF16_exp_1 && |BF16_sig_1;
wire BF16_INF_1  = &BF16_exp_1 && !(|BF16_sig_1);
wire BF16_ZERO_1 = !(|BF16_exp_1 || |BF16_sig_1);

//FP16
wire FP16_NAN_0  = &FP16_exp_0 && |FP16_sig_0;
wire FP16_INF_0  = &FP16_exp_0 && !(|FP16_sig_0);
wire FP16_ZERO_0 = !(|FP16_exp_0 || |FP16_sig_0);

wire FP16_NAN_1  = &FP16_exp_1 && |FP16_sig_1;
wire FP16_INF_1  = &FP16_exp_1 && !(|FP16_sig_1);
wire FP16_ZERO_1 = !(|FP16_exp_1 || |FP16_sig_1);

//E5M2
wire E5M2_NAN_0  = &E5M2_exp_0 && |E5M2_sig_0;
wire E5M2_INF_0  = &E5M2_exp_0 && !(|E5M2_sig_0);
wire E5M2_ZERO_0 = !(|E5M2_exp_0 || |E5M2_sig_0);

wire E5M2_NAN_1  = &E5M2_exp_1 && |E5M2_sig_1;
wire E5M2_INF_1  = &E5M2_exp_1 && !(|E5M2_sig_1);
wire E5M2_ZERO_1 = !(|E5M2_exp_1 || |E5M2_sig_1);

wire E5M2_NAN_2  = &E5M2_exp_2 && |E5M2_sig_2;
wire E5M2_INF_2  = &E5M2_exp_2 && !(|E5M2_sig_2);
wire E5M2_ZERO_2 = !(|E5M2_exp_2 || |E5M2_sig_2);

wire E5M2_NAN_3  = &E5M2_exp_3 && |E5M2_sig_3;
wire E5M2_INF_3  = &E5M2_exp_3 && !(|E5M2_sig_3);
wire E5M2_ZERO_3 = !(|E5M2_exp_3 || |E5M2_sig_3);

//E4M3
wire E4M3_NAN_0  = &E4M3_exp_0 && &E4M3_sig_0;
wire E4M3_NAN_1  = &E4M3_exp_1 && &E4M3_sig_1;
wire E4M3_NAN_2  = &E4M3_exp_2 && &E4M3_sig_2;
wire E4M3_NAN_3  = &E4M3_exp_3 && &E4M3_sig_3;

//输出异常
//NAN
assign NAN[0] = (is_BF16) ? BF16_NAN_0 :
                (is_FP16) ? FP16_NAN_0 :
                (is_E5M2) ? E5M2_NAN_0 :
                (is_E4M3) ? E4M3_NAN_0 :
                            1'd0;
                        
assign NAN[1] = (is_TF32) ? TF32_NAN   :
                (is_BF16) ? BF16_NAN_1 :
                (is_FP16) ? FP16_NAN_1 :
                (is_E5M2) ? E5M2_NAN_1 :
                (is_E4M3) ? E4M3_NAN_1 :
                            1'd0;

assign NAN[2] = (is_E5M2) ? E5M2_NAN_2 :
                (is_E4M3) ? E4M3_NAN_2 :
                            1'd0;

assign NAN[3] = (is_E5M2) ? E5M2_NAN_3 :
                (is_E4M3) ? E4M3_NAN_3 :
                            1'd0;          

//这里只看无穷，后续的符号依据相乘的符号位
assign INF[0] = (is_BF16) ? (BF16_INF_0) :
                (is_FP16) ? (FP16_INF_0) :
                (is_E5M2) ? (E5M2_INF_0) :
                           1'd0;

assign INF[1] = (is_TF32) ? (TF32_INF  ) :
                (is_BF16) ? (BF16_INF_1) :
                (is_FP16) ? (FP16_INF_1) :
                (is_E5M2) ? (E5M2_INF_1) :
                            1'd0;

assign INF[2] = (is_E5M2) ? (E5M2_INF_2) : 1'd0;

assign INF[3] = (is_E5M2) ? (E5M2_INF_3) : 1'd0;

//ZERO 零
assign ZERO[0] = (is_BF16) ? BF16_ZERO_0 :
                 (is_FP16) ? FP16_ZERO_0 :
                 (is_E5M2) ? E5M2_ZERO_0 :
                             1'd0;

assign ZERO[1] = (is_TF32) ? TF32_ZERO :
                 (is_BF16) ? BF16_ZERO_1 :
                 (is_FP16) ? FP16_ZERO_1 :
                 (is_E5M2) ? E5M2_ZERO_1 :
                             1'd0;

assign ZERO[2] = (is_E5M2) ? E5M2_ZERO_2 : 1'd0;

assign ZERO[3] = (is_E5M2) ? E5M2_ZERO_3 : 1'd0;

endmodule
