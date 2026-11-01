// 这个模块负责将输入的32bit的数据解码成对应的指数和尾数
// 并将尾数连接到对应的等线网中(注意这里的尾数只存在三种情况)
// 而指数部分，则输出两个E8，两个E5
// 符号位部分，则输出4bit的符号位向量

`include "defines.sv"

module decode_Bdata(
    input  wire [31 : 0 ] Vector_bits ,

    //from opcode_decode
    input wire is_TF32, is_BF16, is_FP16, 
    input wire is_E5M2, is_E3M2, is_E4M3, 
    input wire is_E2  , is_Int8, is_E8  ,
    input wire is_Int4,
    input wire is_mxint4,
    input wire is_int,
    input wire is_fp16_bf16,
    input wire is_intfp4 , 
    input wire BS_have   ,

    //尾数分配
    output wire [7 :0] b_8[0:1],
    output wire [3 :0] b_4[0:1],

    //输出是否为0
    output wire [3 : 0] Bzero ,

    //符号位向量组(这里解码的话，把fp4也带上)
    output wire [7:0] vec8_sign,

    //判断是否存在每组mxfp8/fp6是否内部数据全0
    output wire [3:0] nz_E5_FP8, nz_E3_FP6,

    //指数组
    output wire signed [4:0] exp_E5 [0 : 3] 
);
//先解码出来
//是否存在先导1
wire [3:0] nz_E5, nz_E4, nz_E3, nz_E2;
wire [1:0] nz_FP16, nz_BF16;
wire       nz_TF32;

assign nz_TF32    = |Vector_bits[30:23];

assign nz_FP16[0] = |Vector_bits[14:10];
assign nz_FP16[1] = |Vector_bits[30:26];

assign nz_BF16[0] = |Vector_bits[14:7];
assign nz_BF16[1] = nz_TF32;

assign nz_E5[0] = |Vector_bits[6:2];
assign nz_E5[1] = |Vector_bits[14:10];
assign nz_E5[2] = |Vector_bits[22:18];
assign nz_E5[3] = |Vector_bits[30:26];

assign nz_E4[0] = |Vector_bits[6:3];
assign nz_E4[1] = |Vector_bits[14:11];
assign nz_E4[2] = |Vector_bits[22:19];
assign nz_E4[3] = |Vector_bits[30:27];

assign nz_E3[0] = |Vector_bits[4:2];
assign nz_E3[1] = |Vector_bits[12:10];
assign nz_E3[2] = |Vector_bits[20:18];
assign nz_E3[3] = |Vector_bits[28:26];

assign nz_E2[0] = |Vector_bits[4:3];
assign nz_E2[1] = |Vector_bits[12:11];
assign nz_E2[2] = |Vector_bits[20:19];
assign nz_E2[3] = |Vector_bits[28:27];

//判断对于该组的mxfp8/fp6是否全为0的高位
assign nz_E5_FP8 = nz_E5;
assign nz_E3_FP6 = nz_E3;

//tf32
wire [7:0] TF32_exp  = nz_TF32 ? Vector_bits[30:23] : 8'd1;
wire [9:0] TF32_sig  = Vector_bits[22:13];
//bf16
wire [7:0] BF16_exp_0  = nz_BF16[0] ? Vector_bits[14:7] : 8'd1;
wire [6:0] BF16_sig_0  = Vector_bits[6:0];

// wire [7:0] BF16_exp_1  = nz_BF16[1] ? Vector_bits[30:23] : 8'd1; 
wire [6:0] BF16_sig_1  = Vector_bits[22:16];

//FP16
wire [4:0] FP16_exp_0  = nz_FP16[0] ? Vector_bits[14:10] : 5'd1;
wire [9:0] FP16_sig_0  = Vector_bits[9:0];

wire [4:0] FP16_exp_1  = nz_FP16[1] ? Vector_bits[30:26] : 5'd1;
wire [9:0] FP16_sig_1  = Vector_bits[25:16] ;

//E5
wire [4:0] E5M2_exp_0  = nz_E5[0] ? Vector_bits[6:2] : 5'd1;
wire [1:0] E5M2_sig_0  = Vector_bits[1:0];

wire [4:0] E5M2_exp_1  = nz_E5[1] ? Vector_bits[14:10] : 5'd1;
wire [1:0] E5M2_sig_1  = Vector_bits[9:8];

wire [4:0] E5M2_exp_2  = nz_E5[2] ? Vector_bits[22:18] : 5'd1;
wire [1:0] E5M2_sig_2  = Vector_bits[17:16] ;

wire [4:0] E5M2_exp_3  = nz_E5[3] ? Vector_bits[30:26] : 5'd1;
wire [1:0] E5M2_sig_3  = Vector_bits[25:24] ;  // 2-bit

//e4m3 (4个8-bit值)
wire [3:0] E4M3_exp_0  = nz_E4[0] ? Vector_bits[6:3] : 4'd1;
wire [2:0] E4M3_sig_0  = Vector_bits[2:0];

wire [3:0] E4M3_exp_1  = nz_E4[1] ? Vector_bits[14:11] : 4'd1;
wire [2:0] E4M3_sig_1  = Vector_bits[10:8];

wire [3:0] E4M3_exp_2  = nz_E4[2] ? Vector_bits[22:19] : 4'd1;
wire [2:0] E4M3_sig_2  = Vector_bits[18:16] ;

wire [3:0] E4M3_exp_3  = nz_E4[3] ? Vector_bits[30:27] : 4'd1;
wire [2:0] E4M3_sig_3  = Vector_bits[26:24] ;

//e3m2 (4个8-bit值)
wire [2:0] E3M2_exp_0  = nz_E3[0] ? Vector_bits[4:2] : 3'd1;

wire [2:0] E3M2_exp_1  = nz_E3[1] ? Vector_bits[12:10] : 3'd1;

wire [2:0] E3M2_exp_2  = nz_E3[2] ? Vector_bits[20:18] : 3'd1;

wire [2:0] E3M2_exp_3  = nz_E3[3] ? Vector_bits[28:26] : 3'd1;

//e2 (4个8-bit值)
wire [1:0] E2_exp_0  = nz_E2[0] ? Vector_bits[4:3] : 2'd1;

wire [1:0] E2_exp_1  = nz_E2[1] ? Vector_bits[12:11] : 2'd1;

wire [1:0] E2_exp_2  = nz_E2[2] ? Vector_bits[20:19] : 2'd1;

wire [1:0] E2_exp_3  = nz_E2[3] ? Vector_bits[28:27] : 2'd1;

//int8 (有符号整数)
// wire signed [7:0] INT8_3 = Vector_bits[31:24];
// wire signed [7:0] INT8_2 = Vector_bits[23:16];
// wire signed [7:0] INT8_1 = Vector_bits[15:8] ;
// wire signed [7:0] INT8_0 = Vector_bits[7:0]  ; 


/*-----------------检测是否存在0-----------------*/
assign Bzero[3] = (is_E5M2 || is_E4M3) ? !(|Vector_bits[30:24]) : !(|Vector_bits[28:24]);//E3M2 or E2M3
assign Bzero[2] = (is_E5M2 || is_E4M3) ? !(|Vector_bits[22:16]) : !(|Vector_bits[20:16]);//E3M2 or E2M3

assign Bzero[1] = (is_E5M2 || is_E4M3) ? !(|Vector_bits[14:8]) : 
                  (is_E3M2 || is_E2  ) ? !(|Vector_bits[12:8]):
                  (is_TF32           ) ? !(|Vector_bits[30:13]):
                                         !(|Vector_bits[30:16]);//BF16 or FP16

assign Bzero[0] = (is_E5M2 || is_E4M3) ? !(|Vector_bits[6 :0]) : 
                  (is_E3M2 || is_E2  ) ? !(|Vector_bits[4 :0]) :
                  (is_TF32           ) ? 1'd1                  :
                                         !(|Vector_bits[14:0]);//BF16 or FP16

/*-----------------分配符号位向量-----------------*/
assign vec8_sign[7] = ((is_Int4 || is_mxint4) && !BS_have) ? 1'b0 : Vector_bits[31];
assign vec8_sign[6] = ((is_Int4 || is_mxint4) && !BS_have) ? 1'b0 : Vector_bits[27];
assign vec8_sign[5] = ((is_Int4 || is_mxint4) && !BS_have) ? 1'b0 : Vector_bits[23];
assign vec8_sign[4] = ((is_Int4 || is_mxint4) && !BS_have) ? 1'b0 : Vector_bits[19];

assign vec8_sign[3] = (is_int && !BS_have)  ? 1'd0            :
                      is_intfp4             ? Vector_bits[15] :
                      (is_E2   ||  is_E3M2) ? Vector_bits[29] :
                                              Vector_bits[31] ;

assign vec8_sign[2] = (is_int && !BS_have)  ? 1'd0            :
                      is_intfp4             ? Vector_bits[11] :
                      (is_E2   ||  is_E3M2) ? Vector_bits[21] :
                                              Vector_bits[23] ;

//因为如果是TF32，则我们使用序列为[1]进行存储和计算，另一个尾数填充0，指数按照最小值来看就可以了
assign vec8_sign[1] = (is_fp16_bf16||is_TF32) ? Vector_bits[31] :
                      (is_int && !BS_have)    ? 1'd0            :
                      is_intfp4               ? Vector_bits[7 ] :
                      (is_E2   ||  is_E3M2  ) ? Vector_bits[13] :
                                                Vector_bits[15] ;

assign vec8_sign[0] = is_fp16_bf16          ? Vector_bits[15] :
                      (is_int && !BS_have)  ? 1'd0            :
                      is_intfp4             ? Vector_bits[3 ] :
                      (is_E2   ||  is_E3M2) ? Vector_bits[5 ] :
                                              Vector_bits[7 ] ;

/*-----------------分配指数位向量-----------------*/
//填充E5，对于E8而言，把其低三位放在这上面,其中对于FP16而言，其虽然是E5，但是我们将其看成{3'd0,E5}
assign exp_E5[0] = (is_BF16) ? {2'd0,BF16_exp_0[2:0]}     :
                   (is_FP16) ? {2'd0,FP16_exp_0[2:0]}     :
                   (is_E5M2) ? E5M2_exp_0                 :
                   (is_E4M3) ? {1'd0, E4M3_exp_0}         :
                   (is_E3M2) ? {2'd0, E3M2_exp_0}         :
                   (is_E2  ) ? {3'd0, E2_exp_0  }         :
                               5'd0                       ;//TF32//INT8

assign exp_E5[1] = (is_E8  ) ? {2'd0,TF32_exp[2:0]}       :
                   (is_FP16) ? {2'd0,FP16_exp_1[2:0]}     :
                   (is_Int8) ? 5'd0                       :
                   (is_E5M2) ? E5M2_exp_1                 :
                   (is_E4M3) ? {1'd0, E4M3_exp_1}         :
                   (is_E3M2) ? {2'd0, E3M2_exp_1}         :
                               {3'd0, E2_exp_1  }         ;

//填充E5,或者E8的高位高5位放在这上面
assign exp_E5[2] = (is_E5M2) ? E5M2_exp_2                 :
                   (is_E4M3) ? {1'd0, E4M3_exp_2}         :
                   (is_E3M2) ? {2'd0, E3M2_exp_2}         :
                   (is_E2  ) ? {3'd0, E2_exp_2  }         :
                   (is_FP16) ? {3'd0, FP16_exp_0[4:3]}    :
                   (is_TF32 || is_Int8) ? 5'd0            :
                               BF16_exp_0[7:3]            ;//BF16

assign exp_E5[3] = (is_Int8) ? 5'd0                       :
                   (is_E5M2) ? E5M2_exp_3                 :
                   (is_E4M3) ? {{1'd0}, E4M3_exp_3}       :
                   (is_E3M2) ? {{2'd0}, E3M2_exp_3}       :
                   (is_FP16) ? {3'd0, FP16_exp_1[4:3]}    :
                   (is_E2  ) ? {3'd0, E2_exp_3    }       :
                               TF32_exp[7:3]              ;//TF32/BF16

//填充尾数到乘法器中(这里不管INT8与NVE2M1,这个在其他模块做,然后进行多路选择就可以了
//填充b_8
assign b_8[0] = (is_BF16) ? {1'd0, nz_BF16[0], BF16_sig_0[6:1]} :
                (is_TF32) ? 8'd0 :
                (is_FP16) ? {1'd0, nz_FP16[0], FP16_sig_0[9:4]} :
                (is_E5M2) ? {4'd0, nz_E5[0] , E5M2_sig_0, 1'd0} :
                (is_E3M2) ? {4'd0, nz_E3[0] , E5M2_sig_0, 1'd0} :
                (is_E4M3) ? {4'd0, nz_E4[0] , E4M3_sig_0      } :
                            {4'd0, nz_E2[0] , E4M3_sig_0      } ;

assign b_8[1] = (is_BF16) ? {1'd0, nz_BF16[1], BF16_sig_1[6:1]} :
                (is_FP16) ? {1'd0, nz_FP16[1], FP16_sig_1[9:4]} :
                (is_TF32) ? {1'd0, nz_TF32   , TF32_sig[9:4]  } :
                (is_E5M2) ? {4'd0, nz_E5[2] , E5M2_sig_2, 1'd0} :
                (is_E3M2) ? {4'd0, nz_E3[2] , E5M2_sig_2, 1'd0} :
                (is_E4M3) ? {4'd0, nz_E4[2] , E4M3_sig_2      } :
                            {4'd0, nz_E2[2] , E4M3_sig_2      } ;

assign b_4[0] = (is_FP16) ? FP16_sig_0[3:0]      : 
                (is_BF16) ? {BF16_sig_0[0],3'd0} : 
                (is_TF32) ? 4'd0  :
                (is_E5M2) ? {nz_E5[1], E5M2_sig_1,1'd0}:
                (is_E3M2) ? {nz_E3[1], E5M2_sig_1,1'd0}:
                (is_E4M3) ? {nz_E4[1], E4M3_sig_1}:
                            {nz_E2[1], E4M3_sig_1};

assign b_4[1] = (is_FP16) ? FP16_sig_1[3:0]      : 
                (is_BF16) ? {BF16_sig_1[0],3'd0} :
                (is_TF32) ? TF32_sig[3:0]        :
                (is_E5M2) ? {nz_E5[3], E5M2_sig_3,1'd0}:
                (is_E3M2) ? {nz_E3[3], E5M2_sig_3,1'd0}:
                (is_E4M3) ? {nz_E4[3], E4M3_sig_3}:
                            {nz_E2[3], E4M3_sig_3};


// //填充尾数到乘法器中(这里不管INT8与NVE2M1,这个在其他模块做,然后进行多路选择就可以了)
// //填充b_88
// assign b_88[0] = (is_BF16) ? {1'd0, nz_BF16[0], BF16_sig_0[6:1]} :
//                  (is_FP16) ? {1'd0, nz_FP16[0], FP16_sig_0[9:4]} :
//                  (is_TF32) ? 8'd0                                :
//                  (is_E5M2) ? {4'd0, nz_E5[1], E5M2_sig_1, 1'd0}:
//                  (is_E3M2) ? {4'd0, nz_E3[1], E5M2_sig_1, 1'd0}:
//                  (is_E4M3) ? {4'd0, nz_E4[1], E4M3_sig_1}:
//                              {4'd0, nz_E2[1], E4M3_sig_1};

// assign b_88[1] = (is_BF16) ? {1'd0, nz_BF16[1], BF16_sig_1[6:1]} :
//                  (is_FP16) ? {1'd0, nz_FP16[1], FP16_sig_1[9:4]} :
//                  (is_TF32) ? {1'd0, nz_TF32   , TF32_sig[9:4]  } :
//                  (is_E5M2) ? {4'd0, nz_E5[3], E5M2_sig_3, 1'd0}:
//                  (is_E3M2) ? {4'd0, nz_E3[3], E5M2_sig_3, 1'd0}:
//                  (is_E4M3) ? {4'd0, nz_E4[3], E4M3_sig_3}:
//                              {4'd0, nz_E2[3], E4M3_sig_3};
// //填充b_48
// assign b_48[0] = (is_BF16) ? {1'd0, nz_BF16[0], BF16_sig_0[6:1]} :
//                  (is_FP16) ? {1'd0, nz_FP16[0], FP16_sig_0[9:4]} :
//                  (is_TF32) ? 8'd0 :
//                  (is_E5M2) ? {4'd0, nz_E5[0], E5M2_sig_0, 1'd0}:
//                  (is_E3M2) ? {4'd0, nz_E3[0], E5M2_sig_0, 1'd0}:
//                  (is_E4M3) ? {4'd0, nz_E4[0], E4M3_sig_0      }:
//                              {4'd0, nz_E2[0], E4M3_sig_0      };

// assign b_48[1] = (is_TF32) ? {1'd0, nz_TF32   , TF32_sig[9:4]  }  :
//                  (is_BF16) ? {1'd0, nz_BF16[1], BF16_sig_1[6:1]} :
//                  (is_FP16) ? {1'd0, nz_FP16[1], FP16_sig_1[9:4]} :
//                  (is_E5M2) ? {4'd0, nz_E5[2], E5M2_sig_2, 1'd0}:
//                  (is_E3M2) ? {4'd0, nz_E3[2], E5M2_sig_2, 1'd0}:
//                  (is_E4M3) ? {4'd0, nz_E4[2], E4M3_sig_2      }:
//                              {4'd0, nz_E2[2], E4M3_sig_2      };
// //填充b_84
// assign b_84[0] = (is_FP16) ? FP16_sig_0[3:0]             : 
//                  (is_BF16) ? {BF16_sig_0[0], 3'd0}       : 4'd0 ;

// assign b_84[1] = (is_FP16) ? FP16_sig_1[3:0]             : 
//                  (is_BF16) ? {BF16_sig_1[0],3'd0}        :
//                  (is_TF32) ? TF32_sig[3:0]               : 4'd0;

// //填充b_44
// assign b_44[0] = (is_FP16) ? FP16_sig_0[3:0]:
//                  (is_BF16) ? {BF16_sig_0[0],3'd0}: 4'd0;

// assign b_44[1] = (is_TF32) ? TF32_sig[3:0] :
//                  (is_FP16) ? FP16_sig_1[3:0]:
//                  (is_BF16) ? {BF16_sig_1, 3'd0}: 4'd0;

endmodule
