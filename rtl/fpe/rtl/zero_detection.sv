module zero_detection (
    input  wire [31:0] A_Vector, B_Vector,
    input  wire is_mxint4, is_mxfp8,

    input wire [3:0] nz_E5_AFP8, nz_E3_AFP6,
    input wire [3:0] nz_E5_BFP8, nz_E3_BFP6,
    
    output wire MXINTFP4_all_zero,
    output wire MXFP86_all_zero
);

    // wire AFP4_all_zero_temp, BFP4_all_zero_temp;

    // wire AFP4_all_zero, BFP4_all_zero, AMXINT4_all_zero, BMXINT4_all_zero;

    // wire [3:0] nz_low_Afp8_6, nz_low_Bfp8_6;

    // assign nz_low_Afp8_6[0] = |A_Vector[1 :0];
    // assign nz_low_Afp8_6[1] = |A_Vector[9 :8];
    // assign nz_low_Afp8_6[2] = |A_Vector[17:16];
    // assign nz_low_Afp8_6[3] = |A_Vector[25:24];

    // assign nz_low_Bfp8_6[0] = |B_Vector[1 :0];
    // assign nz_low_Bfp8_6[1] = |B_Vector[9 :8];
    // assign nz_low_Bfp8_6[2] = |B_Vector[17:16];
    // assign nz_low_Bfp8_6[3] = |B_Vector[25:24];

    // //判断mxfp8/fp6是否全0
    // wire AFP8_all_zero, BFP8_all_zero, AFP6_all_zero, BFP6_all_zero;
    
    // assign AFP8_all_zero = ~|{nz_E5_AFP8, nz_low_Afp8_6};
    // assign BFP8_all_zero = ~|{nz_E5_BFP8, nz_low_Bfp8_6};
    // assign AFP6_all_zero = ~|{nz_E3_AFP6, nz_low_Afp8_6};
    // assign BFP6_all_zero = ~|{nz_E3_BFP6, nz_low_Bfp8_6};

    // assign MXFP86_all_zero = is_mxfp8 ? (AFP8_all_zero || BFP8_all_zero) :
    //                                     (AFP6_all_zero || BFP6_all_zero) ;


    // //判断mxintfp4是否全0
    // assign AFP4_all_zero_temp = |{
    //     A_Vector[30:28],
    //     A_Vector[26],
    //     A_Vector[22:20],
    //     A_Vector[18],
    //     A_Vector[14:12],
    //     A_Vector[10],
    //     A_Vector[6:4],
    //     A_Vector[2]
    // };

    // assign AFP4_all_zero = ~|{AFP4_all_zero_temp, nz_low_Afp8_6};

    // assign BFP4_all_zero_temp = |{
    //     B_Vector[30:28],
    //     B_Vector[26],
    //     B_Vector[22:20],
    //     B_Vector[18],
    //     B_Vector[14:12],
    //     B_Vector[10],
    //     B_Vector[6:4],
    //     B_Vector[2]
    // };

    // assign BFP4_all_zero = ~|{BFP4_all_zero_temp, nz_low_Bfp8_6};

    // assign AMXINT4_all_zero = ~|{
    //     A_Vector[31],
    //     A_Vector[27],
    //     A_Vector[23],
    //     A_Vector[19],
    //     A_Vector[15],
    //     A_Vector[11],
    //     A_Vector[7],
    //     A_Vector[3]
    // };

    // assign BMXINT4_all_zero = ~|{
    //     B_Vector[31],
    //     B_Vector[27],
    //     B_Vector[23],
    //     B_Vector[19],
    //     B_Vector[15],
    //     B_Vector[11],
    //     B_Vector[7],
    //     B_Vector[3]
    // };

    // assign MXINTFP4_all_zero = is_mxint4 ? ((AFP4_all_zero && AMXINT4_all_zero) || (BFP4_all_zero && BMXINT4_all_zero)) :
    //                                        AFP4_all_zero || BFP4_all_zero                                               ;
    
    //对于mxfp8、mxfp6进行0的检测，需要注意的是其检测的是a*b为0，不是单独的一个为0
    wire [3:0] nz_low_Afp8_6, nz_low_Bfp8_6;

    assign nz_low_Afp8_6[0] = |A_Vector[1 :0];
    assign nz_low_Afp8_6[1] = |A_Vector[9 :8];
    assign nz_low_Afp8_6[2] = |A_Vector[17:16];
    assign nz_low_Afp8_6[3] = |A_Vector[25:24];

    assign nz_low_Bfp8_6[0] = |B_Vector[1 :0];
    assign nz_low_Bfp8_6[1] = |B_Vector[9 :8];
    assign nz_low_Bfp8_6[2] = |B_Vector[17:16];
    assign nz_low_Bfp8_6[3] = |B_Vector[25:24];

    //判断mxfp8/fp6是否全0
    wire [3:0] AFP8_all_nz, BFP8_all_nz, AFP6_all_nz, BFP6_all_nz;

    assign AFP8_all_nz = nz_E5_AFP8 | nz_low_Afp8_6;
    assign BFP8_all_nz = nz_E5_BFP8 | nz_low_Bfp8_6;
    assign AFP6_all_nz = nz_E3_AFP6 | nz_low_Afp8_6;
    assign BFP6_all_nz = nz_E3_BFP6 | nz_low_Bfp8_6;

    wire FP8_all_zero, FP6_all_zero;
    assign FP8_all_zero = ~|(AFP8_all_nz & BFP8_all_nz);
    assign FP6_all_zero = ~|(AFP6_all_nz & BFP6_all_nz);

    assign MXFP86_all_zero = is_mxfp8 ? FP8_all_zero : FP6_all_zero;

    //对于mxintfp4是否全0
    wire [7:0] AFP4_all_nz, BFP4_all_nz, AMXINT4_all_nz, BMXINT4_all_nz;

    //检测各个FP4是否为0
    assign AFP4_all_nz[0] =  A_Vector[2]  || nz_low_Afp8_6[0];
    assign AFP4_all_nz[1] = |A_Vector[6:4];
    assign AFP4_all_nz[2] =  A_Vector[10] || nz_low_Afp8_6[1];
    assign AFP4_all_nz[3] = |A_Vector[14:12];
    assign AFP4_all_nz[4] =  A_Vector[18] || nz_low_Afp8_6[2];
    assign AFP4_all_nz[5] = |A_Vector[22:20];
    assign AFP4_all_nz[6] =  A_Vector[26] || nz_low_Afp8_6[3];
    assign AFP4_all_nz[7] = |A_Vector[30:28];

    assign BFP4_all_nz[0] =  B_Vector[2]  || nz_low_Bfp8_6[0];
    assign BFP4_all_nz[1] = |B_Vector[6:4];
    assign BFP4_all_nz[2] =  B_Vector[10] || nz_low_Bfp8_6[1];
    assign BFP4_all_nz[3] = |B_Vector[14:12];
    assign BFP4_all_nz[4] =  B_Vector[18] || nz_low_Bfp8_6[2];
    assign BFP4_all_nz[5] = |B_Vector[22:20];
    assign BFP4_all_nz[6] =  B_Vector[26] || nz_low_Bfp8_6[3];
    assign BFP4_all_nz[7] = |B_Vector[30:28];

    //检测各个mxint4是否为0
    assign AMXINT4_all_nz[0] = A_Vector[3]  || AFP4_all_nz[0];
    assign AMXINT4_all_nz[1] = A_Vector[7]  || AFP4_all_nz[1];
    assign AMXINT4_all_nz[2] = A_Vector[11] || AFP4_all_nz[2];
    assign AMXINT4_all_nz[3] = A_Vector[15] || AFP4_all_nz[3];
    assign AMXINT4_all_nz[4] = A_Vector[19] || AFP4_all_nz[4];
    assign AMXINT4_all_nz[5] = A_Vector[23] || AFP4_all_nz[5];
    assign AMXINT4_all_nz[6] = A_Vector[27] || AFP4_all_nz[6];
    assign AMXINT4_all_nz[7] = A_Vector[31] || AFP4_all_nz[7];

    assign BMXINT4_all_nz[0] = B_Vector[3]  || BFP4_all_nz[0];
    assign BMXINT4_all_nz[1] = B_Vector[7]  || BFP4_all_nz[1];
    assign BMXINT4_all_nz[2] = B_Vector[11] || BFP4_all_nz[2];
    assign BMXINT4_all_nz[3] = B_Vector[15] || BFP4_all_nz[3];
    assign BMXINT4_all_nz[4] = B_Vector[19] || BFP4_all_nz[4];
    assign BMXINT4_all_nz[5] = B_Vector[23] || BFP4_all_nz[5];
    assign BMXINT4_all_nz[6] = B_Vector[27] || BFP4_all_nz[6];
    assign BMXINT4_all_nz[7] = B_Vector[31] || BFP4_all_nz[7];

    //输出a*b是否为全0
    wire FP4_all_zero, MXINT4_all_zero;
    assign FP4_all_zero     = ~|(AFP4_all_nz    & BFP4_all_nz   );
    assign MXINT4_all_zero  = ~|(AMXINT4_all_nz & BMXINT4_all_nz);

    assign MXINTFP4_all_zero = is_mxint4 ? MXINT4_all_zero : FP4_all_zero;
endmodule
