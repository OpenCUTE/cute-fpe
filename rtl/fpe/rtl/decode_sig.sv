module decode_sig (
    input wire  is_High_precision,
    input wire  is_low_precision,
    input wire  is_INTFP4,

    input wire  [7:0] a_8 [0:1],
    input wire  [3:0] a_4 [0:1],
    input wire  [3:0] b_4 [0:1],
    input wire  [7:0] b_8 [0:1],
    input wire  [5:0] exp_E6 [0:3],

    output wire [7 :0] a_84[0:1], a_88[0:1],
    output wire [3 :0] a_48[0:1], a_44[0:1],
    output wire [3 :0] b_84[0:1], b_44[0:1],
    output wire [7 :0] b_48[0:1], b_88[0:1]
);

//高精度尾数寄存器
wire [7 :0] high_a_84[0:1], high_a_88[0:1];
wire [3 :0] high_a_48[0:1], high_a_44[0:1];
wire [3 :0] high_b_84[0:1], high_b_44[0:1];
wire [7 :0] high_b_48[0:1], high_b_88[0:1];

//低精度尾数寄存器
wire [7 :0] low_a_84[0:1], low_a_88[0:1];
wire [3 :0] low_a_48[0:1], low_a_44[0:1];
wire [3 :0] low_b_84[0:1], low_b_44[0:1];
wire [7 :0] low_b_48[0:1], low_b_88[0:1];

//INTFP4尾数寄存器
wire [7 :0] intfp4_a_84[0:1], intfp4_a_88[0:1];
wire [3 :0] intfp4_a_48[0:1], intfp4_a_44[0:1];
wire [3 :0] intfp4_b_84[0:1], intfp4_b_44[0:1];
wire [7 :0] intfp4_b_48[0:1], intfp4_b_88[0:1];

//INT8尾数寄存器
wire [7 :0] int8_a_84[0:1], int8_a_88[0:1];
wire [3 :0] int8_a_48[0:1], int8_a_44[0:1];
wire [3 :0] int8_b_84[0:1], int8_b_44[0:1];
wire [7 :0] int8_b_48[0:1], int8_b_88[0:1];

//现在分配高精度寄存器
assign high_a_84[0] = a_8[0];
assign high_a_84[1] = a_8[1];
assign high_a_88[0] = a_8[0];
assign high_a_88[1] = a_8[1];

assign high_a_48[0] = a_4[0];
assign high_a_48[1] = a_4[1];
assign high_a_44[0] = a_4[0];
assign high_a_44[1] = a_4[1];

assign high_b_84[0] = b_4[0];
assign high_b_84[1] = b_4[1];
assign high_b_44[0] = b_4[0];
assign high_b_44[1] = b_4[1];

assign high_b_48[0] = b_8[0];
assign high_b_48[1] = b_8[1];
assign high_b_88[0] = b_8[0];
assign high_b_88[1] = b_8[1];

//现在分配低精度寄存器
assign low_a_84[0] = '0;
assign low_a_84[1] = '0;
assign low_a_48[0] = a_8[0][3:0];
assign low_a_48[1] = a_8[1][3:0];
assign low_a_88[0] = {4'd0, a_4[0]};
assign low_a_88[1] = {4'd0, a_4[1]};
assign low_a_44[0] = '0;
assign low_a_44[1] = '0;

assign low_b_84[0] = '0;
assign low_b_84[1] = '0;
assign low_b_48[0] = {4'd0,b_8[0][3:0]};
assign low_b_48[1] = {4'd0,b_8[1][3:0]};
assign low_b_88[0] = {4'd0, b_4[0]};
assign low_b_88[1] = {4'd0, b_4[1]};
assign low_b_44[0] = '0;
assign low_b_44[1] = '0;

//现在分配intfp4寄存器
assign intfp4_a_84[0] = {4'd0,a_8[0][3:0]};
assign intfp4_a_84[1] = {4'd0,a_8[1][3:0]};

assign intfp4_a_48[0] = a_8[0][7:4];
assign intfp4_a_48[1] = a_8[1][7:4];

assign intfp4_a_88[0] = {4'd0, a_4[0]};
assign intfp4_a_88[1] = {4'd0, a_4[1]};

assign intfp4_a_44[0] = exp_E6[0][3:0];
assign intfp4_a_44[1] = exp_E6[1][3:0];

assign intfp4_b_84[0] = b_8[0][3:0];
assign intfp4_b_84[1] = b_8[1][3:0];  

assign intfp4_b_48[0] = {4'd0, b_8[0][7:4]};
assign intfp4_b_48[1] = {4'd0, b_8[1][7:4]};

assign intfp4_b_88[0] = {4'd0, b_4[0]};
assign intfp4_b_88[1] = {4'd0, b_4[1]};

assign intfp4_b_44[0] = exp_E6[2][3:0];
assign intfp4_b_44[1] = exp_E6[3][3:0];

//现在分配int8寄存器
assign int8_a_84[0] = a_8[0];
assign int8_a_84[1] = a_8[1];

assign int8_a_48[0] = b_8[0][3:0];
assign int8_a_48[1] = b_8[1][3:0];

assign int8_a_88[0] = {a_4[0], exp_E6[0][3:0]};
assign int8_a_88[1] = {a_4[1], exp_E6[1][3:0]};

assign int8_a_44[0]= '0;
assign int8_a_44[1]= '0;

assign int8_b_84[0] = b_8[0][7:4];
assign int8_b_84[1] = b_8[1][7:4];

assign int8_b_48[0] = a_8[0];
assign int8_b_48[1] = a_8[1];

assign int8_b_88[0] = {b_4[0],exp_E6[2][3:0]};
assign int8_b_88[1] = {b_4[1],exp_E6[3][3:0]};

assign int8_b_44[0] = '0;
assign int8_b_44[1] = '0;

//选择对应的尾数寄存器
assign a_84 =  is_High_precision? high_a_84     :
               is_low_precision ? low_a_84      :
               is_INTFP4        ? intfp4_a_84   :
                                    int8_a_84   ;

assign a_88 =  is_High_precision? high_a_88     :
               is_low_precision ? low_a_88      :
               is_INTFP4        ? intfp4_a_88   :
                                    int8_a_88   ;

assign a_48 =  is_High_precision? high_a_48     : 
               is_low_precision ? low_a_48      :
               is_INTFP4        ? intfp4_a_48   :
                                    int8_a_48   ;

assign a_44 =  is_High_precision ? high_a_44    :
               is_low_precision  ? low_a_44     :
               is_INTFP4         ? intfp4_a_44  :
                                    int8_a_44   ;

assign b_84 =  is_High_precision ? high_b_84    :
               is_low_precision  ? low_b_84     :
               is_INTFP4         ? intfp4_b_84  :
                                     int8_b_84  ;

assign b_44 =  is_High_precision ? high_b_44    :
               is_low_precision  ? low_b_44     :
               is_INTFP4         ? intfp4_b_44  :
                                    int8_b_44   ;

assign b_48 =  is_High_precision ? high_b_48    :
               is_low_precision  ? low_b_48     :
               is_INTFP4         ? intfp4_b_48  :
                                     int8_b_48  ;

assign b_88 =  is_High_precision ? high_b_88    :
               is_low_precision  ? low_b_88     :
               is_INTFP4         ? intfp4_b_88  :
                                     int8_b_88  ;

    
endmodule
