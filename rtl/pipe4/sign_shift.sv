`include "defines.sv"

module sign_shift (
    input  wire [21:0] product_i   [0:1],
    // input  wire [1 :0] data_type,
    input  wire        is_High_precision,
    input  wire        is_low_precision,
    input  wire        is_MXINTFP4,
    // input  wire        is_INT8,
    input  wire [4 :0] exp_shift [0:3],

    input  wire [3 :0] vec4_sign,

    output wire signed [27:0] sign_sig [0:3]
);

//现在我们不需要这个product信号作为中介，我们直接使用将输入信号连接到这个sign_after_shift信号上去就行了
// wire [31:0] product [0:1];
// assign product[0] = is_INT8 ? {exp_shift[1][4:0],product_i[0][21:11],exp_shift[0][4:0],product_i[0][10:0]} : 
//                     is_High_precision ? {4'd0, product_i[0],5'd0} : 
//                                         {5'd0, product_i[0][21:11],5'd0, product_i[0][10:0]} ;

// assign product[1] = is_INT8 ? {exp_shift[3][4:0],product_i[1][21:11],exp_shift[2][4:0],product_i[1][10:0]} :
//                     is_High_precision ? {4'd0, product_i[1],5'd0} :
//                                         {5'd0, product_i[1][21:11],5'd0, product_i[1][10:0]} ;

// wire [26:0] sign_after_shift [0:3];

// // shift stage
// assign sign_after_shift[0] = (is_High_precision) ? (product[0][26:0] >> exp_shift[0])           :
//                              (is_low_precision ) ? ({product[0][9:0], 19'd0} >> exp_shift[0])  :
//                                                    {11'd0, product[0][15:0]}                    ;

// assign sign_after_shift[1] = (is_High_precision) ? (product[1][26:0] >> exp_shift[1]) :
//                              (is_low_precision ) ? ({product[0][25:16], 19'd0} >> exp_shift[1]) :
//                              {11'd0, product[0][31:16]};

// assign sign_after_shift[2] = (is_High_precision) ? 27'd0                                        :
//                              (is_low_precision ) ? ({product[1][9:0], 19'd0} >> exp_shift[2])  :
//                              {11'd0, product[1][15:0]}                                          ;

// assign sign_after_shift[3] = (is_High_precision) ? 27'd0                                        :
//                              (is_low_precision ) ? ({product[1][25:16], 19'd0} >> exp_shift[3]) :
//                              {11'd0, product[1][31:16]}                                         ;


wire [26:0] sign_after_shift [0:3];

// shift stage（我们将int8和mxnvfp4合并在一起，但是在此之前需要将exp_shift信号做一下处理，将其在mxnvfp4点时候设置为0）
assign sign_after_shift[0] = (is_High_precision) ? ({product_i[0],5'd0} >> exp_shift[0])            :
                             (is_low_precision ) ? ({product_i[0][9:0], 19'd0} >> exp_shift[0])       :
                                                   {11'd0, exp_shift[0][4:0], product_i[0][10:0]}   ;

assign sign_after_shift[1] = (is_High_precision) ? ({product_i[1],5'd0} >> exp_shift[1])            :
                             (is_low_precision ) ? ({product_i[0][20:11], 19'd0} >> exp_shift[1])     :
                             {11'd0, exp_shift[1][4:0],product_i[0][21:11]};

assign sign_after_shift[2] = (is_High_precision) ? 27'd0                                            :
                             (is_low_precision ) ? ({product_i[1][9:0], 19'd0} >> exp_shift[2])       :
                             {11'd0, exp_shift[2][4:0], product_i[1][10:0]}                         ;

assign sign_after_shift[3] = (is_High_precision) ? 27'd0                                            :
                             (is_low_precision ) ? ({product_i[1][20:11], 19'd0} >> exp_shift[3])     :
                             {11'd0, exp_shift[3][4:0],product_i[1][21:11]}                  ;

// sign stage：对于nvmxfp4，因为内部本来就是带有符号的，所以我们只需要进行符号位扩展就可以了
assign sign_sig[0] = (is_MXINTFP4) ? {{18{sign_after_shift[0][9]}},sign_after_shift[0][9:0]} :
                     (vec4_sign[0]              ) ? -$signed({1'b0, sign_after_shift[0]})                 :
                                                    $signed({1'b0, sign_after_shift[0]})                  ;

assign sign_sig[1] = (is_MXINTFP4) ? {{18{sign_after_shift[1][9]}},sign_after_shift[1][9:0]} :
                     (vec4_sign[1]              ) ? -$signed({1'b0, sign_after_shift[1]})                  :
                                                    $signed({1'b0, sign_after_shift[1]})                   ;

assign sign_sig[2] = (is_MXINTFP4) ? {{18{sign_after_shift[2][9]}},sign_after_shift[2][9:0]} :
                     (vec4_sign[2]              ) ? -$signed({1'b0, sign_after_shift[2]})                  :
                                                    $signed({1'b0, sign_after_shift[2]})                   ;

assign sign_sig[3] = (is_MXINTFP4) ? {{18{sign_after_shift[3][9]}},sign_after_shift[3][9:0]} :
                     (vec4_sign[3]              ) ? -$signed({1'b0, sign_after_shift[3]})                  :
                                                    $signed({1'b0, sign_after_shift[3]})                   ;

endmodule
