`include "defines.sv"

module decode_opcode(
    input  wire [4: 0] opcode,

    output wire [3: 0] opcode_o,

    output wire is_TF32, is_BF16, is_FP16, 
    output wire is_E5M2, is_E3M2, is_E4M3, 
    output wire is_E2  , is_Int8, is_E8  ,
    output wire is_fp16_bf16,
    output wire is_mxnvfp4, 
    output wire is_Int4,
    output wire is_int,
    output wire is_intfp4,
    output wire is_mxint4,
    output wire is_mxfp8_6, is_mxfp8,

    output wire        is_High_precision_o,
    output wire        is_low_precision_o,
    output wire        is_MXNVFP4_o,
    output wire        is_INT8_o,
    output wire        is_INT4_o,
    output wire        is_MXINT4_o,
    output wire        is_INTFP4_o,

    output wire AS_have, BS_have// 1: s8/s4; 0: u8/u4
);

assign is_TF32    = (opcode == `DataTypeTF32TF32F32 );
assign is_BF16    = (opcode == `DataTypeBF16BF16F32 );
assign is_FP16    = (opcode == `DataTypeF16F16F32   );
assign is_E5M2    = (opcode == `DataTypeMxfp8e5m2F32 || opcode == `DataTypefp8e5m2F32);
assign is_E3M2    = (opcode == `DataTypeMxfp6e3m2F32);
assign is_E4M3    = (opcode == `DataTypeMxfp8e4m3F32 || opcode == `DataTypefp8e4m3F32);
assign is_E2      = (opcode == `DataTypeMxfp6e2m3F32);
assign is_Int8    = (opcode == `DataTypeI8I8I32      || opcode == `DataTypeI8U8I32 || opcode == `DataTypeU8I8I32  || opcode == `DataTypeU8U8I32 );
assign is_Int4    = (opcode == `DataTypeI4I4I32      || opcode == `DataTypeI4U4I32 || opcode == `DataTypeU4I4I32  || opcode == `DataTypeU4U4I32);
assign is_mxint4  = (opcode == `DataTypemxint4F32);
assign is_mxnvfp4 = (opcode == `DataTypenvfp4F32     || opcode == `DataTypemxfp4F32);
assign is_E8      = is_TF32 || is_BF16;
assign is_fp16_bf16 = is_BF16 || is_FP16;

assign is_mxfp8   = opcode == `DataTypeMxfp8e4m3F32 || opcode == `DataTypeMxfp8e5m2F32;
assign is_mxfp8_6   = opcode == `DataTypeMxfp6e2m3F32 || opcode == `DataTypeMxfp6e3m2F32 || is_mxfp8;

assign is_intfp4  = is_Int4 || is_mxnvfp4 || is_mxint4;

assign is_int     = is_Int8 || is_Int4 || is_mxint4;

assign AS_have    = (opcode == `DataTypeI8I8I32 || opcode == `DataTypeI8U8I32 || opcode == `DataTypeI4I4I32 || opcode == `DataTypeI4U4I32 || opcode == `DataTypemxint4F32) ? 1'd1 : 1'd0;
assign BS_have    = (opcode == `DataTypeI8I8I32 || opcode == `DataTypeU8I8I32 || opcode == `DataTypeI4I4I32 || opcode == `DataTypeU4I4I32 || opcode == `DataTypemxint4F32) ? 1'd1 : 1'd0;

assign is_High_precision_o = (is_TF32 || is_BF16 || is_FP16);
assign is_low_precision_o  = (is_E5M2 || is_E3M2 || is_E4M3 || is_E2);
assign is_MXNVFP4_o        = is_mxnvfp4;
assign is_INT8_o           = is_Int8;
assign is_INT4_o           = is_Int4;
assign is_INTFP4_o         = is_intfp4;
assign is_MXINT4_o         = is_mxint4;

//这个往下传，如果不是int4/mxint4，则接入原有的opcode，反之则无意义
assign opcode_o = (is_Int4 || is_mxint4) ? 4'd15 : opcode[3:0];

endmodule
