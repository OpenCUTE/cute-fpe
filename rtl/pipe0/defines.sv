`define ReduceWidth       512
`define LANE_NUM         `ReduceWidth / 32
`define BLOCK_NUM         `LANE_NUM / 2
`define ResultWidth       32
`define MinGroupSize      16
`define MinDataTypeWidth  4
`define ScaleElementWidth 8
`define ScaleBusWidth     `ReduceWidth/`MinDataTypeWidth/`MinGroupSize*`ScaleElementWidth
`define Scale_NUM         `ScaleBusWidth / `ScaleElementWidth
`define cmptreelayers     4
`define fp8cmptreelayers  4
`define fp4GroupSize      `ReduceWidth / 4 / 16

//mxfp8/fp6 group
`define MXFP86_ZERO_GROUP_SIZE  ((`LANE_NUM >= 8) ? 8 : `LANE_NUM)

`define MXFP86_ZERO_GROUP_NUM   (`LANE_NUM / `MXFP86_ZERO_GROUP_SIZE)

`define FP4P0AddNum       2
`define FP4P1AddNum       (16 / `FP4P0AddNum)

//opcode
`define DataTypeI8I8I32      0     //I8 * I8 * I32
`define DataTypeF16F16F32    1     //FP16 * FP16 * FP32
`define DataTypeBF16BF16F32  2     //BF16 * BF16 * FP32
`define DataTypeTF32TF32F32  3     //TF32 * TF32 * FP32
`define DataTypeI8U8I32      4     //I8 * UI8 * I32
`define DataTypeU8I8I32      5     //U8 * I8 * I32
`define DataTypeU8U8I32      6     //U8 * U8 * I32
`define DataTypeMxfp8e4m3F32 7     //Mxfp8e4m3 * Mxfp8e4m3 * FP32
`define DataTypeMxfp8e5m2F32 8     //Mxfp8e5m2 * Mxfp8e5m2 * FP32
`define DataTypenvfp4F32     9     //NVFP4 * NVFP4 * FP32
`define DataTypemxfp4F32     10    //MXFP4 * MXFP4 * FP32
`define DataTypefp8e4m3F32   11    //FP8E4M3 * FP8E4M3 * FP32
`define DataTypefp8e5m2F32   12    //FP8E5M2 * FP8E5M2 * FP32
`define DataTypeMxfp6e3m2F32 13    //Mxfp6e3m2 * Mxfp6e3m2 * FP32
`define DataTypeMxfp6e2m3F32 14    //Mxfp6e2m3 * Mxfp6e2m3 * FP32
`define DataTypeI4I4I32      15    //I4 * I4 * I32
`define DataTypeI4U4I32      16    //I4 * U4 * I32
`define DataTypeU4I4I32      17    //U4 * I4 * I32
`define DataTypeU4U4I32      18    //U4 * U4 * I32
`define DataTypemxint4F32    19    //MXINT4 * MXINT4 * FP32

//opcode0_to1
`define DATA_High_precision  0
`define DATA_low_precision   1
`define DATA_MXNVFP4         2
`define DATA_INT8            3         

//scale_opcode
`define SCALE_UE8M0          0
`define SCALE_UE4M3          1

//求和数量
`define ADD_NUM0             4
`define ADD_NUM1             2
`define ADD_NUM2             4 / (512 / `ReduceWidth)
`define ADD_NUM3             LANE_NUM * 4 / `ADD_NUM0 / `ADD_NUM1 / `ADD_NUM2

`define ADD_GROUPS           (`LANE_NUM * 4) / `ADD_NUM0

`define P4_ADD0_OUT_NUM     8 / (512 / `ReduceWidth) //8
`define P4_ADD0_OUT_W       28 + $clog2(`ADD_NUM0) + $clog2(`ADD_NUM1)

`define P4_ADD1_OUT_NUM     2 //2

//定义两套加法流程的有效小数位
`define decimal_places0      25
`define decimal_places1      35

//定义两条路径的最小指数阈值
`define MIN_EXP_THRESHOLD0   -133
`define MIN_EXP_THRESHOLD1   -139


//每组block的整数宽度：11位幅值 + 1位符号，可覆盖NVFP4最大幅值2025
`define block_int_w          12
