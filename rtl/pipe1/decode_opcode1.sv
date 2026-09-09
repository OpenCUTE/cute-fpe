`include "defines.sv"

module decode_opcode1 (
    input  wire [3 :0] opcode,
    input  wire        is_mxintfp4,

    // output wire [1 :0] data_type,
    output wire        scale_type,
    output wire        scale_en  ,

    output wire    is_threshold_normal,
    output wire    is_threshold_mxintfp4,

    output wire signed [7 :0] bias
);

wire is_TF32    = (opcode == `DataTypeTF32TF32F32 );
wire is_BF16    = (opcode == `DataTypeBF16BF16F32 );
wire is_FP16    = (opcode == `DataTypeF16F16F32   );
wire is_E5M2    = (opcode == `DataTypeMxfp8e5m2F32 || opcode == `DataTypefp8e5m2F32);
wire is_E3M2    = (opcode == `DataTypeMxfp6e3m2F32);
wire is_E4M3    = (opcode == `DataTypeMxfp8e4m3F32 || opcode == `DataTypefp8e4m3F32);
wire is_E2      = (opcode == `DataTypeMxfp6e2m3F32);

// assign data_type  =  (is_TF32 || is_BF16 || is_FP16) ? `DATA_High_precision :
//                      (is_E5M2 || is_E3M2 || is_E4M3 || is_E2  ) ? `DATA_low_precision  :
//                      (is_mxnvfp4                   ) ? `DATA_MXNVFP4        :
//                                                        `DATA_INT8            ;

assign scale_type      = (opcode == `DataTypenvfp4F32) ? `SCALE_UE4M3 : `SCALE_UE8M0;

wire   is_mxfp8fp6fp4  = (opcode == `DataTypeMxfp8e4m3F32 || opcode == `DataTypeMxfp8e5m2F32 || opcode == `DataTypeMxfp6e3m2F32 || opcode == `DataTypeMxfp6e2m3F32);
assign scale_en        = (is_mxfp8fp6fp4 || is_mxintfp4) ? 1'd1: 1'd0;

assign is_threshold_normal = is_TF32 || is_BF16 || is_mxfp8fp6fp4;
assign is_threshold_mxintfp4 = is_mxintfp4 && !scale_type;

assign bias = (is_TF32 || is_BF16) ? 8'sd127    :
              (is_FP16 || is_E5M2) ? -8'sd97    :
              (is_E4M3           ) ? -8'sd113   :
              (is_E3M2           ) ? -8'sd121   :
              (is_E2             ) ? -8'sd125   :
                                     -8'sd127   ;

endmodule
