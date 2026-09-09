`include "defines.sv"

// Adapter between CUTE's FReducePE BlackBox and the standalone SV FPE.
// Scale-valid handling belongs in pipe0_stage.sv; this module only adapts IO.
module CuteFpeSv (
  input  wire                    clock,
  input  wire                    reset,

  input  wire [`ReduceWidth-1:0] AVector_bits,
  input  wire                    AVector_valid,
  output wire                    AVector_ready,

  input  wire [`ReduceWidth-1:0] BVector_bits,
  input  wire                    BVector_valid,
  output wire                    BVector_ready,

  input  wire [`ResultWidth-1:0] CAdd_bits,
  input  wire                    CAdd_valid,
  output wire                    CAdd_ready,

  input  wire [`ScaleBusWidth-1:0] AScale_bits,
  input  wire                    AScale_valid,
  output wire                    AScale_ready,

  input  wire [`ScaleBusWidth-1:0] BScale_bits,
  input  wire                    BScale_valid,
  output wire                    BScale_ready,

  input wire [4:0] opcode,

  output wire [`ResultWidth-1:0] DResult_bits,
  output wire                    DResult_valid,
  input  wire                    DResult_ready
);

  top u_fpe (
    .clock(clock),
    .reset(reset),

    .AVector_bits(AVector_bits),
    .AVector_valid(AVector_valid),
    .AVector_ready(AVector_ready),

    .BVector_bits(BVector_bits),
    .BVector_valid(BVector_valid),
    .BVector_ready(BVector_ready),

    .CAdd_bits(CAdd_bits),
    .CAdd_valid(CAdd_valid),
    .CAdd_ready(CAdd_ready),

    .AScale_bits(AScale_bits),
    .AScale_valid(AScale_valid),
    .AScale_ready(AScale_ready),

    .BScale_bits(BScale_bits),
    .BScale_valid(BScale_valid),
    .BScale_ready(BScale_ready),

    .opcode(opcode),

    .D_bits_o(DResult_bits),
    .valid8_o(DResult_valid),
    .allow9_in_i(DResult_ready)
  );

endmodule
