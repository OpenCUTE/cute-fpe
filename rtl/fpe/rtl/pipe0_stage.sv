`include "defines.sv"

module pipe0_stage (
    input  wire        clock,
    input  wire        reset,

    // A 向量输入
    input  wire [`ReduceWidth - 1 : 0]   AVector_bits,
    input  wire                          AVector_valid,
    output wire                          AVector_ready,   // ← 接 InReady

    // B 向量输入
    input  wire [`ReduceWidth - 1 : 0]   BVector_bits,
    input  wire                          BVector_valid,
    output wire                          BVector_ready,   // ← 接 InReady

    // C 累加输入（FP32），pipe0 不处理，直接打一拍透传
    input  wire [`ResultWidth - 1 : 0]   CAdd_bits,
    input  wire                          CAdd_valid,
    output wire                          CAdd_ready,      // ← 接 InReady

    // A/B Scale 输入（MX/NV 格式），pipe0 不处理，直接打一拍透传
    input  wire [`ScaleBusWidth - 1 : 0] AScale_bits,
    input  wire                          AScale_valid,
    output wire                          AScale_ready,   // ← 接 InReady

    input  wire [`ScaleBusWidth - 1 : 0] BScale_bits,
    input  wire                          BScale_valid,
    output wire                          BScale_ready,   // ← 接 InReady

    // 操作码
    input  wire [4:0]                    opcode,

    //握手信号
    input  wire        allow1_in_i,   // 下游（pipe1）允许接收
    output reg         valid0_o,      // 本级向下游的有效信号

    // ===================== 输出到 pipe1_stage =====================

    // 透传信号（打一拍）
    output reg  [`ResultWidth - 1 : 0]   CAdd_bits_r,
    output reg  [`ScaleBusWidth - 1 : 0] AScale_bits_r,
    output reg  [`ScaleBusWidth - 1 : 0] BScale_bits_r,

    // 异常标志（decode_stage 已聚合为单 bit）
    output reg         NAN_o_r,
    output reg         PINF_o_r,
    output reg         NINF_o_r,

    // opcode 透传
    output reg  [3:0]  opcode_r,

    output reg       is_High_precision_r,
    output reg       is_low_precision_r,
    output reg       is_MXNVFP4_r,
    output reg       is_INT8_r,
    output reg       is_INT4_r,
    output reg       is_MXINT4_r,
    output reg       is_INTFP4_r,
    output reg       is_MXINTFP4_r,

    // 尾数乘法器输入（每通道 2 组）
    output reg  [7:0] a_8_r [0:`LANE_NUM-1][0:1],
    output reg  [3:0] a_4_r [0:`LANE_NUM-1][0:1],
    output reg  [3:0] b_4_r [0:`LANE_NUM-1][0:1],
    output reg  [7:0] b_8_r [0:`LANE_NUM-1][0:1],

    // 指数求和结果
    output reg  [5:0] exp_E6_r [0:`LANE_NUM-1][0:3],

    output reg  [`Scale_NUM-1:0] MXINTFP4_all_zero_r,

    output reg [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero_r,

    // 符号位向量
    output reg  [7:0] vec8_sign_r [0:`LANE_NUM-1]
);
    wire   allow0_in_o;   // 本级向上游的反压
    wire   input_valid;

    //因为握手协议需要用这个信号，所以需要提前定义（否则DC会报错呀！！！）
    wire       is_MXINTFP4_w;
    wire       is_MXFP8_6_w;

    //检测这个scale的使能情况
    wire scale_required = is_MXFP8_6_w || is_MXINTFP4_w;

    assign input_valid = AVector_valid && BVector_valid && CAdd_valid &&
                     (!scale_required || (AScale_valid && BScale_valid));

    //此时该级可以接收信号了，但是还需要保证该级的输入信号都是有有效信号
    assign allow0_in_o = (!valid0_o) || allow1_in_i;
    //此时将所有的输入信号的ready都接到allow0_in_o中
    assign AVector_ready = allow0_in_o;
    assign BVector_ready = allow0_in_o;
    assign CAdd_ready    = allow0_in_o;
    assign AScale_ready  = allow0_in_o;
    assign BScale_ready  = allow0_in_o;

    always @(posedge clock or posedge reset) begin
        if(reset)begin
            valid0_o <= 1'd0;
        end
        else if(allow0_in_o) begin
            valid0_o <= input_valid;
        end
        else begin
            valid0_o <= valid0_o;
        end
    end


    /*=================================================================
     * decode_stage 输出连线（组合逻辑结果，待寄存）
     *=================================================================*/
    wire        NAN_w;
    wire        PINF_w;
    wire        NINF_w;

    wire [3:0] opcode_w;

    wire       is_High_precision_w;
    wire       is_low_precision_w;
    wire       is_MXNVFP4_w;
    wire       is_INT8_w;
    wire       is_INT4_w;
    wire       is_MXINT4_w;
    wire       is_INTFP4_w;

    wire [7:0] a_8_w [0:`LANE_NUM-1][0:1];
    wire [3:0] a_4_w [0:`LANE_NUM-1][0:1];
    wire [3:0] b_4_w [0:`LANE_NUM-1][0:1];
    wire [7:0] b_8_w [0:`LANE_NUM-1][0:1];

    wire [5:0] exp_E6_w [0:`LANE_NUM-1][0:3];

    wire [`Scale_NUM-1:0] MXINTFP4_all_zero_w;
    wire [`MXFP86_ZERO_GROUP_NUM-1:0] MXFP86_all_zero_w;

    wire [7:0] vec8_sign_w [0:`LANE_NUM-1];

    /*=================================================================
     * 例化 decode_stage（端口严格对齐实际定义）
     *=================================================================*/
    decode_stage u_decode_stage (
        .AVector_bits   (AVector_bits   ),
        .BVector_bits   (BVector_bits   ),
        .opcode         (opcode         ),

        // 异常输出（单 bit，内部已聚合）
        .NAN            (NAN_w          ),
        .PINF           (PINF_w         ),
        .NINF           (NINF_w         ),

        .opcode_o       (opcode_w       ),

        .is_High_precision_o (is_High_precision_w),
        .is_low_precision_o  (is_low_precision_w ),
        .is_MXNVFP4_o        (is_MXNVFP4_w       ),
        .is_INT8_o           (is_INT8_w          ),
        .is_INT4_o           (is_INT4_w          ),
        .is_MXINT4_o         (is_MXINT4_w        ),
        .is_INTFP4_o         (is_INTFP4_w        ),
        .is_MXINTFP4_o       (is_MXINTFP4_w      ),
        .is_MXFP8_6_o        (is_MXFP8_6_w       ),

        // 尾数
        .a_8            (a_8_w          ),
        .a_4            (a_4_w          ),
        .b_4            (b_4_w          ),
        .b_8            (b_8_w          ),

        // 指数
        .exp_E6         (exp_E6_w       ),

        // 分组fp4全为0的标志
        .MXINTFP4_all_zero   (MXINTFP4_all_zero_w   ),
        //mxfp8/fp6全是0的标志
        .MXFP86_all_zero     (MXFP86_all_zero_w     ),

        // 符号位
        .vec8_sign      (vec8_sign_w    )
    );

    /*=================================================================
     * 时序逻辑：所有信号统一在时钟上升沿打一拍输出
     * reset 为高电平同步复位
     *=================================================================*/
    integer k, g;

    always @(posedge clock or posedge reset) begin
        if (reset) begin
            // ----- 透传信号复位 -----
            CAdd_bits_r    <= {`ResultWidth{1'b0}};
            AScale_bits_r  <= {`ScaleBusWidth{1'b0}};
            BScale_bits_r  <= {`ScaleBusWidth{1'b0}};

            // ----- 异常标志复位 -----
            NAN_o_r        <= 1'b0;
            PINF_o_r       <= 1'b0;
            NINF_o_r       <= 1'b0;

            opcode_r       <= 4'd0;
            is_High_precision_r <= 1'b0;
            is_low_precision_r  <= 1'b0;
            is_MXNVFP4_r        <= 1'b0;
            is_INT8_r           <= 1'b0;
            is_INT4_r           <= 1'b0;
            is_MXINT4_r         <= 1'b0;
            is_INTFP4_r         <= 1'b0;
            is_MXINTFP4_r       <= 1'b0;

            MXINTFP4_all_zero_r <= {`Scale_NUM{1'b0}};
            MXFP86_all_zero_r   <= 'b0;

            // ----- 数组信号复位（展开循环）-----
            for (k = 0; k < `LANE_NUM; k = k + 1) begin
                vec8_sign_r[k] <= 8'd0;
                for (g = 0; g < 2; g = g + 1) begin
                    a_8_r[k][g] <= 8'd0;
                    a_4_r[k][g] <= 4'd0;
                    b_4_r[k][g] <= 4'd0;
                    b_8_r[k][g] <= 8'd0;
                    exp_E6_r[k][g] <= 6'd0;
                    exp_E6_r[k][g+2] <= 6'd0;
                end
            end

        end else if(allow0_in_o) begin
            // ----- 透传信号打拍 -----
            CAdd_bits_r    <= CAdd_bits;
            AScale_bits_r  <= AScale_bits;
            BScale_bits_r  <= BScale_bits;

            // ----- 异常标志打拍 -----
            NAN_o_r        <= NAN_w;
            PINF_o_r       <= PINF_w;
            NINF_o_r       <= NINF_w;

            opcode_r       <= opcode_w;
            is_High_precision_r <= is_High_precision_w;
            is_low_precision_r  <= is_low_precision_w;
            is_MXNVFP4_r        <= is_MXNVFP4_w;
            is_INT8_r           <= is_INT8_w;
            is_INT4_r           <= is_INT4_w;
            is_MXINT4_r         <= is_MXINT4_w;
            is_INTFP4_r         <= is_INTFP4_w;
            is_MXINTFP4_r       <= is_MXINTFP4_w;

            MXINTFP4_all_zero_r <= MXINTFP4_all_zero_w;
            MXFP86_all_zero_r   <= MXFP86_all_zero_w;

            // ----- 数组信号打拍 -----
            for (k = 0; k < `LANE_NUM; k = k + 1) begin
                vec8_sign_r[k] <= vec8_sign_w[k];
                for (g = 0; g < 2; g = g + 1) begin
                    a_8_r[k][g]    <= a_8_w[k][g];
                    a_4_r[k][g]    <= a_4_w[k][g];
                    b_4_r[k][g]    <= b_4_w[k][g];
                    b_8_r[k][g]    <= b_8_w[k][g];
                    exp_E6_r[k][g] <= exp_E6_w[k][g];
                    exp_E6_r[k][g+2] <= exp_E6_w[k][g+2];
                end
            end
        end
    end

endmodule
