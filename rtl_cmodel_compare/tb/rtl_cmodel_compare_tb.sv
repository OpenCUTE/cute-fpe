`timescale 1ns/1ps

`include "defines.sv"

module rtl_cmodel_compare_tb;
    logic clock;
    logic reset;

    logic [`ReduceWidth - 1:0] AVector_bits;
    logic AVector_valid;
    wire  AVector_ready;

    logic [`ReduceWidth - 1:0] BVector_bits;
    logic BVector_valid;
    wire  BVector_ready;

    logic [`ResultWidth - 1:0] CAdd_bits;
    logic CAdd_valid;
    wire  CAdd_ready;

    logic [`ScaleBusWidth - 1:0] AScale_bits;
    logic AScale_valid;
    wire  AScale_ready;

    logic [`ScaleBusWidth - 1:0] BScale_bits;
    logic BScale_valid;
    wire  BScale_ready;

    logic [4:0] opcode;

    wire valid8_o;
    logic allow9_in_i;
    wire [31:0] D_bits_o;

    top dut (
        .clock(clock),
        .reset(reset),
        .AVector_bits(AVector_bits),
        .AVector_valid(AVector_valid),
        .BVector_bits(BVector_bits),
        .BVector_valid(BVector_valid),
        .CAdd_bits(CAdd_bits),
        .CAdd_valid(CAdd_valid),
        .AScale_bits(AScale_bits),
        .AScale_valid(AScale_valid),
        .BScale_bits(BScale_bits),
        .BScale_valid(BScale_valid),
        .opcode(opcode),
        .AVector_ready(AVector_ready),
        .BVector_ready(BVector_ready),
        .CAdd_ready(CAdd_ready),
        .AScale_ready(AScale_ready),
        .BScale_ready(BScale_ready),
        .valid8_o(valid8_o),
        .allow9_in_i(allow9_in_i),
        .D_bits_o(D_bits_o)
    );

    string a_hex_path;
    string b_hex_path;
    string c_hex_path;
    string ascale_hex_path;
    string bscale_hex_path;
    string expected_hex_path;
    string fail_log_path;
    string wave_file_path;

    int vec_count;
    int vec_start;
    int timeout_cycles;
    int sent_count;
    int recv_count;
    int fail_count;
    int pass_count;
    int opcode_int;
    int ulp_tolerance;
    int stop_on_fail;
    int verbose_pass;
    int dump_wave;
    int fail_log_fd;
    logic scale_inputs_required;

    localparam int MAX_VEC_COUNT = 100000;

    logic [`ReduceWidth - 1:0] a_vec_mem [0:MAX_VEC_COUNT-1];
    logic [`ReduceWidth - 1:0] b_vec_mem [0:MAX_VEC_COUNT-1];
    logic [31:0] c_vec_mem [0:MAX_VEC_COUNT-1];
    logic [`ScaleBusWidth - 1:0] ascale_vec_mem [0:MAX_VEC_COUNT-1];
    logic [`ScaleBusWidth - 1:0] bscale_vec_mem [0:MAX_VEC_COUNT-1];
    logic [31:0] expected_mem [0:MAX_VEC_COUNT-1];
    int expect_queue [0:MAX_VEC_COUNT-1];
    int eq_wr_ptr;
    int eq_rd_ptr;

    initial begin
        clock = 1'b0;
        forever #5 clock = ~clock;
    end

    always @(posedge clock) begin
        if (!reset && !scale_inputs_required) begin
            if (AScale_valid || BScale_valid) begin
                $fatal(1, "Scale valid must remain low for opcode=%0d", opcode_int);
            end
        end
    end

    function automatic void require_plusarg_string(input string name, output string value);
        if (!$value$plusargs({name, "=%s"}, value)) begin
            $fatal(1, "Missing plusarg +%s=<path>", name);
        end
    endfunction

    function automatic int ulp_distance(input logic [31:0] a, input logic [31:0] b);
        int da;
        int db;
        begin
            if (a == b) begin
                ulp_distance = 0;
            end else if (a[31] != b[31]) begin
                ulp_distance = ((a[30:0] == 0) && (b[30:0] == 0)) ? 0 : 32'h7fff_ffff;
            end else begin
                da = {1'b0, a[30:0]};
                db = {1'b0, b[30:0]};
                ulp_distance = (da > db) ? (da - db) : (db - da);
            end
        end
    endfunction

    function automatic logic opcode_requires_scale(input int opcode_value);
        case (opcode_value)
            7, 8, 9, 10, 13, 14, 19: opcode_requires_scale = 1'b1;
            default:                 opcode_requires_scale = 1'b0;
        endcase
    endfunction

    function automatic logic [`ScaleBusWidth - 1:0] random_scale_bits();
        logic [`ScaleBusWidth - 1:0] value;
        begin
            value = {$urandom, $urandom};
            random_scale_bits = value;
        end
    endfunction

    initial begin
        if (!$value$plusargs("VEC_COUNT=%d", vec_count)) begin
            vec_count = 1;
        end
        if (!$value$plusargs("VEC_START=%d", vec_start)) begin
            vec_start = 0;
        end
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
            timeout_cycles = 2000;
        end
        if (!$value$plusargs("OPCODE=%d", opcode_int)) begin
            opcode_int = 1;
        end
        if (!$value$plusargs("ULP_TOLERANCE=%d", ulp_tolerance)) begin
            ulp_tolerance = 0;
        end
        if (!$value$plusargs("STOP_ON_FAIL=%d", stop_on_fail)) begin
            stop_on_fail = 1;
        end
        if (!$value$plusargs("VERBOSE_PASS=%d", verbose_pass)) begin
            verbose_pass = 0;
        end
        if (!$value$plusargs("DUMP_WAVE=%d", dump_wave)) begin
            dump_wave = 0;
        end
        if (vec_count <= 0 || vec_start < 0 || vec_start + vec_count > MAX_VEC_COUNT) begin
            $fatal(1,
                   "VEC_START/VEC_COUNT must select [0, %0d), got start=%0d count=%0d",
                   MAX_VEC_COUNT,
                   vec_start,
                   vec_count);
        end
        if (opcode_int < 0 || opcode_int > 19) begin
            $fatal(1, "Opcode must be in the supported RTL range [0, 19], got %0d", opcode_int);
        end
        scale_inputs_required = opcode_requires_scale(opcode_int);
        require_plusarg_string("A_HEX", a_hex_path);
        require_plusarg_string("B_HEX", b_hex_path);
        require_plusarg_string("C_HEX", c_hex_path);
        require_plusarg_string("ASCALE_HEX", ascale_hex_path);
        require_plusarg_string("BSCALE_HEX", bscale_hex_path);
        require_plusarg_string("EXPECTED_HEX", expected_hex_path);
        if (!$value$plusargs("FAIL_LOG=%s", fail_log_path)) begin
            fail_log_path = "fail_log.txt";
        end
        if (!$value$plusargs("WAVE_FILE=%s", wave_file_path)) begin
            wave_file_path = "tb_top.fst";
        end

        $readmemh(a_hex_path, a_vec_mem);
        $readmemh(b_hex_path, b_vec_mem);
        $readmemh(c_hex_path, c_vec_mem);
        $readmemh(ascale_hex_path, ascale_vec_mem);
        $readmemh(bscale_hex_path, bscale_vec_mem);
        $readmemh(expected_hex_path, expected_mem);

        fail_log_fd = $fopen(fail_log_path, "w");
        if (fail_log_fd != 0) begin
            $fdisplay(fail_log_fd,
                      "# vec_idx | A | B | C | AScale | BScale | got | expected | ULP");
        end else begin
            $display("WARNING failed to open fail log: %s", fail_log_path);
        end

        if (dump_wave != 0) begin
            $dumpfile(wave_file_path);
            $dumpvars(0, rtl_cmodel_compare_tb);
        end

        opcode = opcode_int[4:0];
        allow9_in_i = 1'b1;
        eq_wr_ptr = 0;
        eq_rd_ptr = 0;
        pass_count = 0;
        fail_count = 0;
        recv_count = 0;
        sent_count = 0;

        AVector_bits = '0;
        BVector_bits = '0;
        CAdd_bits = '0;
        AScale_bits = '0;
        BScale_bits = '0;
        AVector_valid = 1'b0;
        BVector_valid = 1'b0;
        CAdd_valid = 1'b0;
        AScale_valid = 1'b0;
        BScale_valid = 1'b0;

        reset = 1'b1;
        repeat (5) @(posedge clock);
        // Keep reset changes away from the DUT's posedge-triggered logic.
        @(negedge clock);
        reset = 1'b0;

        fork
            drive_vectors();
            check_outputs();
            watchdog();
        join_any
        disable fork;

        if (fail_count == 0 && recv_count == vec_count) begin
            $display("PASS compared %0d vectors opcode=%0d", vec_count, opcode_int);
            if (fail_log_fd != 0) begin
                $fclose(fail_log_fd);
            end
            $finish;
        end else begin
            if (fail_log_fd != 0) begin
                $fclose(fail_log_fd);
            end
            $fatal(1, "FAIL compared=%0d expected=%0d mismatches=%0d", recv_count, vec_count, fail_count);
        end
    end

    task automatic drive_vectors();
        int source_vec;
        sent_count = 0;
        while (sent_count < vec_count) begin
            source_vec = vec_start + sent_count;

            // Scaled formats use five channels; other formats use A/B/C only.
            do begin
                @(negedge clock);
            end while (!(AVector_ready && BVector_ready && CAdd_ready &&
                         (!scale_inputs_required || (AScale_ready && BScale_ready))));

            AVector_bits = a_vec_mem[source_vec];
            BVector_bits = b_vec_mem[source_vec];
            CAdd_bits = c_vec_mem[source_vec];
            AScale_bits = scale_inputs_required ? ascale_vec_mem[source_vec] : random_scale_bits();
            BScale_bits = scale_inputs_required ? bscale_vec_mem[source_vec] : random_scale_bits();
            AVector_valid = 1'b1;
            BVector_valid = 1'b1;
            CAdd_valid = 1'b1;
            AScale_valid = scale_inputs_required;
            BScale_valid = scale_inputs_required;

            @(posedge clock);
            #1ps;

            expect_queue[eq_wr_ptr] = source_vec;
            eq_wr_ptr++;
            sent_count++;

            // Drop valid after the handshake so a stalled next transfer
            // cannot resend the current vector.
            AVector_valid = 1'b0;
            BVector_valid = 1'b0;
            CAdd_valid = 1'b0;
            AScale_valid = 1'b0;
            BScale_valid = 1'b0;
        end

        // Keep this task alive until the checker has observed the final
        // result.  Otherwise drive_vectors can finish before the pipeline
        // drains and prematurely satisfy the enclosing join_any.
        while (recv_count < vec_count && !(stop_on_fail != 0 && fail_count != 0)) begin
            @(posedge clock);
        end
    endtask

    task automatic check_outputs();
        int out_idx;
        int ulp;
        forever begin
            @(posedge clock);
            #1ps;
            if (!reset && valid8_o && allow9_in_i) begin
                if (eq_rd_ptr >= eq_wr_ptr) begin
                    $fatal(1, "DUT produced output before an expected vector was queued");
                end

                out_idx = expect_queue[eq_rd_ptr];
                eq_rd_ptr++;
                ulp = ulp_distance(D_bits_o, expected_mem[out_idx]);

                recv_count++;
                if (ulp <= ulp_tolerance) begin
                    pass_count++;
                    if (verbose_pass != 0) begin
                        $display("[PASS] vec=%0d got=0x%08x expected=0x%08x ulp=%0d",
                                 out_idx, D_bits_o, expected_mem[out_idx], ulp);
                    end
                end else begin
                    fail_count++;
                    $display("[FAIL] vec=%0d opcode=%0d expected=0x%08x got=0x%08x ulp=%0d",
                             out_idx, opcode_int, expected_mem[out_idx], D_bits_o, ulp);
                    $display("  A=0x%0128x", a_vec_mem[out_idx]);
                    $display("  B=0x%0128x", b_vec_mem[out_idx]);
                    $display("  C=0x%08x", c_vec_mem[out_idx]);
                    if (fail_log_fd != 0) begin
                        $fdisplay(fail_log_fd,
                                  "%0d | %0128h | %0128h | %08h | %016h | %016h | %08h | %08h | %0d",
                                  out_idx,
                                  a_vec_mem[out_idx],
                                  b_vec_mem[out_idx],
                                  c_vec_mem[out_idx],
                                  ascale_vec_mem[out_idx],
                                  bscale_vec_mem[out_idx],
                                  D_bits_o,
                                  expected_mem[out_idx],
                                  ulp);
                    end
                    if (stop_on_fail != 0) begin
                        return;
                    end
                end
                if (recv_count == vec_count) begin
                    return;
                end
            end
        end
    endtask

    task automatic watchdog();
        repeat (timeout_cycles) @(posedge clock);
        $fatal(1, "Timeout after %0d cycles sent=%0d recv=%0d", timeout_cycles, sent_count, recv_count);
    endtask
endmodule
