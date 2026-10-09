`timescale 1ns/1ps

module proposal_bench #(parameter integer CASCADE = 0,
    parameter integer SKIP_ZERO_GROUPS = 0, parameter integer MEASURE_CYCLES = 0);
    localparam integer MAX_PATCHES = 2048;
    localparam integer INPUT_WORDS = CASCADE ? 512 : 96;
    localparam TEST_NAME = CASCADE ? "tb_proposal_engine" : "tb_readout_layer_engine";
    localparam INPUT_FILE = CASCADE ? "proposal_inputs.txt" : "readout_inputs.txt";
    localparam EXPECTED_FILE = CASCADE ? "proposal_expected.txt" : "readout_expected.txt";
    localparam TRACE_FILE = CASCADE ? "proposal_trace.txt" : "readout_trace.txt";
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1, s_valid = 0, m_ready = 0;
    logic [7:0] s_data = 0;
    logic exact_ready, approx_ready, exact_valid, approx_valid;
    logic [31:0] exact_data, approx_data;
    logic [7:0] exact_tree, approx_tree;
    logic [2:0] exact_mem, approx_mem;
    logic [7:0] inputs [0:MAX_PATCHES*INPUT_WORDS-1];
    logic [85:0] expected [0:MAX_PATCHES*16-1];
    logic [117:0] expected_trace [0:MAX_PATCHES*64-1];
    logic [81:0] expected_first [0:MAX_PATCHES*192-1];
    logic [81:0] expected_second [0:MAX_PATCHES*96-1];
    logic trace_valid, approx_trace_valid, first_valid, second_valid;
    logic second_busy, readout_busy;
    logic [117:0] trace_data;
    logic [81:0] first_data, second_data;
    integer descriptor, capture_descriptor, fields, patches, expected_records, row, fixture;
    logic [31:0] a, b, c, d, e, f, g, h;
    string extra;
    integer cycles = 0, received = 0, accepted = 0, traced = 0, first_count = 0, second_count = 0;
    integer records = 0, replay_records = 0, trace_records = 0, first_records = 0, second_records = 0;
    integer active_fixture = 0, start_cycle, duration;
    integer no_stall_cycles = 0, minimum_cycles = 2147483647, maximum_cycles = 0;
    integer backpressure_cycles = 0, gap_cycles = 0;
    integer exact_tree_total = 0, approx_tree_total = 0, exact_mem_total = 0, approx_mem_total = 0;
    bit checking = 0, regular_record = 0, stress = 0, force_stall = 0, was_stalled = 0;
    logic [85:0] held;

    generate if (CASCADE) begin : cascade
        proposal_engine #(.APPROX_LOW_BITS(0), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)) exact_dut (
            .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(exact_ready),
            .m_data(exact_data), .m_valid(exact_valid), .m_ready(m_ready),
            .m_tree_saturations(exact_tree), .m_membrane_saturations(exact_mem)
        );
        proposal_engine #(.APPROX_LOW_BITS(2), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)) approx_dut (
            .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(approx_ready),
            .m_data(approx_data), .m_valid(approx_valid), .m_ready(m_ready),
            .m_tree_saturations(approx_tree), .m_membrane_saturations(approx_mem)
        );
        assign trace_valid = exact_dut.readout.trace_valid;
        assign approx_trace_valid = approx_dut.readout.trace_valid;
        assign trace_data = {exact_dut.readout.membrane_out, exact_dut.readout.next_sum,
            exact_dut.readout.step_tree_saturations, exact_dut.readout.membrane_saturated,
            approx_dut.readout.membrane_out, approx_dut.readout.next_sum,
            approx_dut.readout.step_tree_saturations, approx_dut.readout.membrane_saturated};
        assign first_valid = exact_dut.hidden.link_valid && exact_dut.hidden.link_ready;
        assign second_valid = exact_dut.link_valid && exact_dut.link_ready;
        assign first_data = {exact_dut.hidden.link_data, exact_dut.hidden.link_tree_saturations,
            exact_dut.hidden.link_membrane_saturated, approx_dut.hidden.link_data,
            approx_dut.hidden.link_tree_saturations, approx_dut.hidden.link_membrane_saturated};
        assign second_data = {exact_dut.link_data, exact_dut.link_tree_saturations, exact_dut.link_membrane_saturated,
            approx_dut.link_data, approx_dut.link_tree_saturations, approx_dut.link_membrane_saturated};
        assign second_busy = !exact_dut.hidden.link_ready;
        assign readout_busy = !exact_dut.link_ready;
        always @(posedge clk) if (!rst &&
            ({exact_dut.hidden.link_valid, exact_dut.hidden.link_ready, exact_dut.link_valid, exact_dut.link_ready} !==
             {approx_dut.hidden.link_valid, approx_dut.hidden.link_ready, approx_dut.link_valid, approx_dut.link_ready}))
            $fatal(1, "Inter-layer handshake mismatch");
    end else begin : standalone
        readout_layer_engine #(.APPROX_LOW_BITS(0)) exact_dut (
            .clk(clk), .rst(rst), .s_data(s_data[0]), .s_valid(s_valid), .s_ready(exact_ready),
            .m_data(exact_data), .m_valid(exact_valid), .m_ready(m_ready),
            .m_tree_saturations(exact_tree), .m_membrane_saturations(exact_mem)
        );
        readout_layer_engine #(.APPROX_LOW_BITS(2)) approx_dut (
            .clk(clk), .rst(rst), .s_data(s_data[0]), .s_valid(s_valid), .s_ready(approx_ready),
            .m_data(approx_data), .m_valid(approx_valid), .m_ready(m_ready),
            .m_tree_saturations(approx_tree), .m_membrane_saturations(approx_mem)
        );
        assign trace_valid = exact_dut.trace_valid;
        assign approx_trace_valid = approx_dut.trace_valid;
        assign trace_data = {exact_dut.membrane_out, exact_dut.next_sum, exact_dut.step_tree_saturations,
            exact_dut.membrane_saturated, approx_dut.membrane_out, approx_dut.next_sum,
            approx_dut.step_tree_saturations, approx_dut.membrane_saturated};
        assign first_valid = 0;
        assign second_valid = 0;
        assign second_busy = 0;
        assign readout_busy = !exact_ready;
    end endgenerate

    always @(negedge clk) m_ready = !force_stall && (!stress || cycles % 7 >= 3);
    always @(posedge clk) begin
        cycles = cycles + 1;
        if (cycles > (MAX_PATCHES + 20) * 20000) $fatal(1, "Proposal cycle watchdog");
        if (rst) begin
            received = 0;
            accepted = 0;
            traced = 0;
            first_count = 0;
            second_count = 0;
            was_stalled = 0;
        end else begin
            if ({exact_ready, exact_valid, trace_valid} !== {approx_ready, approx_valid, approx_trace_valid})
                $fatal(1, "Exact/approx handshake differs");
            if (was_stalled && (!exact_valid || !approx_valid ||
                {exact_data, exact_tree, exact_mem, approx_data, approx_tree, approx_mem} !== held))
                $fatal(1, "Readout changed under backpressure");
            was_stalled = exact_valid && !m_ready;
            held = {exact_data, exact_tree, exact_mem, approx_data, approx_tree, approx_mem};
            if (checking) begin
                if (accepted == INPUT_WORDS && received < 16 && exact_ready)
                    $fatal(1, "Reopened input before patch completion");
                if (s_valid && exact_ready) accepted = accepted + 1;
                if (!s_valid && exact_ready) gap_cycles = gap_cycles + 1;
                if (exact_valid && !m_ready) backpressure_cycles = backpressure_cycles + 1;
                if (first_valid) begin
                    if (first_count >= 192 || first_data !== expected_first[active_fixture * 192 + first_count])
                        $fatal(1, "Layer-one mismatch");
                    first_count = first_count + 1;
                    if (regular_record) first_records = first_records + 1;
                end
                if (second_valid) begin
                    if (second_count >= 96 || second_data !== expected_second[active_fixture * 96 + second_count])
                        $fatal(1, "Layer-two mismatch");
                    second_count = second_count + 1;
                    if (regular_record) second_records = second_records + 1;
                end
                if (trace_valid) begin
                    if (traced >= 64 || trace_data !== expected_trace[active_fixture * 64 + traced])
                        $fatal(1, "Readout step mismatch: fixture %0d record %0d actual=%h expected=%h hidden=%0d/%0d",
                            active_fixture, traced, trace_data, expected_trace[active_fixture * 64 + traced], first_count, second_count);
                    traced = traced + 1;
                    if (regular_record) trace_records = trace_records + 1;
                end
                if (exact_valid && m_ready) begin
                    if (received >= 16 || held !== expected[active_fixture * 16 + received])
                        $fatal(1, "Logit mismatch: fixture %0d channel %0d exact=%h approximate=%h",
                            active_fixture, received, exact_data, approx_data);
                    if (regular_record) begin
                        // Capture DUT pins only on accepted transfers; reset replays are excluded.
                        if (CASCADE) $fdisplay(capture_descriptor, "%0d %0d %0d %08h %0d %0d %08h %0d %0d",
                            active_fixture, received, cycles, exact_data, exact_tree, exact_mem,
                            approx_data, approx_tree, approx_mem);
                        records = records + 1;
                        exact_tree_total = exact_tree_total + exact_tree;
                        approx_tree_total = approx_tree_total + approx_tree;
                        exact_mem_total = exact_mem_total + exact_mem;
                        approx_mem_total = approx_mem_total + approx_mem;
                    end else replay_records = replay_records + 1;
                    received = received + 1;
                end
            end
        end
    end

    task automatic reset_engine;
        begin
            @(negedge clk);
            checking = 0;
            rst = 1;
            s_valid = 0;
            repeat (3) @(negedge clk);
            if (exact_valid || approx_valid || trace_valid || approx_trace_valid)
                $fatal(1, "Valid during reset");
            rst = 0;
            force_stall = 0;
            @(negedge clk);
            if (!exact_ready || !approx_ready || exact_valid || approx_valid) $fatal(1, "Reset did not empty pipeline");
        end
    endtask
    task automatic send_inputs(input integer index_fixture, input integer count, input bit gaps);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                if (gaps && index % 11 == 4) begin
                    s_valid = 0;
                    repeat (2) @(negedge clk);
                end
                s_data = inputs[index_fixture * INPUT_WORDS + index];
                s_valid = 1;
                @(posedge clk);
                while (!exact_ready) @(posedge clk);
            end
            @(negedge clk);
            s_valid = 0;
        end
    endtask
    task automatic run_patch(input integer index_fixture, input bit gaps, input bit count_regular);
        begin
            @(negedge clk);
            active_fixture = index_fixture;
            received = 0;
            accepted = 0;
            traced = 0;
            first_count = 0;
            second_count = 0;
            regular_record = count_regular;
            checking = 1;
            stress = gaps;
            start_cycle = cycles;
            send_inputs(index_fixture, INPUT_WORDS, gaps);
            wait (received == 16);
            @(negedge clk);
            duration = cycles - start_cycle;
            if (count_regular) begin
                if (duration < minimum_cycles) minimum_cycles = duration;
                if (duration > maximum_cycles) maximum_cycles = duration;
                if (!gaps) no_stall_cycles = duration;
                if (MEASURE_CYCLES) $display("WD_CYCLE patch=%0d cycles=%0d", index_fixture, duration);
            end
            if (!exact_ready || exact_valid || approx_valid || accepted != INPUT_WORDS || traced != 64 ||
                (CASCADE && (first_count != 192 || second_count != 96))) $fatal(1, "Incomplete patch");
            // Keep the scoreboard armed between patches to reject stray outputs/traces.
        end
    endtask
    task automatic load_hidden(input string filename, input integer count, input bit first_layer);
        integer index;
        begin
            descriptor = $fopen(filename, "r");
            if (!descriptor) $fatal(1, "Missing hidden oracle");
            for (index = 0; index < count; index = index + 1) begin
                fields = $fscanf(descriptor, "%h %d %d %h %d %d", a, b, c, d, e, f);
                if (fields != 6 || $isunknown({a,b,c,d,e,f})) $fatal(1, "Malformed/truncated hidden oracle");
                if (first_layer) expected_first[index] = {a,b[7:0],c[0],d,e[7:0],f[0]};
                else expected_second[index] = {a,b[7:0],c[0],d,e[7:0],f[0]};
            end
            fields = $fscanf(descriptor, "%s", extra);
            if (fields == 1) $fatal(1, "Extra hidden oracle");
            $fclose(descriptor);
        end
    endtask
    initial begin
        descriptor = $fopen("expected_records.txt", "r");
        if (!descriptor) $fatal(1, "Missing record count");
        fields = $fscanf(descriptor, "%d", expected_records);
        $fclose(descriptor);
        if (fields != 1 || expected_records <= 0 || expected_records % 16 != 0) $fatal(1, "Invalid count");
        patches = expected_records / 16;
        if (patches < 2 || patches > MAX_PATCHES) $fatal(1, "Invalid patch count");
        descriptor = $fopen(INPUT_FILE, "r");
        if (!descriptor) $fatal(1, "Missing inputs");
        for (row = 0; row < patches * INPUT_WORDS; row = row + 1) begin
            fields = $fscanf(descriptor, "%h", inputs[row]);
            if (fields != 1 || $isunknown(inputs[row]) || (!CASCADE && inputs[row] > 1)) $fatal(1, "Malformed input");
        end
        fields = $fscanf(descriptor, "%s", extra);
        if (fields == 1) $fatal(1, "Extra input");
        $fclose(descriptor);
        descriptor = $fopen(EXPECTED_FILE, "r");
        if (!descriptor) $fatal(1, "Missing expected logits");
        for (row = 0; row < expected_records; row = row + 1) begin
            fields = $fscanf(descriptor, "%h %d %d %h %d %d", a,b,c,d,e,f);
            if (fields != 6 || $isunknown({a,b,c,d,e,f})) $fatal(1, "Malformed logit expectation");
            expected[row] = {a,b[7:0],c[2:0],d,e[7:0],f[2:0]};
        end
        fields = $fscanf(descriptor, "%s", extra);
        if (fields == 1) $fatal(1, "Extra logit expectation");
        $fclose(descriptor);
        descriptor = $fopen(TRACE_FILE, "r");
        if (!descriptor) $fatal(1, "Missing readout trace");
        for (row = 0; row < patches * 64; row = row + 1) begin
            fields = $fscanf(descriptor, "%h %h %d %d %h %h %d %d", a,b,c,d,e,f,g,h);
            if (fields != 8 || $isunknown({a,b,c,d,e,f,g,h})) $fatal(1, "Malformed readout trace");
            expected_trace[row] = {a[23:0],b[25:0],c[7:0],d[0],e[23:0],f[25:0],g[7:0],h[0]};
        end
        fields = $fscanf(descriptor, "%s", extra);
        if (fields == 1) $fatal(1, "Extra readout trace");
        $fclose(descriptor);
        if (CASCADE) begin
            load_hidden("proposal_layer1.txt", patches * 192, 1);
            load_hidden("proposal_layer2.txt", patches * 96, 0);
            capture_descriptor = $fopen("proposal_actual.txt", "w");
            if (!capture_descriptor) $fatal(1, "Cannot open actual output capture");
            $fdisplay(capture_descriptor, "WD_CAPTURE version=1 patches=%0d channels=16", patches);
        end
        reset_engine();
        send_inputs(1, 50, 0);
        reset_engine();
        run_patch(1, 1, 0);
        checking = 0;
        send_inputs(1, INPUT_WORDS, 0);
        repeat (100) @(negedge clk);
        reset_engine();
        run_patch(1, 0, 0);
        if (CASCADE) begin
            checking = 0;
            send_inputs(1, INPUT_WORDS, 0);
            wait (second_busy);
            repeat (100) @(negedge clk);
            reset_engine();
            run_patch(1, 1, 0);
            checking = 0;
            send_inputs(1, INPUT_WORDS, 0);
            wait (readout_busy);
            repeat (100) @(negedge clk);
            reset_engine();
            run_patch(1, 0, 0);
        end
        checking = 0;
        force_stall = 1;
        send_inputs(1, INPUT_WORDS, 0);
        wait (exact_valid);
        @(negedge clk);
        s_valid = 1;
        s_data = 8'hAA;
        repeat (5) begin
            @(negedge clk);
            if (exact_ready || approx_ready) $fatal(1, "Accepted a new patch under backpressure");
        end
        reset_engine();
        run_patch(1, 1, 0);
        for (fixture = 0; fixture < patches; fixture = fixture + 1)
            run_patch(fixture, !MEASURE_CYCLES && fixture % 2 != 0, 1);
        repeat (20) @(negedge clk);
        if (records != expected_records || replay_records != (CASCADE ? 80 : 48) || trace_records != patches * 64 ||
            !backpressure_cycles || !gap_cycles || (CASCADE && (first_records != patches * 192 || second_records != patches * 96)))
            $fatal(1, "Incomplete Proposal coverage");
        $display("WD_STAT patches=%0d", patches);
        $display("WD_STAT skip_zero_groups=%0d", SKIP_ZERO_GROUPS);
        $display("WD_STAT measured_cycle_patches=%0d", MEASURE_CYCLES ? patches : 0);
        $display("WD_STAT minimum_patch_cycles=%0d", minimum_cycles);
        $display("WD_STAT maximum_patch_cycles=%0d", maximum_cycles);
        $display("WD_STAT no_stall_patch_cycles=%0d", no_stall_cycles);
        $display("WD_STAT reset_scenarios=%0d", CASCADE ? 5 : 3);
        $display("WD_STAT reset_replay_records=%0d", replay_records);
        $display("WD_STAT readout_step_records=%0d", trace_records);
        $display("WD_STAT layer1_records=%0d", first_records);
        $display("WD_STAT layer2_records=%0d", second_records);
        $display("WD_STAT output_backpressure_cycles=%0d", backpressure_cycles);
        $display("WD_STAT input_gap_cycles=%0d", gap_cycles);
        $display("WD_STAT exact_tree_saturations=%0d", exact_tree_total);
        $display("WD_STAT approximate_tree_saturations=%0d", approx_tree_total);
        $display("WD_STAT exact_membrane_saturations=%0d", exact_mem_total);
        $display("WD_STAT approximate_membrane_saturations=%0d", approx_mem_total);
        if (CASCADE) begin
            $fdisplay(capture_descriptor, "WD_CAPTURE_END records=%0d", records);
            $fclose(capture_descriptor);
        end
        $display("WD_PASS top=%0s records=%0d", TEST_NAME, records);
        $finish;
    end
endmodule

module tb_readout_layer_engine;
    proposal_bench #(.CASCADE(0)) bench();
endmodule
module tb_proposal_engine #(
`ifdef WD_ZERO_SKIP
    parameter integer SKIP_ZERO_GROUPS = 1,
`else
    parameter integer SKIP_ZERO_GROUPS = 0,
`endif
`ifdef WD_MEASURE_CYCLES
    parameter integer MEASURE_CYCLES = 1
`else
    parameter integer MEASURE_CYCLES = 0
`endif
);
    proposal_bench #(.CASCADE(1), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS), .MEASURE_CYCLES(MEASURE_CYCLES)) bench();
endmodule

module tb_readout_rounding;
    logic signed [25:0] value, rounded;
    logic [31:0] expected, actual;
    integer descriptor, fields, count, records;
    string extra;
    signed_round_shift #(.WIDTH(26), .RIGHT_SHIFT(2)) dut (.value_in(value), .value_out(rounded));
    assign actual = {{8{rounded[23]}}, rounded[23:0]};
    initial begin
        descriptor = $fopen("expected_records.txt", "r");
        if (!descriptor) $fatal(1, "Missing rounding count");
        fields = $fscanf(descriptor, "%d", count);
        $fclose(descriptor);
        if (fields != 1 || count <= 0) $fatal(1, "Invalid rounding count");
        descriptor = $fopen("readout_rounding.txt", "r");
        if (!descriptor) $fatal(1, "Missing rounding oracle");
        for (records = 0; records < count; records = records + 1) begin
            fields = $fscanf(descriptor, "%h %h", value, expected);
            if (fields != 2 || $isunknown({value, expected})) $fatal(1, "Malformed rounding oracle");
            #1;
            if (actual !== expected) $fatal(1, "Readout rounding mismatch");
        end
        fields = $fscanf(descriptor, "%s", extra);
        if (fields == 1) $fatal(1, "Extra rounding oracle");
        $fclose(descriptor);
        $display("WD_PASS top=tb_readout_rounding records=%0d", records);
        $finish;
    end
endmodule
