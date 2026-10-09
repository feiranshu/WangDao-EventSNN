`timescale 1ns/1ps

module tb_first_layer_engine #(
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
    localparam integer MAX_PATCHES = 346;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1;
    logic [7:0] s_data = 0;
    logic s_valid = 0;
    logic exact_ready, approx_ready;
    logic [31:0] exact_data, approx_data;
    logic exact_valid, approx_valid;
    logic m_ready = 0;
    logic [7:0] exact_tree, approx_tree;
    logic exact_mem_sat, approx_mem_sat;
    logic [7:0] features [0:MAX_PATCHES*512-1];
    logic [31:0] expected_exact [0:MAX_PATCHES*192-1];
    logic [31:0] expected_approx [0:MAX_PATCHES*192-1];
    integer expected_exact_tree [0:MAX_PATCHES*192-1];
    integer expected_approx_tree [0:MAX_PATCHES*192-1];
    integer expected_exact_mem [0:MAX_PATCHES*192-1];
    integer expected_approx_mem [0:MAX_PATCHES*192-1];
    integer descriptor, fields, expected_records, patches, row, patch_index;
    integer received = 0;
    integer records = 0;
    integer replay_records = 0;
    integer expected_base = 0;
    integer cycles = 0;
    integer patch_start, patch_cycles;
    integer minimum_cycles = 2147483647;
    integer maximum_cycles = 0;
    integer no_stall_cycles = 0;
    integer stalled_output_cycles = 0;
    integer input_gap_cycles = 0;
    integer exact_tree_total = 0, approx_tree_total = 0;
    integer exact_mem_total = 0, approx_mem_total = 0;
    bit checking = 0;
    bit regular_record = 0;
    bit stress = 0;
    bit force_stall = 0;
    bit was_stalled = 0;
    logic [81:0] held_output;
    string extra_token;

    first_layer_engine #(.APPROX_LOW_BITS(0), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)) exact_dut (
        .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(exact_ready),
        .m_data(exact_data), .m_valid(exact_valid), .m_ready(m_ready),
        .m_tree_saturations(exact_tree), .m_membrane_saturated(exact_mem_sat)
    );
    first_layer_engine #(.APPROX_LOW_BITS(2), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)) approximate_dut (
        .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(approx_ready),
        .m_data(approx_data), .m_valid(approx_valid), .m_ready(m_ready),
        .m_tree_saturations(approx_tree), .m_membrane_saturated(approx_mem_sat)
    );

    always @(negedge clk)
        m_ready = !force_stall && (!stress || ((cycles % 7) >= 3));

    always @(posedge clk) begin
        cycles = cycles + 1;
        if (cycles > (MAX_PATCHES + 10) * 20000) $fatal(1, "Engine cycle watchdog");
        if (rst) begin
            received = 0;
            was_stalled = 0;
        end else begin
            if ((exact_ready !== approx_ready) || (exact_valid !== approx_valid))
                $fatal(1, "Exact/approx handshake differs");
            if (was_stalled) begin
                if (!exact_valid || !approx_valid ||
                    ({exact_data, exact_tree, exact_mem_sat, approx_data, approx_tree, approx_mem_sat} !== held_output))
                    $fatal(1, "Output changed under backpressure");
            end
            was_stalled = exact_valid && !m_ready;
            held_output = {exact_data, exact_tree, exact_mem_sat, approx_data, approx_tree, approx_mem_sat};
            if (checking && exact_valid && !m_ready)
                stalled_output_cycles = stalled_output_cycles + 1;
            if (checking && exact_ready && !s_valid)
                input_gap_cycles = input_gap_cycles + 1;
            if (exact_valid && m_ready) begin
                if (checking) begin
                    if (received >= 192) $fatal(1, "Extra output word");
                    if ((exact_data !== expected_exact[expected_base + received]) ||
                        (approx_data !== expected_approx[expected_base + received]) ||
                        (exact_tree !== expected_exact_tree[expected_base + received][7:0]) ||
                        (approx_tree !== expected_approx_tree[expected_base + received][7:0]) ||
                        (exact_mem_sat !== expected_exact_mem[expected_base + received][0]) ||
                        (approx_mem_sat !== expected_approx_mem[expected_base + received][0]))
                        $fatal(1, "Engine mismatch at record %0d: exact=%h approximate=%h",
                            expected_base + received, exact_data, approx_data);
                    if (regular_record) begin
                        records = records + 1;
                        exact_tree_total = exact_tree_total + exact_tree;
                        approx_tree_total = approx_tree_total + approx_tree;
                        exact_mem_total = exact_mem_total + exact_mem_sat;
                        approx_mem_total = approx_mem_total + approx_mem_sat;
                    end else replay_records = replay_records + 1;
                end
                received = received + 1;
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
            if (exact_valid || approx_valid) $fatal(1, "Output valid during reset");
            rst = 0;
            force_stall = 0;
            @(negedge clk);
            if (!exact_ready || !approx_ready || exact_valid || approx_valid)
                $fatal(1, "Reset did not return to empty input state");
        end
    endtask

    task automatic send_bytes(input integer fixture, input integer count, input bit gaps);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                if (gaps && ((index % 11) == 4)) begin
                    s_valid = 0;
                    repeat (2) @(negedge clk);
                end
                s_data = features[fixture * 512 + index];
                s_valid = 1;
                @(posedge clk);
                while (!exact_ready) @(posedge clk);
            end
            @(negedge clk);
            s_valid = 0;
        end
    endtask

    task automatic run_patch(input integer fixture, input bit gaps, input bit count_record);
        begin
            @(negedge clk);
            expected_base = fixture * 192;
            received = 0;
            regular_record = count_record;
            checking = 1;
            stress = gaps;
            patch_start = cycles;
            send_bytes(fixture, 512, gaps);
            wait (received == 192);
            @(negedge clk);
            patch_cycles = cycles - patch_start;
            if (count_record) begin
                if (patch_cycles < minimum_cycles) minimum_cycles = patch_cycles;
                if (patch_cycles > maximum_cycles) maximum_cycles = patch_cycles;
                if (!gaps) no_stall_cycles = patch_cycles;
                if (MEASURE_CYCLES) $display("WD_CYCLE patch=%0d cycles=%0d", fixture, patch_cycles);
            end
            if (!exact_ready || exact_valid || approx_valid) $fatal(1, "Patch did not finish cleanly");
            // Remain armed between patches so stray or repeated outputs fail.
        end
    endtask

    initial begin
        descriptor = $fopen("expected_records.txt", "r");
        if (!descriptor) $fatal(1, "Missing expected record count");
        fields = $fscanf(descriptor, "%d", expected_records);
        $fclose(descriptor);
        if ((fields != 1) || (expected_records <= 0) || (expected_records % 192 != 0))
            $fatal(1, "Invalid engine record count");
        patches = expected_records / 192;
        if ((patches < 2) || (patches > MAX_PATCHES)) $fatal(1, "Invalid patch count");
        descriptor = $fopen("first_layer_inputs.txt", "r");
        if (!descriptor) $fatal(1, "Missing engine inputs");
        for (row = 0; row < patches * 512; row = row + 1) begin
            fields = $fscanf(descriptor, "%h", features[row]);
            if (fields != 1 || $isunknown(features[row])) $fatal(1, "Malformed/truncated engine input");
        end
        fields = $fscanf(descriptor, "%s", extra_token);
        if (fields == 1) $fatal(1, "Extra engine input");
        $fclose(descriptor);
        descriptor = $fopen("first_layer_expected.txt", "r");
        if (!descriptor) $fatal(1, "Missing engine expected output");
        for (row = 0; row < expected_records; row = row + 1) begin
            fields = $fscanf(descriptor, "%h %d %d %h %d %d",
                expected_exact[row], expected_exact_tree[row], expected_exact_mem[row],
                expected_approx[row], expected_approx_tree[row], expected_approx_mem[row]);
            if (fields != 6) $fatal(1, "Malformed/truncated engine expectation");
        end
        fields = $fscanf(descriptor, "%s", extra_token);
        if (fields == 1) $fatal(1, "Extra engine expectation");
        $fclose(descriptor);

        reset_engine();
        send_bytes(1, 100, 0);
        reset_engine();
        run_patch(1, 1, 0);
        // Abort after earlier outputs have changed membrane state, during computation.
        checking = 0;
        send_bytes(1, 512, 0);
        repeat (100) @(negedge clk);
        reset_engine();
        run_patch(1, 0, 0);
        checking = 0;
        force_stall = 1;
        send_bytes(1, 512, 0);
        wait (exact_valid);
        repeat (5) @(negedge clk);
        reset_engine();
        run_patch(1, 1, 0);

        if (SKIP_ZERO_GROUPS) begin
            // Abort inside a partly loaded group, then overwrite every flag with zero input.
            checking = 0;
            send_bytes(1, 511, 1);
            reset_engine();
            run_patch(0, 1, 0);
            checking = 0;
            send_bytes(0, 512, 0);
            wait (exact_dut.state == 1 && exact_dut.group_index == 15);
            reset_engine();
            run_patch(1, 0, 0);
            // No reset between dense and empty patches.
            run_patch(0, 0, 0);
        end
        for (patch_index = 0; patch_index < patches; patch_index = patch_index + 1)
            run_patch(patch_index, !MEASURE_CYCLES && (patch_index % 2) != 0, 1);
        repeat (20) @(negedge clk);
        if (exact_valid || approx_valid || received != 192) $fatal(1, "Extra output after last patch");
        if ((records != expected_records) || (replay_records != (SKIP_ZERO_GROUPS ? 1152 : 576)) ||
            (stalled_output_cycles == 0) || (input_gap_cycles == 0))
            $fatal(1, "Incomplete engine coverage");
        $display("WD_STAT patches=%0d", patches);
        $display("WD_STAT skip_zero_groups=%0d", SKIP_ZERO_GROUPS);
        $display("WD_STAT measured_cycle_patches=%0d", MEASURE_CYCLES ? patches : 0);
        $display("WD_STAT minimum_patch_cycles=%0d", minimum_cycles);
        $display("WD_STAT maximum_patch_cycles=%0d", maximum_cycles);
        $display("WD_STAT no_stall_patch_cycles=%0d", no_stall_cycles);
        $display("WD_STAT reset_scenarios=%0d", SKIP_ZERO_GROUPS ? 5 : 3);
        $display("WD_STAT reset_replay_records=%0d", replay_records);
        $display("WD_STAT output_backpressure_cycles=%0d", stalled_output_cycles);
        $display("WD_STAT input_gap_cycles=%0d", input_gap_cycles);
        $display("WD_STAT exact_tree_saturations=%0d", exact_tree_total);
        $display("WD_STAT approximate_tree_saturations=%0d", approx_tree_total);
        $display("WD_STAT exact_membrane_saturations=%0d", exact_mem_total);
        $display("WD_STAT approximate_membrane_saturations=%0d", approx_mem_total);
        $display("WD_PASS top=tb_first_layer_engine records=%0d", records);
        $finish;
    end
endmodule
