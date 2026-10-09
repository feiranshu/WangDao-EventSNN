`timescale 1ns/1ps

module hidden_layers_bench #(parameter integer CASCADE = 0);
    localparam integer INPUT_WORDS = CASCADE ? 512 : 192;
    localparam TEST_NAME = CASCADE ? "tb_two_layer_engine" : "tb_second_layer_engine";
    localparam INPUT_FILE = CASCADE ? "two_layer_inputs.txt" : "second_layer_inputs.txt";
    localparam EXPECTED_FILE = CASCADE ? "two_layer_expected.txt" : "second_layer_expected.txt";
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
    logic [7:0] features [0:MAX_PATCHES*INPUT_WORDS-1];
    logic [31:0] expected_exact [0:MAX_PATCHES*96-1];
    logic [31:0] expected_approx [0:MAX_PATCHES*96-1];
    integer expected_exact_tree [0:MAX_PATCHES*96-1];
    integer expected_approx_tree [0:MAX_PATCHES*96-1];
    integer expected_exact_mem [0:MAX_PATCHES*96-1];
    integer expected_approx_mem [0:MAX_PATCHES*96-1];
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

    logic link_valid, link_ready;
    logic [31:0] link_exact, link_approx;
    logic [7:0] link_exact_tree, link_approx_tree;
    logic link_exact_mem, link_approx_mem;
    integer linked = 0, intermediate_records = 0;
    integer accepted = 0;
    logic [31:0] intermediate_exact [0:MAX_PATCHES*192-1];
    logic [31:0] intermediate_approx [0:MAX_PATCHES*192-1];
    integer intermediate_exact_tree [0:MAX_PATCHES*192-1];
    integer intermediate_approx_tree [0:MAX_PATCHES*192-1];
    integer intermediate_exact_mem [0:MAX_PATCHES*192-1];
    integer intermediate_approx_mem [0:MAX_PATCHES*192-1];
    integer intermediate_base;
    generate if (CASCADE) begin : cascade
    two_layer_engine #(.APPROX_LOW_BITS(0)) exact_dut (
        .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(exact_ready),
        .m_data(exact_data), .m_valid(exact_valid), .m_ready(m_ready),
        .m_tree_saturations(exact_tree), .m_membrane_saturated(exact_mem_sat)
    );
    two_layer_engine #(.APPROX_LOW_BITS(2)) approximate_dut (
        .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(approx_ready),
        .m_data(approx_data), .m_valid(approx_valid), .m_ready(m_ready),
        .m_tree_saturations(approx_tree), .m_membrane_saturated(approx_mem_sat)
    );

        assign link_valid = exact_dut.link_valid;
        assign link_ready = exact_dut.link_ready;
        assign link_exact = exact_dut.link_data;
        assign link_approx = approximate_dut.link_data;
        assign link_exact_tree = exact_dut.link_tree_saturations;
        assign link_approx_tree = approximate_dut.link_tree_saturations;
        assign link_exact_mem = exact_dut.link_membrane_saturated;
        assign link_approx_mem = approximate_dut.link_membrane_saturated;
        always @(posedge clk) if (!rst &&
            ((exact_dut.link_valid !== approximate_dut.link_valid) ||
             (exact_dut.link_ready !== approximate_dut.link_ready)))
            $fatal(1, "Layer-link handshake differs between modes");
    end else begin : standalone
    second_layer_engine #(.APPROX_LOW_BITS(0)) exact_dut (
        .clk(clk), .rst(rst), .s_data(s_data[0]), .s_valid(s_valid), .s_ready(exact_ready),
        .m_data(exact_data), .m_valid(exact_valid), .m_ready(m_ready),
        .m_tree_saturations(exact_tree), .m_membrane_saturated(exact_mem_sat)
    );
    second_layer_engine #(.APPROX_LOW_BITS(2)) approximate_dut (
        .clk(clk), .rst(rst), .s_data(s_data[0]), .s_valid(s_valid), .s_ready(approx_ready),
        .m_data(approx_data), .m_valid(approx_valid), .m_ready(m_ready),
        .m_tree_saturations(approx_tree), .m_membrane_saturated(approx_mem_sat)
    );

        assign link_valid = 0;
        assign link_ready = 0;
    end endgenerate

    always @(negedge clk)
        m_ready = !force_stall && (!stress || ((cycles % 7) >= 3));

    always @(posedge clk) begin
        cycles = cycles + 1;
        if (cycles > (MAX_PATCHES + 10) * 20000) $fatal(1, "Engine cycle watchdog");
        if (rst) begin
            received = 0;
            linked = 0;
            accepted = 0;
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
            if (checking) begin
                if (accepted == INPUT_WORDS && received < 96 && exact_ready)
                    $fatal(1, "Input reopened before the patch completed");
                if (s_valid && exact_ready) accepted = accepted + 1;
            end
            if (checking && link_valid && link_ready) begin
                intermediate_base = (expected_base / 96) * 192 + linked;
                if (linked >= 192) $fatal(1, "Extra layer-one output");
                if ((link_exact !== intermediate_exact[intermediate_base]) ||
                    (link_approx !== intermediate_approx[intermediate_base]) ||
                    (link_exact_tree !== intermediate_exact_tree[intermediate_base][7:0]) ||
                    (link_approx_tree !== intermediate_approx_tree[intermediate_base][7:0]) ||
                    (link_exact_mem !== intermediate_exact_mem[intermediate_base][0]) ||
                    (link_approx_mem !== intermediate_approx_mem[intermediate_base][0]))
                    $fatal(1, "Layer-one intermediate mismatch at record %0d", intermediate_base);
                linked = linked + 1;
                if (regular_record) intermediate_records = intermediate_records + 1;
            end
            if (exact_valid && m_ready) begin
                if (checking) begin
                    if (received >= 96) $fatal(1, "Extra output word");
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
                s_data = features[fixture * INPUT_WORDS + index];
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
            expected_base = fixture * 96;
            received = 0;
            linked = 0;
            accepted = 0;
            regular_record = count_record;
            checking = 1;
            stress = gaps;
            patch_start = cycles;
            send_bytes(fixture, INPUT_WORDS, gaps);
            wait (received == 96);
            @(negedge clk);
            patch_cycles = cycles - patch_start;
            if (count_record) begin
                if (patch_cycles < minimum_cycles) minimum_cycles = patch_cycles;
                if (patch_cycles > maximum_cycles) maximum_cycles = patch_cycles;
                if (!gaps) no_stall_cycles = patch_cycles;
            end
            if (!exact_ready || exact_valid || approx_valid) $fatal(1, "Patch did not finish cleanly");
            if (accepted != INPUT_WORDS) $fatal(1, "Incorrect input transfer count");
            if (CASCADE && linked != 192) $fatal(1, "Missing layer-one outputs");
            checking = 0;
        end
    endtask

    initial begin
        descriptor = $fopen("expected_records.txt", "r");
        if (!descriptor) $fatal(1, "Missing expected record count");
        fields = $fscanf(descriptor, "%d", expected_records);
        $fclose(descriptor);
        if ((fields != 1) || (expected_records <= 0) || (expected_records % 96 != 0))
            $fatal(1, "Invalid engine record count");
        patches = expected_records / 96;
        if ((patches < 2) || (patches > MAX_PATCHES)) $fatal(1, "Invalid patch count");
        descriptor = $fopen(INPUT_FILE, "r");
        if (!descriptor) $fatal(1, "Missing engine inputs");
        for (row = 0; row < patches * INPUT_WORDS; row = row + 1) begin
            fields = $fscanf(descriptor, "%h", features[row]);
            if ((fields != 1) || $isunknown(features[row]) || (!CASCADE && features[row] > 1))
                $fatal(1, "Malformed/truncated engine input");
        end
        fields = $fscanf(descriptor, "%s", extra_token);
        if (fields == 1) $fatal(1, "Extra engine input");
        $fclose(descriptor);
        descriptor = $fopen(EXPECTED_FILE, "r");
        if (!descriptor) $fatal(1, "Missing engine expected output");
        for (row = 0; row < expected_records; row = row + 1) begin
            fields = $fscanf(descriptor, "%h %d %d %h %d %d",
                expected_exact[row], expected_exact_tree[row], expected_exact_mem[row],
                expected_approx[row], expected_approx_tree[row], expected_approx_mem[row]);
            if ((fields != 6) || $isunknown({expected_exact[row], expected_approx[row],
                expected_exact_tree[row], expected_approx_tree[row], expected_exact_mem[row], expected_approx_mem[row]}))
                $fatal(1, "Malformed/truncated engine expectation");
        end
        fields = $fscanf(descriptor, "%s", extra_token);
        if (fields == 1) $fatal(1, "Extra engine expectation");
        $fclose(descriptor);

        if (CASCADE) begin
            descriptor = $fopen("two_layer_intermediate.txt", "r");
            if (!descriptor) $fatal(1, "Missing intermediate oracle");
            for (row = 0; row < patches * 192; row = row + 1) begin
                fields = $fscanf(descriptor, "%h %d %d %h %d %d",
                    intermediate_exact[row], intermediate_exact_tree[row], intermediate_exact_mem[row],
                    intermediate_approx[row], intermediate_approx_tree[row], intermediate_approx_mem[row]);
                if ((fields != 6) || $isunknown({intermediate_exact[row], intermediate_approx[row],
                    intermediate_exact_tree[row], intermediate_approx_tree[row], intermediate_exact_mem[row], intermediate_approx_mem[row]}))
                    $fatal(1, "Malformed/truncated intermediate oracle");
            end
            fields = $fscanf(descriptor, "%s", extra_token);
            if (fields == 1) $fatal(1, "Extra intermediate expectation");
            $fclose(descriptor);
        end

        reset_engine();
        send_bytes(1, 100, 0);
        reset_engine();
        run_patch(1, 1, 0);
        // Abort after earlier outputs have changed membrane state, during computation.
        send_bytes(1, INPUT_WORDS, 0);
        repeat (100) @(negedge clk);
        reset_engine();
        run_patch(1, 0, 0);
        if (CASCADE) begin
            // Abort after the complete inter-layer transfer, during layer-two compute.
            send_bytes(1, INPUT_WORDS, 0);
            wait (!link_ready);
            repeat (100) @(negedge clk);
            reset_engine();
            run_patch(1, 0, 0);
        end
        force_stall = 1;
        send_bytes(1, INPUT_WORDS, 0);
        wait (exact_valid);
        @(negedge clk);
        s_valid = 1;
        s_data = 8'hAA;
        repeat (5) begin
            @(negedge clk);
            if (exact_ready || approx_ready) $fatal(1, "Accepted another patch under output backpressure");
        end
        reset_engine();
        run_patch(1, 1, 0);

        for (patch_index = 0; patch_index < patches; patch_index = patch_index + 1)
            run_patch(patch_index, (patch_index % 2) != 0, 1);
        repeat (20) @(negedge clk);
        if (exact_valid || approx_valid || received != 96) $fatal(1, "Extra output after last patch");
        if ((records != expected_records) || (replay_records != (CASCADE ? 384 : 288)) ||
            (stalled_output_cycles == 0) || (input_gap_cycles == 0))
            $fatal(1, "Incomplete engine coverage");
        $display("WD_STAT patches=%0d", patches);
        $display("WD_STAT minimum_patch_cycles=%0d", minimum_cycles);
        $display("WD_STAT maximum_patch_cycles=%0d", maximum_cycles);
        $display("WD_STAT no_stall_patch_cycles=%0d", no_stall_cycles);
        $display("WD_STAT reset_scenarios=%0d", CASCADE ? 4 : 3);
        $display("WD_STAT intermediate_records=%0d", intermediate_records);
        if (CASCADE && intermediate_records != patches * 192) $fatal(1, "Incomplete link coverage");
        $display("WD_STAT reset_replay_records=%0d", replay_records);
        $display("WD_STAT output_backpressure_cycles=%0d", stalled_output_cycles);
        $display("WD_STAT input_gap_cycles=%0d", input_gap_cycles);
        $display("WD_STAT exact_tree_saturations=%0d", exact_tree_total);
        $display("WD_STAT approximate_tree_saturations=%0d", approx_tree_total);
        $display("WD_STAT exact_membrane_saturations=%0d", exact_mem_total);
        $display("WD_STAT approximate_membrane_saturations=%0d", approx_mem_total);
        $display("WD_PASS top=%0s records=%0d", TEST_NAME, records);
        $finish;
    end
endmodule

module tb_second_layer_engine;
    hidden_layers_bench #(.CASCADE(0)) bench();
endmodule

module tb_two_layer_engine;
    hidden_layers_bench #(.CASCADE(1)) bench();
endmodule
