`timescale 1ns/1ps

// AXI-Lite protocol smoke test for proposal_axi_lite.
// It writes two complete frozen proposal fixtures, starts one transaction at
// a time, and compares all sixteen exact logits with proposal_expected.txt.
module tb_proposal_axi_lite;
    localparam logic [11:0] CONTROL_ADDR = 12'h000;
    localparam logic [11:0] STATUS_ADDR = 12'h004;
    localparam logic [11:0] INPUT_BASE = 12'h100;
    localparam logic [11:0] OUTPUT_BASE = 12'h900;

    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic aresetn = 1'b0;
    logic [11:0] awaddr = 12'h0;
    logic awvalid = 1'b0;
    logic awready;
    logic [31:0] wdata = 32'h0;
    logic [3:0] wstrb = 4'h0;
    logic wvalid = 1'b0;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready = 1'b1;
    logic [11:0] araddr = 12'h0;
    logic arvalid = 1'b0;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready = 1'b0;

    logic [31:0] tokens [0:3];
    logic [31:0] expected_word;
    logic [31:0] ignored_approx;
    integer expected_tree, expected_mem, ignored_tree, ignored_mem;
    integer descriptor, fields, patch, word, byte_index, channel;
    integer status_reads;
    logic [31:0] status_word;
    logic [31:0] actual_word;

`ifdef WD_USE_AXI_LITE_REF
    proposal_axi_lite_ref #(.APPROX_LOW_BITS(0), .SKIP_ZERO_GROUPS(0)) dut (
`else
    proposal_axi_lite #(.APPROX_LOW_BITS(0), .SKIP_ZERO_GROUPS(0)) dut (
`endif
        .s_axi_aclk(clk), .s_axi_aresetn(aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready)
    );

    task automatic axi_write(input logic [11:0] address, input logic [31:0] value);
        begin
            @(negedge clk);
            awaddr <= address;
            wdata <= value;
            wstrb <= 4'hf;
            awvalid <= 1'b1;
            wvalid <= 1'b1;
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid <= 1'b0;
            wvalid <= 1'b0;
            while (!bvalid) @(posedge clk);
            @(negedge clk);
        end
    endtask

    task automatic axi_read(input logic [11:0] address, output logic [31:0] value);
        begin
            @(negedge clk);
            araddr <= address;
            arvalid <= 1'b1;
            while (!arready) @(posedge clk);
            @(negedge clk);
            arvalid <= 1'b0;
            while (!rvalid) @(posedge clk);
            value = rdata;
            @(negedge clk);
            rready <= 1'b1;
            @(negedge clk);
            rready <= 1'b0;
        end
    endtask

    task automatic write_patch(input integer target_patch);
        begin
            descriptor = $fopen("proposal_inputs.txt", "r");
            if (!descriptor) $fatal(1, "Missing proposal_inputs.txt");
            for (patch = 0; patch <= target_patch; patch = patch + 1) begin
                for (word = 0; word < 128; word = word + 1) begin
                    for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1) begin
                        fields = $fscanf(descriptor, "%h", tokens[byte_index]);
                        if (fields != 1) $fatal(1, "Malformed proposal input fixture");
                    end
                    if (patch == target_patch)
                        axi_write(INPUT_BASE + word * 4,
                            {tokens[3][7:0], tokens[2][7:0], tokens[1][7:0], tokens[0][7:0]});
                end
            end
            $fclose(descriptor);
        end
    endtask

    task automatic check_patch(input integer target_patch);
        begin
            descriptor = $fopen("proposal_expected.txt", "r");
            if (!descriptor) $fatal(1, "Missing proposal_expected.txt");
            for (patch = 0; patch <= target_patch; patch = patch + 1) begin
                for (channel = 0; channel < 16; channel = channel + 1) begin
                    fields = $fscanf(descriptor, "%h %d %d %h %d %d",
                        expected_word, expected_tree, expected_mem,
                        ignored_approx, ignored_tree, ignored_mem);
                    if (fields != 6) $fatal(1, "Malformed proposal expected fixture");
                    if (patch == target_patch) begin
                        axi_read(OUTPUT_BASE + channel * 4, actual_word);
                        if (actual_word !== expected_word)
                            $fatal(1, "AXI wrapper mismatch patch=%0d channel=%0d actual=%h expected=%h",
                                patch, channel, actual_word, expected_word);
                    end
                end
            end
            $fclose(descriptor);
        end
    endtask

    task automatic run_patch(input integer target_patch);
        begin
            write_patch(target_patch);
            axi_write(CONTROL_ADDR, 32'h00000001);
            status_reads = 0;
            do begin
                axi_read(STATUS_ADDR, status_word);
                status_reads = status_reads + 1;
                if (status_reads == 1 || status_reads == 100 || status_reads == 1000)
`ifdef WD_USE_AXI_LITE_REF
                    $display("AXI_STATUS patch=%0d reads=%0d value=%h state=%0d",
                        target_patch, status_reads, status_word, dut.core.state);
`else
                    $display("AXI_STATUS patch=%0d reads=%0d value=%h state=%0d",
                        target_patch, status_reads, status_word, dut.state);
`endif
                if (status_reads > 20000) $fatal(1, "AXI wrapper completion watchdog");
            end while (!status_word[1] || status_word[0]);
            if (status_word[2]) $fatal(1, "AXI wrapper error flag set");
            check_patch(target_patch);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        aresetn = 1'b1;
        repeat (2) @(posedge clk);
        run_patch(0);
        run_patch(1);
        $display("WD_PASS top=tb_proposal_axi_lite patches=2 logits=32");
        $finish;
    end
endmodule
