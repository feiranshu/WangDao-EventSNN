`timescale 1ns/1ps

module tb_fixed_add_tree;
    logic signed [23:0] values [0:4];
    logic signed [23:0] exact_result;
    logic signed [23:0] approximate_result;
    logic [2:0] exact_saturations;
    logic [2:0] approximate_saturations;
    logic [23:0] expected_exact;
    logic [23:0] expected_approximate;
    integer expected_exact_saturations;
    integer expected_approximate_saturations;
    integer descriptor;
    integer fields;
    integer records;
    integer errors;
    string vector_file;
    string extra_token;
    integer expected_records;
    integer count_file;
    integer record_index;

    fixed_add_tree #(
        .WIDTH(24), .INPUTS(5), .APPROX_LOW_BITS(0)
    ) exact_dut (
        .values(values),
        .result(exact_result),
        .saturation_count(exact_saturations)
    );

    fixed_add_tree #(
        .WIDTH(24), .INPUTS(5), .APPROX_LOW_BITS(2)
    ) approximate_dut (
        .values(values),
        .result(approximate_result),
        .saturation_count(approximate_saturations)
    );

    initial begin
        count_file = $fopen("expected_records.txt", "r");
        if (count_file == 0) $fatal(1, "Missing expected record count");
        fields = $fscanf(count_file, "%d", expected_records);
        $fclose(count_file);
        if ((fields != 1) || (expected_records <= 0))
            $fatal(1, "Invalid expected record count");
        records = 0;
        errors = 0;
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            vector_file = "fixed_add_tree.txt";
        end
        descriptor = $fopen(vector_file, "r");
        if (descriptor == 0) begin
            $fatal(1, "Cannot open %s", vector_file);
        end
        for (record_index = 0; record_index < expected_records; record_index = record_index + 1) begin
            fields = $fscanf(
                descriptor,
                "%h %h %h %h %h %h %d %h %d\n",
                values[0], values[1], values[2], values[3], values[4],
                expected_exact, expected_exact_saturations,
                expected_approximate, expected_approximate_saturations
            );
            if (fields != 9)
                $fatal(1, "Malformed/truncated vector at record %0d", record_index);
            begin
                #1;
                records = records + 1;
                if (($unsigned(exact_result) !== expected_exact)
                        || (exact_saturations !== expected_exact_saturations[2:0])) begin
                    errors = errors + 1;
                    $display("Exact tree mismatch at record %0d", records);
                end
                if (($unsigned(approximate_result) !== expected_approximate)
                        || (approximate_saturations !== expected_approximate_saturations[2:0])) begin
                    errors = errors + 1;
                    $display("Approximate tree mismatch at record %0d", records);
                end
            end
        end
        fields = $fscanf(descriptor, "%s", extra_token);
        if (fields == 1) $fatal(1, "Unexpected extra vector data");
        $fclose(descriptor);
        if (records != expected_records) $fatal(1, "Record count mismatch");
        if (errors != 0) begin
            $fatal(1, "fixed_add_tree failed: %0d errors in %0d records", errors, records);
        end
        $display("WD_PASS top=tb_fixed_add_tree records=%0d", records);
        $finish;
    end
endmodule
