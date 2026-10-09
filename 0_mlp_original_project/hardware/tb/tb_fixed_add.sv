`timescale 1ns/1ps

module tb_fixed_add;
    logic signed [23:0] first;
    logic signed [23:0] second;
    logic signed [23:0] exact_result;
    logic signed [23:0] approximate_result;
    logic exact_saturated;
    logic approximate_saturated;
    logic [23:0] expected_exact;
    logic [23:0] expected_approximate;
    integer expected_exact_saturated;
    integer expected_approximate_saturated;
    integer descriptor;
    integer fields;
    integer records;
    integer errors;
    string vector_file;

    fixed_add #(.WIDTH(24), .APPROX_LOW_BITS(0)) exact_dut (
        .first(first),
        .second(second),
        .result(exact_result),
        .saturated(exact_saturated)
    );

    fixed_add #(.WIDTH(24), .APPROX_LOW_BITS(2)) approximate_dut (
        .first(first),
        .second(second),
        .result(approximate_result),
        .saturated(approximate_saturated)
    );

    initial begin
        records = 0;
        errors = 0;
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            vector_file = "fixed_add.txt";
        end
        descriptor = $fopen(vector_file, "r");
        if (descriptor == 0) begin
            $fatal(1, "Cannot open %s", vector_file);
        end
        while (!$feof(descriptor)) begin
            fields = $fscanf(
                descriptor,
                "%h %h %h %d %h %d\n",
                first,
                second,
                expected_exact,
                expected_exact_saturated,
                expected_approximate,
                expected_approximate_saturated
            );
            if (fields == 6) begin
                #1;
                records = records + 1;
                if (($unsigned(exact_result) !== expected_exact)
                        || (exact_saturated !== expected_exact_saturated[0])) begin
                    errors = errors + 1;
                    $display("Exact mismatch at record %0d", records);
                end
                if (($unsigned(approximate_result) !== expected_approximate)
                        || (approximate_saturated !== expected_approximate_saturated[0])) begin
                    errors = errors + 1;
                    $display("Approximate mismatch at record %0d", records);
                end
            end
        end
        $fclose(descriptor);
        if (errors != 0) begin
            $fatal(1, "fixed_add failed: %0d errors in %0d records", errors, records);
        end
        $display("fixed_add passed: %0d records", records);
        $finish;
    end
endmodule
