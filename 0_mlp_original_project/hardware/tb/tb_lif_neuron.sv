`timescale 1ns/1ps

module tb_lif_neuron;
    logic signed [23:0] membrane_in;
    logic signed [23:0] current_in;
    logic fire_enable;
    logic signed [23:0] exact_membrane;
    logic signed [23:0] approximate_membrane;
    logic exact_spike;
    logic approximate_spike;
    logic exact_saturated;
    logic approximate_saturated;
    logic [23:0] expected_exact_membrane;
    logic [23:0] expected_approximate_membrane;
    integer expected_exact_spike;
    integer expected_approximate_spike;
    integer expected_exact_saturated;
    integer expected_approximate_saturated;
    integer descriptor;
    integer fields;
    integer records;
    integer errors;
    string vector_file;

    lif_neuron #(.APPROX_LOW_BITS(0)) exact_dut (
        .membrane_in(membrane_in),
        .current_in(current_in),
        .fire_enable(fire_enable),
        .membrane_out(exact_membrane),
        .spike(exact_spike),
        .saturated(exact_saturated)
    );

    lif_neuron #(.APPROX_LOW_BITS(2)) approximate_dut (
        .membrane_in(membrane_in),
        .current_in(current_in),
        .fire_enable(fire_enable),
        .membrane_out(approximate_membrane),
        .spike(approximate_spike),
        .saturated(approximate_saturated)
    );

    initial begin
        records = 0;
        errors = 0;
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            vector_file = "lif_neuron.txt";
        end
        descriptor = $fopen(vector_file, "r");
        if (descriptor == 0) begin
            $fatal(1, "Cannot open %s", vector_file);
        end
        while (!$feof(descriptor)) begin
            fields = $fscanf(
                descriptor,
                "%h %h %d %h %d %d %h %d %d\n",
                membrane_in, current_in, fire_enable,
                expected_exact_membrane, expected_exact_spike,
                expected_exact_saturated, expected_approximate_membrane,
                expected_approximate_spike, expected_approximate_saturated
            );
            if (fields == 9) begin
                #1;
                records = records + 1;
                if (($unsigned(exact_membrane) !== expected_exact_membrane)
                        || (exact_spike !== expected_exact_spike[0])
                        || (exact_saturated !== expected_exact_saturated[0])) begin
                    errors = errors + 1;
                    $display("Exact LIF mismatch at record %0d", records);
                end
                if (($unsigned(approximate_membrane) !== expected_approximate_membrane)
                        || (approximate_spike !== expected_approximate_spike[0])
                        || (approximate_saturated !== expected_approximate_saturated[0])) begin
                    errors = errors + 1;
                    $display("Approximate LIF mismatch at record %0d", records);
                end
            end
        end
        $fclose(descriptor);
        if (errors != 0) begin
            $fatal(1, "lif_neuron failed: %0d errors in %0d records", errors, records);
        end
        $display("lif_neuron passed: %0d records", records);
        $finish;
    end
endmodule
