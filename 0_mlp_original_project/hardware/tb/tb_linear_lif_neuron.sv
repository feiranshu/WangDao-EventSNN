`timescale 1ns/1ps

module tb_linear_lif_neuron;
    localparam integer CHANNELS = 3;
    localparam integer INPUTS = 128;

    logic [7:0] features [0:INPUTS-1];
    logic signed [7:0] weights_f4 [0:INPUTS-1];
    logic signed [7:0] weights_f5 [0:INPUTS-1];
    logic signed [7:0] weights_f6 [0:INPUTS-1];
    logic signed [23:0] exact_membrane_in [0:CHANNELS-1];
    logic signed [23:0] approximate_membrane_in [0:CHANNELS-1];
    logic signed [23:0] exact_membrane_out [0:CHANNELS-1];
    logic signed [23:0] approximate_membrane_out [0:CHANNELS-1];
    logic exact_spike [0:CHANNELS-1];
    logic approximate_spike [0:CHANNELS-1];
    logic [7:0] exact_tree_saturations [0:CHANNELS-1];
    logic [7:0] approximate_tree_saturations [0:CHANNELS-1];
    logic exact_membrane_saturated [0:CHANNELS-1];
    logic approximate_membrane_saturated [0:CHANNELS-1];

    logic [23:0] expected_exact_membrane [0:CHANNELS-1];
    logic [23:0] expected_approximate_membrane [0:CHANNELS-1];
    integer expected_exact_spike [0:CHANNELS-1];
    integer expected_approximate_spike [0:CHANNELS-1];
    integer expected_exact_tree_saturations [0:CHANNELS-1];
    integer expected_approximate_tree_saturations [0:CHANNELS-1];
    integer expected_exact_membrane_saturated [0:CHANNELS-1];
    integer expected_approximate_membrane_saturated [0:CHANNELS-1];
    integer descriptor;
    integer fields;
    integer records;
    integer errors;
    integer input_index;
    integer channel;
    string vector_file;

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(4),
        .SYN_APPROX_LOW_BITS(0),
        .MEM_APPROX_LOW_BITS(0)
    ) exact_f4 (
        .features(features), .weights(weights_f4),
        .membrane_in(exact_membrane_in[0]), .fire_enable(1'b1),
        .membrane_out(exact_membrane_out[0]), .spike(exact_spike[0]),
        .tree_saturation_count(exact_tree_saturations[0]),
        .membrane_saturated(exact_membrane_saturated[0])
    );

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(4),
        .SYN_APPROX_LOW_BITS(2),
        .MEM_APPROX_LOW_BITS(2)
    ) approximate_f4 (
        .features(features), .weights(weights_f4),
        .membrane_in(approximate_membrane_in[0]), .fire_enable(1'b1),
        .membrane_out(approximate_membrane_out[0]), .spike(approximate_spike[0]),
        .tree_saturation_count(approximate_tree_saturations[0]),
        .membrane_saturated(approximate_membrane_saturated[0])
    );

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(5),
        .SYN_APPROX_LOW_BITS(0),
        .MEM_APPROX_LOW_BITS(0)
    ) exact_f5 (
        .features(features), .weights(weights_f5),
        .membrane_in(exact_membrane_in[1]), .fire_enable(1'b1),
        .membrane_out(exact_membrane_out[1]), .spike(exact_spike[1]),
        .tree_saturation_count(exact_tree_saturations[1]),
        .membrane_saturated(exact_membrane_saturated[1])
    );

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(5),
        .SYN_APPROX_LOW_BITS(2),
        .MEM_APPROX_LOW_BITS(2)
    ) approximate_f5 (
        .features(features), .weights(weights_f5),
        .membrane_in(approximate_membrane_in[1]), .fire_enable(1'b1),
        .membrane_out(approximate_membrane_out[1]), .spike(approximate_spike[1]),
        .tree_saturation_count(approximate_tree_saturations[1]),
        .membrane_saturated(approximate_membrane_saturated[1])
    );

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(6),
        .SYN_APPROX_LOW_BITS(0),
        .MEM_APPROX_LOW_BITS(0)
    ) exact_f6 (
        .features(features), .weights(weights_f6),
        .membrane_in(exact_membrane_in[2]), .fire_enable(1'b1),
        .membrane_out(exact_membrane_out[2]), .spike(exact_spike[2]),
        .tree_saturation_count(exact_tree_saturations[2]),
        .membrane_saturated(exact_membrane_saturated[2])
    );

    linear_lif_neuron #(
        .WEIGHT_FRACTION_BITS(6),
        .SYN_APPROX_LOW_BITS(2),
        .MEM_APPROX_LOW_BITS(2)
    ) approximate_f6 (
        .features(features), .weights(weights_f6),
        .membrane_in(approximate_membrane_in[2]), .fire_enable(1'b1),
        .membrane_out(approximate_membrane_out[2]), .spike(approximate_spike[2]),
        .tree_saturation_count(approximate_tree_saturations[2]),
        .membrane_saturated(approximate_membrane_saturated[2])
    );

    initial begin
        $readmemh("layer1_weight_f4.txt", weights_f4);
        $readmemh("layer1_weight_f5.txt", weights_f5);
        $readmemh("layer1_weight_f6.txt", weights_f6);
        records = 0;
        errors = 0;
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            vector_file = "linear_lif_neuron.txt";
        end
        descriptor = $fopen(vector_file, "r");
        if (descriptor == 0) begin
            $fatal(1, "Cannot open %s", vector_file);
        end
        while (!$feof(descriptor)) begin
            fields = 0;
            for (input_index = 0; input_index < INPUTS; input_index = input_index + 1) begin
                fields = fields + $fscanf(descriptor, "%h", features[input_index]);
            end
            for (channel = 0; channel < CHANNELS; channel = channel + 1) begin
                fields = fields + $fscanf(
                    descriptor,
                    "%h %h %h %d %d %d %h %d %d %d",
                    exact_membrane_in[channel], approximate_membrane_in[channel],
                    expected_exact_membrane[channel], expected_exact_spike[channel],
                    expected_exact_tree_saturations[channel],
                    expected_exact_membrane_saturated[channel],
                    expected_approximate_membrane[channel],
                    expected_approximate_spike[channel],
                    expected_approximate_tree_saturations[channel],
                    expected_approximate_membrane_saturated[channel]
                );
            end
            if (fields == INPUTS + CHANNELS * 10) begin
                #1;
                records = records + 1;
                for (channel = 0; channel < CHANNELS; channel = channel + 1) begin
                    if (($unsigned(exact_membrane_out[channel]) !== expected_exact_membrane[channel])
                            || (exact_spike[channel] !== expected_exact_spike[channel][0])
                            || (exact_tree_saturations[channel] !== expected_exact_tree_saturations[channel][7:0])
                            || (exact_membrane_saturated[channel] !== expected_exact_membrane_saturated[channel][0])) begin
                        errors = errors + 1;
                        $display("Exact linear/LIF mismatch at record %0d channel %0d", records, channel);
                    end
                    if (($unsigned(approximate_membrane_out[channel]) !== expected_approximate_membrane[channel])
                            || (approximate_spike[channel] !== expected_approximate_spike[channel][0])
                            || (approximate_tree_saturations[channel] !== expected_approximate_tree_saturations[channel][7:0])
                            || (approximate_membrane_saturated[channel] !== expected_approximate_membrane_saturated[channel][0])) begin
                        errors = errors + 1;
                        $display("Approximate linear/LIF mismatch at record %0d channel %0d", records, channel);
                    end
                end
            end
        end
        $fclose(descriptor);
        if (errors != 0) begin
            $fatal(1, "linear_lif_neuron failed: %0d errors in %0d records", errors, records);
        end
        $display("linear_lif_neuron passed: %0d records", records);
        $finish;
    end
endmodule
