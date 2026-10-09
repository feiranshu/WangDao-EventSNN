`timescale 1ns/1ps

// Balanced fixed-point reduction tree.
// The level array preserves the former recursive module's pair order while
// keeping the hierarchy non-recursive so Vivado Module Reference can use the
// enclosing Proposal wrapper in a Block Design.
module fixed_add_tree #(
    parameter integer WIDTH = 24,
    parameter integer INPUTS = 5,
    parameter integer APPROX_LOW_BITS = 0
) (
    input logic signed [WIDTH-1:0] values [0:INPUTS-1],
    output logic signed [WIDTH-1:0] result,
    output logic [$clog2(INPUTS+1)-1:0] saturation_count
);
    localparam integer SAT_WIDTH = $clog2(INPUTS + 1);
    localparam integer TREE_LEVELS = (INPUTS > 1) ? $clog2(INPUTS) : 0;

    initial begin
        if (INPUTS < 1)
            $error("INPUTS must be positive");
    end

    generate
        if (INPUTS == 1) begin : single_input
            always_comb begin
                result = values[0];
                saturation_count = '0;
            end
        end else begin : multi_input
            logic signed [WIDTH-1:0] level_values [0:TREE_LEVELS][0:INPUTS-1];
            logic [SAT_WIDTH-1:0] level_saturations [0:TREE_LEVELS][0:INPUTS-1];

            genvar input_index;
            for (input_index = 0; input_index < INPUTS; input_index = input_index + 1) begin : input_nodes
                assign level_values[0][input_index] = values[input_index];
                assign level_saturations[0][input_index] = '0;
            end

            genvar level;
            for (level = 0; level < TREE_LEVELS; level = level + 1) begin : reduction_levels
                localparam integer LEVEL_INPUTS = (INPUTS + (1 << level) - 1) >> level;
                localparam integer NEXT_INPUTS = (LEVEL_INPUTS + 1) >> 1;

                genvar pair;
                for (pair = 0; pair < NEXT_INPUTS; pair = pair + 1) begin : reduction_nodes
                    if ((pair * 2 + 1) < LEVEL_INPUTS) begin : add_pair
                        logic signed [WIDTH-1:0] pair_result;
                        logic pair_saturated;

                        fixed_add #(
                            .WIDTH(WIDTH),
                            .APPROX_LOW_BITS(APPROX_LOW_BITS)
                        ) add_pair_inst (
                            .first(level_values[level][pair * 2]),
                            .second(level_values[level][pair * 2 + 1]),
                            .result(pair_result),
                            .saturated(pair_saturated)
                        );

                        assign level_values[level + 1][pair] = pair_result;
                        assign level_saturations[level + 1][pair] =
                            level_saturations[level][pair * 2] +
                            level_saturations[level][pair * 2 + 1] +
                            pair_saturated;
                    end else begin : carry_tail
                        assign level_values[level + 1][pair] =
                            level_values[level][pair * 2];
                        assign level_saturations[level + 1][pair] =
                            level_saturations[level][pair * 2];
                    end
                end
            end

            assign result = level_values[TREE_LEVELS][0];
            assign saturation_count = level_saturations[TREE_LEVELS][0];
        end
    endgenerate
endmodule
