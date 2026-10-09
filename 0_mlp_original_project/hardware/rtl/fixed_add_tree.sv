`timescale 1ns/1ps

module fixed_add_tree #(
    parameter integer WIDTH = 24,
    parameter integer INPUTS = 5,
    parameter integer APPROX_LOW_BITS = 0
) (
    input  logic signed [WIDTH-1:0] values [0:INPUTS-1],
    output logic signed [WIDTH-1:0] result,
    output logic [$clog2(INPUTS+1)-1:0] saturation_count
);
    initial begin
        if (INPUTS < 1) begin
            $error("INPUTS must be positive");
        end
    end

    generate
        if (INPUTS == 1) begin : leaf
            always_comb begin
                result = values[0];
                saturation_count = '0;
            end
        end else begin : branch
            localparam integer PAIRS = INPUTS / 2;
            localparam integer NEXT_INPUTS = (INPUTS + 1) / 2;
            logic signed [WIDTH-1:0] next_values [0:NEXT_INPUTS-1];
            logic [PAIRS-1:0] pair_saturated;
            logic [$clog2(NEXT_INPUTS+1)-1:0] downstream_saturations;
            integer index;

            genvar pair;
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin : pairs
                fixed_add #(
                    .WIDTH(WIDTH),
                    .APPROX_LOW_BITS(APPROX_LOW_BITS)
                ) add_pair (
                    .first(values[pair * 2]),
                    .second(values[pair * 2 + 1]),
                    .result(next_values[pair]),
                    .saturated(pair_saturated[pair])
                );
            end

            if ((INPUTS % 2) != 0) begin : carry_tail
                always_comb begin
                    next_values[NEXT_INPUTS-1] = values[INPUTS-1];
                end
            end

            fixed_add_tree #(
                .WIDTH(WIDTH),
                .INPUTS(NEXT_INPUTS),
                .APPROX_LOW_BITS(APPROX_LOW_BITS)
            ) next_level (
                .values(next_values),
                .result(result),
                .saturation_count(downstream_saturations)
            );

            always_comb begin
                saturation_count = downstream_saturations;
                for (index = 0; index < PAIRS; index = index + 1) begin
                    saturation_count = saturation_count + pair_saturated[index];
                end
            end
        end
    endgenerate
endmodule
