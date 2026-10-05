`timescale 1ns/1ps

module fixed_add #(
    parameter integer WIDTH = 24,
    parameter integer APPROX_LOW_BITS = 0
) (
    input  logic signed [WIDTH-1:0] first,
    input  logic signed [WIDTH-1:0] second,
    output logic signed [WIDTH-1:0] result,
    output logic saturated
);
    localparam logic signed [WIDTH:0] MAXIMUM = {
        2'b00, {(WIDTH-1){1'b1}}
    };
    localparam logic signed [WIDTH:0] MINIMUM = {
        2'b11, {(WIDTH-1){1'b0}}
    };

    logic signed [WIDTH:0] raw_result;
    wire signed [WIDTH:0] first_extended = {first[WIDTH-1], first};
    wire signed [WIDTH:0] second_extended = {second[WIDTH-1], second};

    initial begin
        if (WIDTH < 2) begin
            $error("WIDTH must be at least 2");
        end
        if ((APPROX_LOW_BITS < 0) || (APPROX_LOW_BITS >= WIDTH)) begin
            $error("APPROX_LOW_BITS must be in [0, WIDTH)");
        end
    end

    generate
        if (APPROX_LOW_BITS == 0) begin : exact_addition
            always_comb begin
                raw_result = first_extended + second_extended;
            end
        end else begin : loa_addition
            localparam logic [WIDTH-1:0] LOW_MASK =
                {WIDTH{1'b1}} >> (WIDTH - APPROX_LOW_BITS);
            logic [WIDTH-1:0] low_result;
            logic carry;
            logic signed [WIDTH:0] high_result;

            always_comb begin
                low_result = (first | second) & LOW_MASK;
                carry = first[APPROX_LOW_BITS-1]
                    & second[APPROX_LOW_BITS-1];
                high_result = (first_extended >>> APPROX_LOW_BITS)
                    + (second_extended >>> APPROX_LOW_BITS)
                    + carry;
                raw_result = (high_result <<< APPROX_LOW_BITS)
                    + $signed({1'b0, low_result});
            end
        end
    endgenerate

    always_comb begin
        saturated = 1'b0;
        if (raw_result > MAXIMUM) begin
            result = {1'b0, {(WIDTH-1){1'b1}}};
            saturated = 1'b1;
        end else if (raw_result < MINIMUM) begin
            result = {1'b1, {(WIDTH-1){1'b0}}};
            saturated = 1'b1;
        end else begin
            result = raw_result[WIDTH-1:0];
        end
    end
endmodule
