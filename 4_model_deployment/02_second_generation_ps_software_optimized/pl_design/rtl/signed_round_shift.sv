`timescale 1ns/1ps

module signed_round_shift #(
    parameter integer WIDTH = 24,
    parameter integer RIGHT_SHIFT = 0
) (
    input  logic signed [WIDTH-1:0] value_in,
    output logic signed [WIDTH-1:0] value_out
);
    localparam logic signed [WIDTH:0] ROUNDING_HALF =
        (RIGHT_SHIFT == 0) ? '0 : ({{WIDTH{1'b0}}, 1'b1} <<< (RIGHT_SHIFT - 1));

    logic signed [WIDTH:0] extended;
    logic signed [WIDTH:0] magnitude;
    logic signed [WIDTH:0] rounded;

    initial begin
        if ((RIGHT_SHIFT < 0) || (RIGHT_SHIFT >= WIDTH)) begin
            $error("RIGHT_SHIFT must be in [0, WIDTH)");
        end
    end

    always_comb begin
        extended = {value_in[WIDTH-1], value_in};
        magnitude = '0;
        rounded = extended;
        if (RIGHT_SHIFT == 0) begin
            value_out = value_in;
        end else if (extended < 0) begin
            magnitude = -extended;
            rounded = -((magnitude + ROUNDING_HALF) >>> RIGHT_SHIFT);
            value_out = rounded[WIDTH-1:0];
        end else begin
            rounded = (extended + ROUNDING_HALF) >>> RIGHT_SHIFT;
            value_out = rounded[WIDTH-1:0];
        end
    end
endmodule
