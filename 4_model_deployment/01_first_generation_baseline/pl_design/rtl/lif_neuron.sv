`timescale 1ns/1ps

module lif_neuron #(
    parameter integer WIDTH = 24,
    parameter integer BETA_NUMERATOR = 7,
    parameter integer BETA_SHIFT = 3,
    parameter integer THRESHOLD_Q = 1024,
    parameter integer APPROX_LOW_BITS = 0
) (
    input  logic signed [WIDTH-1:0] membrane_in,
    input  logic signed [WIDTH-1:0] current_in,
    input  logic fire_enable,
    output logic signed [WIDTH-1:0] membrane_out,
    output logic spike,
    output logic saturated
);
    localparam integer PRODUCT_WIDTH = WIDTH * 2;
    localparam logic signed [PRODUCT_WIDTH-1:0] ROUNDING_HALF =
        1 <<< (BETA_SHIFT - 1);

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [PRODUCT_WIDTH-1:0] leaked_wide;
    logic signed [WIDTH-1:0] leaked;
    logic signed [WIDTH-1:0] updated;

    initial begin
        if (BETA_NUMERATOR <= 0) begin
            $error("BETA_NUMERATOR must be positive");
        end
        if (BETA_SHIFT <= 0) begin
            $error("BETA_SHIFT must be positive");
        end
    end

    always_comb begin
        product = $signed(membrane_in) * BETA_NUMERATOR;
        if (product < 0) begin
            leaked_wide = -((-product + ROUNDING_HALF) >>> BETA_SHIFT);
        end else begin
            leaked_wide = (product + ROUNDING_HALF) >>> BETA_SHIFT;
        end
        leaked = leaked_wide[WIDTH-1:0];
    end

    fixed_add #(
        .WIDTH(WIDTH),
        .APPROX_LOW_BITS(APPROX_LOW_BITS)
    ) membrane_addition (
        .first(leaked),
        .second(current_in),
        .result(updated),
        .saturated(saturated)
    );

    always_comb begin
        spike = fire_enable && ($signed(updated) >= THRESHOLD_Q);
        if (spike) begin
            membrane_out = $signed(updated) - THRESHOLD_Q;
        end else begin
            membrane_out = updated;
        end
    end
endmodule
