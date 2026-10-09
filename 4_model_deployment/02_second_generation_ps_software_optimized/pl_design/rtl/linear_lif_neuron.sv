`timescale 1ns/1ps

module linear_lif_neuron #(
    parameter integer INPUTS = 128,
    parameter integer FEATURE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 24,
    parameter integer INPUT_FRACTION_BITS = 8,
    parameter integer WEIGHT_FRACTION_BITS = 5,
    parameter integer MEMBRANE_FRACTION_BITS = 12,
    parameter integer BETA_NUMERATOR = 7,
    parameter integer BETA_SHIFT = 3,
    parameter integer THRESHOLD_Q = 1024,
    parameter integer SYN_APPROX_LOW_BITS = 0,
    parameter integer MEM_APPROX_LOW_BITS = 0
) (
    input  logic [FEATURE_WIDTH-1:0] features [0:INPUTS-1],
    input  logic signed [WEIGHT_WIDTH-1:0] weights [0:INPUTS-1],
    input  logic signed [ACC_WIDTH-1:0] membrane_in,
    input  logic fire_enable,
    output logic signed [ACC_WIDTH-1:0] membrane_out,
    output logic spike,
    output logic [$clog2(INPUTS+1)-1:0] tree_saturation_count,
    output logic membrane_saturated
);
    localparam integer PRODUCT_WIDTH = FEATURE_WIDTH + WEIGHT_WIDTH + 1;
    localparam integer RIGHT_SHIFT =
        INPUT_FRACTION_BITS + WEIGHT_FRACTION_BITS - MEMBRANE_FRACTION_BITS;

    logic signed [ACC_WIDTH-1:0] products [0:INPUTS-1];
    logic signed [ACC_WIDTH-1:0] accumulated;
    logic signed [ACC_WIDTH-1:0] aligned_current;

    initial begin
        if (ACC_WIDTH < PRODUCT_WIDTH) begin
            $error("ACC_WIDTH is too small for one feature-weight product");
        end
        if ((RIGHT_SHIFT < 0) || (RIGHT_SHIFT >= ACC_WIDTH)) begin
            $error("The frozen first-layer scale requires a valid right shift");
        end
    end

    genvar input_index;
    generate
        for (input_index = 0; input_index < INPUTS; input_index = input_index + 1) begin : products_by_index
            logic signed [FEATURE_WIDTH:0] positive_feature;
            logic signed [PRODUCT_WIDTH-1:0] product_wide;

            always_comb begin
                positive_feature = $signed({1'b0, features[input_index]});
                product_wide = positive_feature * $signed(weights[input_index]);
                products[input_index] = {
                    {(ACC_WIDTH-PRODUCT_WIDTH){product_wide[PRODUCT_WIDTH-1]}},
                    product_wide
                };
            end
        end
    endgenerate

    fixed_add_tree #(
        .WIDTH(ACC_WIDTH),
        .INPUTS(INPUTS),
        .APPROX_LOW_BITS(SYN_APPROX_LOW_BITS)
    ) synaptic_tree (
        .values(products),
        .result(accumulated),
        .saturation_count(tree_saturation_count)
    );

    signed_round_shift #(
        .WIDTH(ACC_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT)
    ) align_to_membrane (
        .value_in(accumulated),
        .value_out(aligned_current)
    );

    lif_neuron #(
        .WIDTH(ACC_WIDTH),
        .BETA_NUMERATOR(BETA_NUMERATOR),
        .BETA_SHIFT(BETA_SHIFT),
        .THRESHOLD_Q(THRESHOLD_Q),
        .APPROX_LOW_BITS(MEM_APPROX_LOW_BITS)
    ) neuron (
        .membrane_in(membrane_in),
        .current_in(aligned_current),
        .fire_enable(fire_enable),
        .membrane_out(membrane_out),
        .spike(spike),
        .saturated(membrane_saturated)
    );
endmodule
