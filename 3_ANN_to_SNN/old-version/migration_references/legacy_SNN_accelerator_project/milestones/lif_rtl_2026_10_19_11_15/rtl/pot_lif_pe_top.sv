`timescale 1ns/1ps

module pot_lif_pe_top #(
  parameter int ACC_WIDTH = 20,
  parameter int FRACTION_BITS = 8,
  parameter int APPROX_LSB_BITS = 4,
  parameter int LEAK_SHIFT = 3,
  parameter int RESET_MODE = 0
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,
  input  logic                        soft_reset_i,
  input  logic                        approximate_i,
  input  logic                        synapse_valid_i,
  input  logic                        spike_i,
  input  logic                        weight_zero_i,
  input  logic                        weight_negative_i,
  input  logic signed [4:0]           weight_exponent_i,
  input  logic signed [4:0]           layer_scale_i,
  input  logic                        timestep_commit_i,
  input  logic signed [ACC_WIDTH-1:0] threshold_i,
  input  logic signed [ACC_WIDTH-1:0] reset_potential_i,
  output logic signed [ACC_WIDTH-1:0] synaptic_sum_o,
  output logic signed [ACC_WIDTH-1:0] membrane_o,
  output logic                        spike_o,
  output logic                        synapse_saturated_o,
  output logic                        synapse_underflow_o,
  output logic                        lif_saturated_o,
  output logic                        protocol_error_o
);
  pot_shift_accumulator #(
    .ACC_WIDTH(ACC_WIDTH),
    .FRACTION_BITS(FRACTION_BITS),
    .APPROX_LSB_BITS(APPROX_LSB_BITS)
  ) synapse_accumulator (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .clear_i(soft_reset_i || timestep_commit_i),
    .enable_i(synapse_valid_i),
    .approximate_i(approximate_i),
    .spike_i(spike_i),
    .weight_zero_i(weight_zero_i),
    .weight_negative_i(weight_negative_i),
    .weight_exponent_i(weight_exponent_i),
    .layer_scale_i(layer_scale_i),
    .accumulator_o(synaptic_sum_o),
    .saturated_o(synapse_saturated_o),
    .underflow_o(synapse_underflow_o)
  );

  lif_neuron_dual_mode #(
    .WIDTH(ACC_WIDTH),
    .APPROX_LSB_BITS(APPROX_LSB_BITS),
    .LEAK_SHIFT(LEAK_SHIFT),
    .RESET_MODE(RESET_MODE)
  ) lif (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .enable_i(timestep_commit_i),
    .approximate_i(approximate_i),
    .synaptic_i(synaptic_sum_o),
    .threshold_i(threshold_i),
    .reset_potential_i(reset_potential_i),
    .membrane_o(membrane_o),
    .spike_o(spike_o),
    .saturated_o(lif_saturated_o)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni || soft_reset_i)
      protocol_error_o <= 1'b0;
    else if (synapse_valid_i && timestep_commit_i)
      protocol_error_o <= 1'b1;
  end
endmodule
