`timescale 1ns/1ps

module int8_lif_baseline_top #(
  parameter int ACC_WIDTH = 20,
  parameter int LEAK_SHIFT = 3,
  parameter int RESET_MODE = 0
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,
  input  logic                        soft_reset_i,
  input  logic                        synapse_valid_i,
  input  logic signed [7:0]           activation_i,
  input  logic signed [7:0]           weight_i,
  input  logic                        timestep_commit_i,
  input  logic signed [ACC_WIDTH-1:0] threshold_i,
  input  logic signed [ACC_WIDTH-1:0] reset_potential_i,
  output logic signed [ACC_WIDTH-1:0] synaptic_sum_o,
  output logic signed [ACC_WIDTH-1:0] membrane_o,
  output logic                        spike_o,
  output logic                        synapse_saturated_o,
  output logic                        lif_saturated_o,
  output logic                        protocol_error_o
);
  int8_mac_accumulator #(
    .ACC_WIDTH(ACC_WIDTH)
  ) synapse_accumulator (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .clear_i(soft_reset_i || timestep_commit_i),
    .enable_i(synapse_valid_i),
    .activation_i(activation_i),
    .weight_i(weight_i),
    .accumulator_o(synaptic_sum_o),
    .saturated_o(synapse_saturated_o)
  );

  lif_neuron_dual_mode #(
    .WIDTH(ACC_WIDTH),
    .APPROX_LSB_BITS(4),
    .LEAK_SHIFT(LEAK_SHIFT),
    .RESET_MODE(RESET_MODE)
  ) lif (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .enable_i(timestep_commit_i),
    .approximate_i(1'b0),
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
