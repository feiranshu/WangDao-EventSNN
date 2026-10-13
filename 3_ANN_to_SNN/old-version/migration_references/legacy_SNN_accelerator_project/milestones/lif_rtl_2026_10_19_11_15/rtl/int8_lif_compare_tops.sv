`timescale 1ns/1ps

// These two OOC tops isolate the INT8 neuron itself.  They intentionally do
// not include the INT8 MAC baseline; that baseline remains in
// int8_lif_baseline_top.sv.
module int8_lif_exact_top (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  input  logic                    soft_reset_i,
  input  logic                    enable_i,
  input  logic signed [7:0]       current_i,
  input  logic signed [7:0]       threshold_i,
  output logic signed [7:0]       membrane_o,
  output logic                    spike_o,
  output logic                    saturated_o
);
  lif_neuron_dual_mode #(
    .WIDTH(8),
    .APPROX_LSB_BITS(2),
    .LEAK_SHIFT(2),
    .RESET_MODE(0)
  ) neuron (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .enable_i(enable_i),
    .approximate_i(1'b0),
    .synaptic_i(current_i),
    .threshold_i(threshold_i),
    .reset_potential_i(8'sd0),
    .membrane_o(membrane_o),
    .spike_o(spike_o),
    .saturated_o(saturated_o)
  );
endmodule

module int8_lif_approx_top (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  input  logic                    soft_reset_i,
  input  logic                    enable_i,
  input  logic signed [7:0]       current_i,
  input  logic signed [7:0]       threshold_i,
  output logic signed [7:0]       membrane_o,
  output logic                    spike_o,
  output logic                    saturated_o
);
  lif_neuron_dual_mode #(
    .WIDTH(8),
    .APPROX_LSB_BITS(2),
    .LEAK_SHIFT(2),
    .RESET_MODE(0)
  ) neuron (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .enable_i(enable_i),
    .approximate_i(1'b1),
    .synaptic_i(current_i),
    .threshold_i(threshold_i),
    .reset_potential_i(8'sd0),
    .membrane_o(membrane_o),
    .spike_o(spike_o),
    .saturated_o(saturated_o)
  );
endmodule
