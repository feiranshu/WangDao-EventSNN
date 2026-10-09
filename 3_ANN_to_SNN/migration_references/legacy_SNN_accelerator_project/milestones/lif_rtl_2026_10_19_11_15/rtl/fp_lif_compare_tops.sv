`timescale 1ns/1ps

module fp32_lif_exact_top (
  input logic clk_i, rst_ni, soft_reset_i,
  input logic input_valid_i,
  output logic input_ready_o,
  input logic [31:0] current_i, alpha_i, threshold_i,
  output logic output_valid_o,
  input logic output_ready_i,
  output logic [31:0] membrane_o,
  output logic spike_o, busy_o
);
  fp_lif_neuron_xilinx #(.WIDTH(32), .APPROXIMATE(1'b0)) neuron (.*);
endmodule

module fp32_lif_approx_top (
  input logic clk_i, rst_ni, soft_reset_i,
  input logic input_valid_i,
  output logic input_ready_o,
  input logic [31:0] current_i, alpha_i, threshold_i,
  output logic output_valid_o,
  input logic output_ready_i,
  output logic [31:0] membrane_o,
  output logic spike_o, busy_o
);
  fp_lif_neuron_xilinx #(.WIDTH(32), .APPROXIMATE(1'b1)) neuron (.*);
endmodule

module fp16_lif_exact_top (
  input logic clk_i, rst_ni, soft_reset_i,
  input logic input_valid_i,
  output logic input_ready_o,
  input logic [15:0] current_i, alpha_i, threshold_i,
  output logic output_valid_o,
  input logic output_ready_i,
  output logic [15:0] membrane_o,
  output logic spike_o, busy_o
);
  fp_lif_neuron_xilinx #(.WIDTH(16), .APPROXIMATE(1'b0)) neuron (.*);
endmodule

module fp16_lif_approx_top (
  input logic clk_i, rst_ni, soft_reset_i,
  input logic input_valid_i,
  output logic input_ready_o,
  input logic [15:0] current_i, alpha_i, threshold_i,
  output logic output_valid_o,
  input logic output_ready_i,
  output logic [15:0] membrane_o,
  output logic spike_o, busy_o
);
  fp_lif_neuron_xilinx #(.WIDTH(16), .APPROXIMATE(1'b1)) neuron (.*);
endmodule
