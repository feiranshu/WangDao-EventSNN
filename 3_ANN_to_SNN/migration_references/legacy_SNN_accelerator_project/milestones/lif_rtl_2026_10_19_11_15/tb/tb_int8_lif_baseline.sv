`timescale 1ns/1ps

module tb_int8_lif_baseline;
  localparam int ACC_WIDTH = 20;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic soft_reset = 1'b0;
  logic synapse_valid = 1'b0;
  logic signed [7:0] activation = '0;
  logic signed [7:0] weight = '0;
  logic commit = 1'b0;
  logic signed [ACC_WIDTH-1:0] threshold = 20'sd25;
  logic signed [ACC_WIDTH-1:0] reset_potential = '0;
  logic signed [ACC_WIDTH-1:0] synaptic_sum, membrane;
  logic spike, synapse_saturated, lif_saturated, protocol_error;

  always #5 clk = ~clk;

  int8_lif_baseline_top #(
    .ACC_WIDTH(ACC_WIDTH), .LEAK_SHIFT(0), .RESET_MODE(0)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .soft_reset_i(soft_reset),
    .synapse_valid_i(synapse_valid), .activation_i(activation),
    .weight_i(weight), .timestep_commit_i(commit),
    .threshold_i(threshold), .reset_potential_i(reset_potential),
    .synaptic_sum_o(synaptic_sum), .membrane_o(membrane), .spike_o(spike),
    .synapse_saturated_o(synapse_saturated), .lif_saturated_o(lif_saturated),
    .protocol_error_o(protocol_error)
  );

  task automatic add_product(input logic signed [7:0] a, input logic signed [7:0] w);
    begin
      @(negedge clk);
      activation = a;
      weight = w;
      synapse_valid = 1'b1;
      @(posedge clk); #1;
      synapse_valid = 1'b0;
    end
  endtask

  task automatic commit_step;
    begin
      @(negedge clk);
      commit = 1'b1;
      @(posedge clk); #1;
      commit = 1'b0;
    end
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1'b1;
    add_product(8'sd10, 8'sd3);
    add_product(-8'sd4, 8'sd2);
    if (synaptic_sum !== 20'sd22)
      $fatal(1, "INT8 MAC sum mismatch: %0d", synaptic_sum);
    commit_step();
    if (membrane !== 20'sd22 || spike || synaptic_sum !== 0)
      $fatal(1, "first commit failed");

    add_product(8'sd5, 8'sd1);
    commit_step();
    if (!spike || membrane !== 0)
      $fatal(1, "second commit should fire and reset");

    @(negedge clk);
    synapse_valid = 1'b1;
    commit = 1'b1;
    @(posedge clk); #1;
    synapse_valid = 1'b0;
    commit = 1'b0;
    if (!protocol_error)
      $fatal(1, "protocol error was not latched");

    $display("PASS tb_int8_lif_baseline");
    $finish;
  end
endmodule

