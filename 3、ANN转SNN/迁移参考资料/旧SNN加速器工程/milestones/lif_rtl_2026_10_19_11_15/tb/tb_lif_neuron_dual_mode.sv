`timescale 1ns/1ps

module tb_lif_neuron_dual_mode;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic soft_reset = 1'b0;
  logic enable = 1'b0;
  logic approximate = 1'b0;
  logic signed [7:0] synaptic = '0;
  logic signed [7:0] threshold = 8'sd100;
  logic signed [7:0] reset_potential = '0;
  logic signed [7:0] membrane;
  logic spike, saturated;

  always #5 clk = ~clk;

  lif_neuron_dual_mode #(
    .WIDTH(8),
    .APPROX_LSB_BITS(2),
    .LEAK_SHIFT(0),
    .RESET_MODE(0)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .soft_reset_i(soft_reset),
    .enable_i(enable), .approximate_i(approximate),
    .synaptic_i(synaptic), .threshold_i(threshold),
    .reset_potential_i(reset_potential), .membrane_o(membrane),
    .spike_o(spike), .saturated_o(saturated)
  );

  task automatic update(input logic signed [7:0] value);
    begin
      @(negedge clk);
      synaptic = value;
      enable = 1'b1;
      @(posedge clk);
      #1;
      enable = 1'b0;
      synaptic = '0;
    end
  endtask

  task automatic reset_state;
    begin
      @(negedge clk);
      soft_reset = 1'b1;
      enable = 1'b1;
      synaptic = 8'sd120;
      @(posedge clk);
      #1;
      soft_reset = 1'b0;
      enable = 1'b0;
      synaptic = '0;
      if (membrane !== 0 || spike !== 0)
        $fatal(1, "soft reset did not take priority");
    end
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1'b1;

    update(8'sd60);
    if (membrane !== 8'sd60 || spike)
      $fatal(1, "exact first update failed");
    update(8'sd50);
    if (membrane !== 0 || !spike)
      $fatal(1, "threshold/hard reset failed");
    @(posedge clk); #1;
    if (spike)
      $fatal(1, "spike must be one cycle");

    reset_state();
    approximate = 1'b1;
    update(8'sd3);
    update(8'sd7);
    if (membrane !== 8'sd7)
      $fatal(1, "LOA update mismatch: %0d", membrane);

    reset_state();
    approximate = 1'b0;
    update(8'sd90);
    update(8'sd100);
    if (!spike || !saturated || membrane !== 0)
      $fatal(1, "saturation/spike reporting failed");

    $display("PASS tb_lif_neuron_dual_mode");
    $finish;
  end
endmodule

