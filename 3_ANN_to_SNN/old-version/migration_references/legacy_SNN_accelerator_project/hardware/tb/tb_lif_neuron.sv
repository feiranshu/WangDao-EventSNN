`timescale 1ns/1ps

module tb_lif_neuron;
  logic clk = 0;
  logic rst_n = 0;
  logic enable = 0;
  logic clear = 0;
  logic approximate = 0;
  logic signed [15:0] synaptic = 0;
  logic signed [15:0] membrane;
  logic spike, saturated;

  always #5 clk = ~clk;

  lif_neuron #(
    .WIDTH(16), .APPROX_LSB_BITS(2), .LEAK_SHIFT(0),
    .THRESHOLD(16'sd100)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .enable_i(enable), .clear_i(clear),
    .approximate_i(approximate), .synaptic_i(synaptic),
    .membrane_o(membrane), .spike_o(spike), .saturated_o(saturated)
  );

  task automatic drive(input logic signed [15:0] value);
    begin
      @(negedge clk);
      synaptic = value;
      enable = 1;
      @(negedge clk);
      enable = 0;
      synaptic = 0;
    end
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1;
    drive(16'sd60);
    if (membrane !== 16'sd60 || spike !== 1'b0) $fatal(1, "first accumulation failed");
    drive(16'sd50);
    if (membrane !== 16'sd0 || spike !== 1'b1) $fatal(1, "spike/reset failed");
    @(negedge clk);
    if (spike !== 1'b0) $fatal(1, "spike was not a pulse");
    $display("PASS tb_lif_neuron");
    $finish;
  end
endmodule

