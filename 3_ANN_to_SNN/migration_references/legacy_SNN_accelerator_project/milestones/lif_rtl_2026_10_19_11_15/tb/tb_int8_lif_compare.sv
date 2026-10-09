`timescale 1ns/1ps

module tb_int8_lif_compare;
  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic enable_i = 1'b0;
  logic signed [7:0] current_i = '0;
  logic signed [7:0] threshold_i = 8'sd100;
  logic signed [7:0] exact_membrane;
  logic signed [7:0] approximate_membrane;
  logic exact_spike;
  logic approximate_spike;
  logic exact_saturated;
  logic approximate_saturated;

  always #5 clk_i = ~clk_i;

  int8_lif_exact_top exact_dut (
    .clk_i, .rst_ni, .soft_reset_i, .enable_i, .current_i, .threshold_i,
    .membrane_o(exact_membrane), .spike_o(exact_spike),
    .saturated_o(exact_saturated)
  );

  int8_lif_approx_top approximate_dut (
    .clk_i, .rst_ni, .soft_reset_i, .enable_i, .current_i, .threshold_i,
    .membrane_o(approximate_membrane), .spike_o(approximate_spike),
    .saturated_o(approximate_saturated)
  );

  task automatic update(input logic signed [7:0] current);
    begin
      @(negedge clk_i);
      current_i = current;
      enable_i = 1'b1;
      @(negedge clk_i);
      enable_i = 1'b0;
    end
  endtask

  initial begin
    repeat (2) @(negedge clk_i);
    rst_ni = 1'b1;

    update(8'sd3);
    update(8'sd7);
    if (exact_membrane !== 8'sd10)
      $fatal(1, "exact membrane mismatch: %0d", exact_membrane);
    if (approximate_membrane !== 8'sd7)
      $fatal(1, "approximate membrane mismatch: %0d", approximate_membrane);
    if (exact_spike || approximate_spike || exact_saturated || approximate_saturated)
      $fatal(1, "unexpected spike or saturation");

    $display("PASS tb_int8_lif_compare");
    $finish;
  end
endmodule
