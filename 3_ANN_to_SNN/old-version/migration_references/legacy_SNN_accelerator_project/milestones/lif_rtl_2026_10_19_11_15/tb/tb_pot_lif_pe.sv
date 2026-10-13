`timescale 1ns/1ps

module tb_pot_lif_pe;
  localparam int ACC_WIDTH = 20;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic soft_reset = 1'b0;
  logic approximate = 1'b0;
  logic synapse_valid = 1'b0;
  logic spike_in = 1'b0;
  logic weight_zero = 1'b0;
  logic weight_negative = 1'b0;
  logic signed [4:0] exponent = '0;
  logic signed [4:0] layer_scale = '0;
  logic commit = 1'b0;
  logic signed [ACC_WIDTH-1:0] threshold = 20'sd100;
  logic signed [ACC_WIDTH-1:0] reset_potential = '0;
  logic signed [ACC_WIDTH-1:0] synaptic_sum, membrane;
  logic spike_out, synapse_saturated, synapse_underflow, lif_saturated, protocol_error;

  always #5 clk = ~clk;

  pot_lif_pe_top #(
    .ACC_WIDTH(ACC_WIDTH), .FRACTION_BITS(0),
    .APPROX_LSB_BITS(2), .LEAK_SHIFT(0), .RESET_MODE(0)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .soft_reset_i(soft_reset),
    .approximate_i(approximate), .synapse_valid_i(synapse_valid),
    .spike_i(spike_in), .weight_zero_i(weight_zero),
    .weight_negative_i(weight_negative), .weight_exponent_i(exponent),
    .layer_scale_i(layer_scale), .timestep_commit_i(commit),
    .threshold_i(threshold), .reset_potential_i(reset_potential),
    .synaptic_sum_o(synaptic_sum), .membrane_o(membrane), .spike_o(spike_out),
    .synapse_saturated_o(synapse_saturated), .synapse_underflow_o(synapse_underflow),
    .lif_saturated_o(lif_saturated), .protocol_error_o(protocol_error)
  );

  task automatic add_pot(
    input logic event_spike,
    input logic zero,
    input logic negative,
    input logic signed [4:0] exp
  );
    begin
      @(negedge clk);
      spike_in = event_spike;
      weight_zero = zero;
      weight_negative = negative;
      exponent = exp;
      synapse_valid = 1'b1;
      @(posedge clk); #1;
      synapse_valid = 1'b0;
    end
  endtask

  task automatic reset_state;
    begin
      @(negedge clk);
      soft_reset = 1'b1;
      @(posedge clk); #1;
      soft_reset = 1'b0;
    end
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1'b1;

    add_pot(1'b1, 1'b0, 1'b0, 5'sd0); // +1
    add_pot(1'b0, 1'b0, 1'b0, 5'sd7); // hold
    add_pot(1'b1, 1'b1, 1'b0, 5'sd7); // hold
    add_pot(1'b1, 1'b0, 1'b0, 5'sd1); // +2
    if (synaptic_sum !== 20'sd3)
      $fatal(1, "exact PoT accumulation failed: %0d", synaptic_sum);

    reset_state();
    approximate = 1'b1;
    add_pot(1'b1, 1'b0, 1'b0, 5'sd0); // 1
    add_pot(1'b1, 1'b0, 1'b0, 5'sd0); // LOA: 1 OR 1 = 1
    if (synaptic_sum !== 20'sd1)
      $fatal(1, "approximate PoT accumulation failed: %0d", synaptic_sum);

    reset_state();
    approximate = 1'b0;
    layer_scale = -5'sd8;
    add_pot(1'b1, 1'b0, 1'b0, 5'sd0);
    if (!synapse_underflow || synaptic_sum !== 0)
      $fatal(1, "underflow reporting failed");

    $display("PASS tb_pot_lif_pe");
    $finish;
  end
endmodule

