`timescale 1ns/1ps

module lif_neuron_dual_mode #(
  parameter int WIDTH = 20,
  parameter int APPROX_LSB_BITS = 4,
  parameter int LEAK_SHIFT = 3,
  parameter int RESET_MODE = 0
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  input  logic                    soft_reset_i,
  input  logic                    enable_i,
  input  logic                    approximate_i,
  input  logic signed [WIDTH-1:0] synaptic_i,
  input  logic signed [WIDTH-1:0] threshold_i,
  input  logic signed [WIDTH-1:0] reset_potential_i,
  output logic signed [WIDTH-1:0] membrane_o,
  output logic                    spike_o,
  output logic                    saturated_o
);
  logic signed [WIDTH-1:0] leak_value;
  logic signed [WIDTH-1:0] decayed_membrane;
  logic signed [WIDTH-1:0] updated_membrane;
  logic signed [WIDTH-1:0] subtractive_reset_value;
  logic add_saturated;

  initial begin
    if (RESET_MODE != 0 && RESET_MODE != 1)
      $fatal(1, "RESET_MODE must be 0 (hard) or 1 (subtractive)");
    if (LEAK_SHIFT < 0 || LEAK_SHIFT >= WIDTH)
      $fatal(1, "LEAK_SHIFT must be in [0, WIDTH-1]");
  end

  always_comb begin
    if (LEAK_SHIFT == 0)
      leak_value = '0;
    else
      leak_value = membrane_o >>> LEAK_SHIFT;
    decayed_membrane = membrane_o - leak_value;
    subtractive_reset_value = updated_membrane - threshold_i;
  end

  dual_mode_sat_add #(
    .WIDTH(WIDTH),
    .APPROX_LSB_BITS(APPROX_LSB_BITS)
  ) update_adder (
    .a_i(decayed_membrane),
    .b_i(synaptic_i),
    .approximate_i(approximate_i),
    .sum_o(updated_membrane),
    .saturated_o(add_saturated)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni || soft_reset_i) begin
      membrane_o <= reset_potential_i;
      spike_o <= 1'b0;
      saturated_o <= 1'b0;
    end else begin
      spike_o <= 1'b0;
      saturated_o <= 1'b0;
      if (enable_i) begin
        saturated_o <= add_saturated;
        if (updated_membrane >= threshold_i) begin
          spike_o <= 1'b1;
          if (RESET_MODE == 0)
            membrane_o <= reset_potential_i;
          else
            membrane_o <= subtractive_reset_value;
        end else begin
          membrane_o <= updated_membrane;
        end
      end
    end
  end
endmodule
