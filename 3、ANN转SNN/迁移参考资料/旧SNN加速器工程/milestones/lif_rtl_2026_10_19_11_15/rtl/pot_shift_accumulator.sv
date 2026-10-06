`timescale 1ns/1ps

module pot_shift_accumulator #(
  parameter int ACC_WIDTH = 20,
  parameter int FRACTION_BITS = 8,
  parameter int APPROX_LSB_BITS = 4
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,
  input  logic                        clear_i,
  input  logic                        enable_i,
  input  logic                        approximate_i,
  input  logic                        spike_i,
  input  logic                        weight_zero_i,
  input  logic                        weight_negative_i,
  input  logic signed [4:0]           weight_exponent_i,
  input  logic signed [4:0]           layer_scale_i,
  output logic signed [ACC_WIDTH-1:0] accumulator_o,
  output logic                        saturated_o,
  output logic                        underflow_o
);
  localparam logic signed [ACC_WIDTH-1:0] MAX_VALUE = {1'b0, {(ACC_WIDTH-1){1'b1}}};
  localparam logic signed [ACC_WIDTH-1:0] MIN_VALUE = {1'b1, {(ACC_WIDTH-1){1'b0}}};

  integer signed bit_index;
  logic do_update;
  logic signed [ACC_WIDTH-1:0] magnitude;
  logic signed [ACC_WIDTH-1:0] contribution;
  logic contribution_saturated;
  logic contribution_underflow;
  logic signed [ACC_WIDTH-1:0] next_accumulator;
  logic add_saturated;

  initial begin
    if (ACC_WIDTH < 4)
      $fatal(1, "ACC_WIDTH must be at least four");
    if (FRACTION_BITS < 0 || FRACTION_BITS >= ACC_WIDTH)
      $fatal(1, "FRACTION_BITS must be in [0, ACC_WIDTH-1]");
  end

  always_comb begin
    do_update = enable_i && spike_i && !weight_zero_i;
    bit_index = FRACTION_BITS + $signed(weight_exponent_i) + $signed(layer_scale_i);
    magnitude = '0;
    contribution = '0;
    contribution_saturated = 1'b0;
    contribution_underflow = 1'b0;

    if (do_update) begin
      if (bit_index < 0) begin
        contribution = '0;
        contribution_underflow = 1'b1;
      end else if (bit_index > ACC_WIDTH-1) begin
        contribution = weight_negative_i ? MIN_VALUE : MAX_VALUE;
        contribution_saturated = 1'b1;
      end else if (bit_index == ACC_WIDTH-1) begin
        if (weight_negative_i) begin
          contribution = MIN_VALUE;
        end else begin
          contribution = MAX_VALUE;
          contribution_saturated = 1'b1;
        end
      end else begin
        magnitude = {{(ACC_WIDTH-1){1'b0}}, 1'b1} << bit_index;
        contribution = weight_negative_i ? -magnitude : magnitude;
      end
    end
  end

  dual_mode_sat_add #(
    .WIDTH(ACC_WIDTH),
    .APPROX_LSB_BITS(APPROX_LSB_BITS)
  ) accumulator_adder (
    .a_i(accumulator_o),
    .b_i(contribution),
    .approximate_i(approximate_i),
    .sum_o(next_accumulator),
    .saturated_o(add_saturated)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni || clear_i) begin
      accumulator_o <= '0;
      saturated_o <= 1'b0;
      underflow_o <= 1'b0;
    end else begin
      saturated_o <= 1'b0;
      underflow_o <= 1'b0;
      if (do_update) begin
        accumulator_o <= next_accumulator;
        saturated_o <= contribution_saturated || add_saturated;
        underflow_o <= contribution_underflow;
      end
    end
  end
endmodule
