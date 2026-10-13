`timescale 1ns/1ps

module int8_mac_accumulator #(
  parameter int ACC_WIDTH = 20
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            clear_i,
  input  logic                            enable_i,
  input  logic signed [7:0]               activation_i,
  input  logic signed [7:0]               weight_i,
  output logic signed [ACC_WIDTH-1:0]     accumulator_o,
  output logic                            saturated_o
);
  logic signed [15:0] product;
  logic signed [ACC_WIDTH-1:0] product_extended;
  logic signed [ACC_WIDTH-1:0] next_accumulator;
  logic add_saturated;

  initial begin
    if (ACC_WIDTH < 17)
      $fatal(1, "ACC_WIDTH must be at least 17 for signed INT8 products");
  end

  always_comb begin
    product = activation_i * weight_i;
    product_extended = {{(ACC_WIDTH-16){product[15]}}, product};
  end

  dual_mode_sat_add #(
    .WIDTH(ACC_WIDTH),
    .APPROX_LSB_BITS(4)
  ) accumulator_adder (
    .a_i(accumulator_o),
    .b_i(product_extended),
    .approximate_i(1'b0),
    .sum_o(next_accumulator),
    .saturated_o(add_saturated)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni || clear_i) begin
      accumulator_o <= '0;
      saturated_o <= 1'b0;
    end else begin
      saturated_o <= 1'b0;
      if (enable_i) begin
        accumulator_o <= next_accumulator;
        saturated_o <= add_saturated;
      end
    end
  end
endmodule
