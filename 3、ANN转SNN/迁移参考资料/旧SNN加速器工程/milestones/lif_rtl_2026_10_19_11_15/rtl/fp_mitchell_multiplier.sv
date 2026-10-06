`timescale 1ns/1ps

// Mitchell log-domain approximate floating-point multiplier.
// - Supports IEEE-754 binary16 (EXP=5, FRAC=10) and binary32 (EXP=8, FRAC=23).
// - Normal finite operands use log2(1+f) ~= f, replacing the mantissa
//   multiplier with an adder and exponent adjustment.
// - Subnormal inputs/results are flushed to zero (FTZ).
// - NaN/Inf/zero cases are handled explicitly.
module fp_mitchell_multiplier #(
  parameter int EXP_WIDTH = 8,
  parameter int FRAC_WIDTH = 23
) (
  input  logic [EXP_WIDTH+FRAC_WIDTH:0] a_i,
  input  logic [EXP_WIDTH+FRAC_WIDTH:0] b_i,
  output logic [EXP_WIDTH+FRAC_WIDTH:0] result_o
);
  localparam int TOTAL_WIDTH = 1 + EXP_WIDTH + FRAC_WIDTH;
  localparam int EXP_MAX_INT = (1 << EXP_WIDTH) - 1;
  localparam int EXP_BIAS = (1 << (EXP_WIDTH - 1)) - 1;
  localparam logic [EXP_WIDTH-1:0] EXP_ALL_ONES = {EXP_WIDTH{1'b1}};
  localparam logic [FRAC_WIDTH-1:0] QUIET_NAN_FRACTION =
    {1'b1, {(FRAC_WIDTH-1){1'b0}}};

  logic sign_result;
  logic [EXP_WIDTH-1:0] exponent_a;
  logic [EXP_WIDTH-1:0] exponent_b;
  logic [FRAC_WIDTH-1:0] fraction_a;
  logic [FRAC_WIDTH-1:0] fraction_b;
  logic [FRAC_WIDTH:0] fraction_sum;
  logic a_zero_or_subnormal;
  logic b_zero_or_subnormal;
  logic a_infinite;
  logic b_infinite;
  logic a_nan;
  logic b_nan;
  integer exponent_result;

  initial begin
    if (!((EXP_WIDTH == 5 && FRAC_WIDTH == 10) ||
          (EXP_WIDTH == 8 && FRAC_WIDTH == 23)))
      $fatal(1, "Only IEEE binary16 and binary32 formats are supported");
    if (TOTAL_WIDTH != 16 && TOTAL_WIDTH != 32)
      $fatal(1, "Unexpected floating-point width");
  end

  always_comb begin
    sign_result = a_i[TOTAL_WIDTH-1] ^ b_i[TOTAL_WIDTH-1];
    exponent_a = a_i[FRAC_WIDTH +: EXP_WIDTH];
    exponent_b = b_i[FRAC_WIDTH +: EXP_WIDTH];
    fraction_a = a_i[FRAC_WIDTH-1:0];
    fraction_b = b_i[FRAC_WIDTH-1:0];
    fraction_sum = {1'b0, fraction_a} + {1'b0, fraction_b};

    a_zero_or_subnormal = (exponent_a == '0);
    b_zero_or_subnormal = (exponent_b == '0);
    a_infinite = (exponent_a == EXP_ALL_ONES) && (fraction_a == '0);
    b_infinite = (exponent_b == EXP_ALL_ONES) && (fraction_b == '0);
    a_nan = (exponent_a == EXP_ALL_ONES) && (fraction_a != '0);
    b_nan = (exponent_b == EXP_ALL_ONES) && (fraction_b != '0);

    exponent_result = $signed({1'b0, exponent_a})
                    + $signed({1'b0, exponent_b})
                    - EXP_BIAS
                    + fraction_sum[FRAC_WIDTH];

    result_o = '0;
    if (a_nan || b_nan ||
        ((a_infinite || b_infinite) &&
         (a_zero_or_subnormal || b_zero_or_subnormal))) begin
      result_o = {1'b0, EXP_ALL_ONES, QUIET_NAN_FRACTION};
    end else if (a_infinite || b_infinite) begin
      result_o = {sign_result, EXP_ALL_ONES, {FRAC_WIDTH{1'b0}}};
    end else if (a_zero_or_subnormal || b_zero_or_subnormal) begin
      result_o = {sign_result, {EXP_WIDTH{1'b0}}, {FRAC_WIDTH{1'b0}}};
    end else if (exponent_result >= EXP_MAX_INT) begin
      result_o = {sign_result, EXP_ALL_ONES, {FRAC_WIDTH{1'b0}}};
    end else if (exponent_result <= 0) begin
      result_o = {sign_result, {EXP_WIDTH{1'b0}}, {FRAC_WIDTH{1'b0}}};
    end else begin
      result_o = {
        sign_result,
        exponent_result[EXP_WIDTH-1:0],
        fraction_sum[FRAC_WIDTH-1:0]
      };
    end
  end
endmodule
