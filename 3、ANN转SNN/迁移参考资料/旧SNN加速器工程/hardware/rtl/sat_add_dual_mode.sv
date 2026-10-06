module sat_add_dual_mode #(
  parameter int WIDTH = 20,
  parameter int APPROX_LSB_BITS = 4
) (
  input  logic signed [WIDTH-1:0] a_i,
  input  logic signed [WIDTH-1:0] b_i,
  input  logic                    approximate_i,
  output logic signed [WIDTH-1:0] sum_o,
  output logic                    saturated_o
);
  localparam logic signed [WIDTH:0] MAX_EXT = {2'b00, {(WIDTH-1){1'b1}}};
  localparam logic signed [WIDTH:0] MIN_EXT = {2'b11, {(WIDTH-1){1'b0}}};

  logic signed [WIDTH:0] exact_raw;
  logic signed [WIDTH:0] approx_high;
  logic signed [WIDTH:0] approx_raw;
  logic signed [WIDTH:0] selected_raw;

  initial begin
    if (WIDTH < 2 || APPROX_LSB_BITS < 1 || APPROX_LSB_BITS >= WIDTH)
      $error("invalid sat_add_dual_mode parameters");
  end

  always_comb begin
    exact_raw = $signed({a_i[WIDTH-1], a_i}) + $signed({b_i[WIDTH-1], b_i});
    approx_high =
      ($signed({a_i[WIDTH-1], a_i}) >>> APPROX_LSB_BITS) +
      ($signed({b_i[WIDTH-1], b_i}) >>> APPROX_LSB_BITS);
    approx_raw = approx_high <<< APPROX_LSB_BITS;
    approx_raw[APPROX_LSB_BITS-1:0] =
      a_i[APPROX_LSB_BITS-1:0] | b_i[APPROX_LSB_BITS-1:0];
    selected_raw = approximate_i ? approx_raw : exact_raw;

    saturated_o = 1'b0;
    if (selected_raw > MAX_EXT) begin
      sum_o = MAX_EXT[WIDTH-1:0];
      saturated_o = 1'b1;
    end else if (selected_raw < MIN_EXT) begin
      sum_o = MIN_EXT[WIDTH-1:0];
      saturated_o = 1'b1;
    end else begin
      sum_o = selected_raw[WIDTH-1:0];
    end
  end
endmodule

