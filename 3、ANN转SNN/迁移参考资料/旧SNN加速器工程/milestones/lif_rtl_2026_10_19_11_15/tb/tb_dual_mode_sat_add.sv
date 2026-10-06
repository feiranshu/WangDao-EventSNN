`timescale 1ns/1ps

module tb_dual_mode_sat_add;
  localparam int WIDTH = 20;
  localparam int APPROX_BITS = 4;

  logic signed [WIDTH-1:0] a;
  logic signed [WIDTH-1:0] b;
  logic approximate;
  logic signed [WIDTH-1:0] sum;
  logic saturated;
  logic [WIDTH-1:0] vector_a, vector_b, expected_exact, expected_approx;
  logic expected_exact_sat, expected_approx_sat;
  integer fd, parsed, count;
  string vector_file;

  dual_mode_sat_add #(
    .WIDTH(WIDTH),
    .APPROX_LSB_BITS(APPROX_BITS)
  ) dut (
    .a_i(a),
    .b_i(b),
    .approximate_i(approximate),
    .sum_o(sum),
    .saturated_o(saturated)
  );

  initial begin
    if (!$value$plusargs("VECTOR_FILE=%s", vector_file))
      vector_file = "vectors/add_vectors.mem";
    fd = $fopen(vector_file, "r");
    if (fd == 0)
      $fatal(1, "cannot open vector file %s", vector_file);

    count = 0;
    while (!$feof(fd)) begin
      parsed = $fscanf(fd, "%h %h %h %h %h %h\n",
        vector_a, vector_b, expected_exact, expected_exact_sat,
        expected_approx, expected_approx_sat);
      if (parsed == 6) begin
        a = vector_a;
        b = vector_b;
        approximate = 1'b0;
        #1;
        if (sum !== expected_exact || saturated !== expected_exact_sat)
          $fatal(1, "exact mismatch vector %0d: a=%0d b=%0d got=%0d/%0b expected=%0d/%0b",
            count, a, b, sum, saturated, $signed(expected_exact), expected_exact_sat);
        approximate = 1'b1;
        #1;
        if (sum !== expected_approx || saturated !== expected_approx_sat)
          $fatal(1, "approx mismatch vector %0d: a=%0d b=%0d got=%0d/%0b expected=%0d/%0b",
            count, a, b, sum, saturated, $signed(expected_approx), expected_approx_sat);
        count = count + 1;
      end
    end
    $fclose(fd);
    if (count < 100)
      $fatal(1, "insufficient vectors: %0d", count);
    $display("PASS tb_dual_mode_sat_add vectors=%0d", count);
    $finish;
  end
endmodule

