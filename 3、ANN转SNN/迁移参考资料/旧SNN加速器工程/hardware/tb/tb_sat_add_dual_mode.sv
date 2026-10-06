`timescale 1ns/1ps

module tb_sat_add_dual_mode;
  logic signed [7:0] a, b, sum;
  logic approximate, saturated;

  sat_add_dual_mode #(.WIDTH(8), .APPROX_LSB_BITS(2)) dut (
    .a_i(a), .b_i(b), .approximate_i(approximate),
    .sum_o(sum), .saturated_o(saturated)
  );

  task automatic check(
    input logic signed [7:0] test_a,
    input logic signed [7:0] test_b,
    input logic test_approx,
    input logic signed [7:0] expected,
    input logic expected_saturated
  );
    begin
      a = test_a;
      b = test_b;
      approximate = test_approx;
      #1;
      if (sum !== expected || saturated !== expected_saturated) begin
        $display("FAIL a=%0d b=%0d approx=%0b got=%0d/%0b expected=%0d/%0b",
                 a, b, approximate, sum, saturated, expected, expected_saturated);
        $fatal(1);
      end
    end
  endtask

  initial begin
    check(8'sd100, 8'sd20, 1'b0, 8'sd120, 1'b0);
    check(8'sd120, 8'sd20, 1'b0, 8'sd127, 1'b1);
    check(-8'sd120, -8'sd20, 1'b0, -8'sd128, 1'b1);
    check(8'sd7, 8'sd3, 1'b1, 8'sd7, 1'b0);
    check(8'sd120, 8'sd120, 1'b1, 8'sd127, 1'b1);
    $display("PASS tb_sat_add_dual_mode");
    $finish;
  end
endmodule

