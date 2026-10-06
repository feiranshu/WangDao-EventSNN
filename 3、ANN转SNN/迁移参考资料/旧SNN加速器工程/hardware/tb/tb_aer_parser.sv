`timescale 1ns/1ps

module tb_aer_parser;
  logic clk = 0;
  logic rst_n = 0;
  logic in_valid = 0;
  logic in_ready;
  logic [63:0] in_data = 0;
  logic in_last = 0;
  logic event_valid;
  logic approximate;
  logic slice_end;
  logic illegal;
  logic [63:0] event_count;
  logic [31:0] last_slice_count;
  logic [8:0] x;
  logic [7:0] y;
  logic p;
  logic [3:0] tbin;
  logic [15:0] slice_id;

  always #5 clk = ~clk;

  aer_parser dut (
    .clk_i(clk), .rst_ni(rst_n), .clear_errors_i(1'b0),
    .control_mode_i(2'b10), .density_low_i(32'd2), .density_high_i(32'd4),
    .in_valid_i(in_valid), .in_ready_o(in_ready), .in_data_i(in_data), .in_last_i(in_last),
    .event_valid_o(event_valid), .event_ready_i(1'b1),
    .event_x_o(x), .event_y_o(y), .event_polarity_o(p),
    .event_time_bin_o(tbin), .event_slice_id_o(slice_id),
    .approximate_o(approximate), .slice_end_pulse_o(slice_end),
    .illegal_sticky_o(illegal), .event_count_o(event_count),
    .last_slice_event_count_o(last_slice_count)
  );

  task automatic send(input logic [63:0] word, input logic last);
    begin
      @(negedge clk);
      in_data = word;
      in_last = last;
      in_valid = 1;
      @(negedge clk);
      in_valid = 0;
      in_last = 0;
    end
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1;
    send((64'd1 << 24) | 64'd1, 1'b0); // slice start, id=1
    send((64'd1 << 24) | (64'd20 << 11) | (64'd10 << 2), 1'b0);
    send((64'd1 << 24) | (64'd21 << 11) | (64'd11 << 2), 1'b0);
    send((64'd1 << 24) | 64'd2, 1'b1); // slice end
    @(negedge clk);
    if (event_count !== 2 || last_slice_count !== 2 || approximate !== 1'b1 || illegal)
      $fatal(1, "parser counters or hysteresis failed");
    $display("PASS tb_aer_parser");
    $finish;
  end
endmodule
