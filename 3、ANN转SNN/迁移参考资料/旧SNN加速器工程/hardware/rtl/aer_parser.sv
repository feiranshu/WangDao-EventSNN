module aer_parser (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clear_errors_i,
  input  logic [1:0]  control_mode_i,
  input  logic [31:0] density_low_i,
  input  logic [31:0] density_high_i,
  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic [63:0] in_data_i,
  input  logic        in_last_i,
  output logic        event_valid_o,
  input  logic        event_ready_i,
  output logic [8:0]  event_x_o,
  output logic [7:0]  event_y_o,
  output logic        event_polarity_o,
  output logic [3:0]  event_time_bin_o,
  output logic [15:0] event_slice_id_o,
  output logic        approximate_o,
  output logic        slice_end_pulse_o,
  output logic        illegal_sticky_o,
  output logic [63:0] event_count_o,
  output logic [31:0] last_slice_event_count_o
);
  localparam logic [1:0] TYPE_EVENT = 2'b00;
  localparam logic [1:0] TYPE_SLICE_START = 2'b01;
  localparam logic [1:0] TYPE_SLICE_END = 2'b10;
  localparam logic [1:0] TYPE_RESET = 2'b11;
  localparam logic [1:0] MODE_EXACT = 2'b00;
  localparam logic [1:0] MODE_APPROX = 2'b01;
  localparam logic [1:0] MODE_AUTO = 2'b10;

  logic [1:0] packet_type;
  logic [31:0] slice_event_count;
  logic auto_approximate;
  logic accepted;
  logic malformed;

  assign packet_type = in_data_i[1:0];
  assign event_valid_o = in_valid_i && (packet_type == TYPE_EVENT);
  assign in_ready_o = (packet_type == TYPE_EVENT) ? event_ready_i : 1'b1;
  assign accepted = in_valid_i && in_ready_o;
  assign event_x_o = in_data_i[10:2];
  assign event_y_o = in_data_i[18:11];
  assign event_polarity_o = in_data_i[19];
  assign event_time_bin_o = in_data_i[23:20];
  assign event_slice_id_o = in_data_i[39:24];
  assign malformed =
    (in_data_i[63:40] != '0) ||
    ((packet_type == TYPE_SLICE_END) && !in_last_i) ||
    ((packet_type != TYPE_SLICE_END) && in_last_i) ||
    (control_mode_i == 2'b11);

  always_comb begin
    case (control_mode_i)
      MODE_EXACT: approximate_o = 1'b0;
      MODE_APPROX: approximate_o = 1'b1;
      default: approximate_o = auto_approximate;
    endcase
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      slice_event_count <= '0;
      last_slice_event_count_o <= '0;
      event_count_o <= '0;
      auto_approximate <= 1'b0;
      slice_end_pulse_o <= 1'b0;
      illegal_sticky_o <= 1'b0;
    end else begin
      slice_end_pulse_o <= 1'b0;
      if (clear_errors_i)
        illegal_sticky_o <= 1'b0;
      else if (in_valid_i && malformed)
        illegal_sticky_o <= 1'b1;

      if (accepted) begin
        case (packet_type)
          TYPE_EVENT: begin
            slice_event_count <= slice_event_count + 1'b1;
            event_count_o <= event_count_o + 1'b1;
          end
          TYPE_SLICE_START: slice_event_count <= '0;
          TYPE_SLICE_END: begin
            last_slice_event_count_o <= slice_event_count;
            slice_end_pulse_o <= 1'b1;
            if (control_mode_i == MODE_AUTO) begin
              if (slice_event_count >= density_high_i)
                auto_approximate <= 1'b0;
              else if (slice_event_count <= density_low_i)
                auto_approximate <= 1'b1;
            end
          end
          TYPE_RESET: begin
            slice_event_count <= '0;
            auto_approximate <= 1'b0;
          end
        endcase
      end
    end
  end
endmodule

