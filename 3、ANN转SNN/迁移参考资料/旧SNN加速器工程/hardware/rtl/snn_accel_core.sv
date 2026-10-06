module snn_accel_core #(
  parameter int FIFO_DEPTH = 16,
  parameter int ACC_WIDTH = 20,
  parameter int APPROX_LSB_BITS = 4,
  parameter int LEAK_SHIFT = 3,
  parameter logic signed [ACC_WIDTH-1:0] THRESHOLD = 20'sd512
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         clear_errors_i,
  input  logic [1:0]                   control_mode_i,
  input  logic [31:0]                  density_low_i,
  input  logic [31:0]                  density_high_i,
  input  logic signed [ACC_WIDTH-1:0]  positive_weight_i,
  input  logic signed [ACC_WIDTH-1:0]  negative_weight_i,
  input  logic                         s_axis_tvalid,
  output logic                         s_axis_tready,
  input  logic [63:0]                  s_axis_tdata,
  input  logic                         s_axis_tlast,
  output logic                         spike_valid_o,
  output logic [8:0]                   spike_x_o,
  output logic [7:0]                   spike_y_o,
  output logic [15:0]                  spike_slice_id_o,
  output logic                         approximate_active_o,
  output logic                         fifo_overflow_o,
  output logic                         illegal_packet_o,
  output logic [63:0]                  event_count_o,
  output logic [63:0]                  saturation_count_o
);
  logic fifo_valid, fifo_ready, fifo_last;
  logic [63:0] fifo_data;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo_level;
  logic event_valid, event_ready, event_polarity;
  logic [8:0] event_x;
  logic [7:0] event_y;
  logic [3:0] event_tbin;
  logic [15:0] event_slice_id;
  logic slice_end;
  logic [31:0] last_slice_count;
  logic signed [ACC_WIDTH-1:0] synaptic_value;
  logic signed [ACC_WIDTH-1:0] membrane;
  logic neuron_spike, neuron_saturated;
  logic [8:0] pending_x;
  logic [7:0] pending_y;
  logic [15:0] pending_slice_id;

  aer_fifo #(.DEPTH(FIFO_DEPTH)) input_fifo (
    .clk_i, .rst_ni, .clear_errors_i,
    .s_valid_i(s_axis_tvalid), .s_ready_o(s_axis_tready),
    .s_data_i(s_axis_tdata), .s_last_i(s_axis_tlast),
    .m_valid_o(fifo_valid), .m_ready_i(fifo_ready),
    .m_data_o(fifo_data), .m_last_o(fifo_last),
    .overflow_sticky_o(fifo_overflow_o), .level_o(fifo_level)
  );

  aer_parser parser (
    .clk_i, .rst_ni, .clear_errors_i, .control_mode_i,
    .density_low_i, .density_high_i,
    .in_valid_i(fifo_valid), .in_ready_o(fifo_ready),
    .in_data_i(fifo_data), .in_last_i(fifo_last),
    .event_valid_o(event_valid), .event_ready_i(event_ready),
    .event_x_o(event_x), .event_y_o(event_y),
    .event_polarity_o(event_polarity), .event_time_bin_o(event_tbin),
    .event_slice_id_o(event_slice_id), .approximate_o(approximate_active_o),
    .slice_end_pulse_o(slice_end), .illegal_sticky_o(illegal_packet_o),
    .event_count_o(event_count_o), .last_slice_event_count_o(last_slice_count)
  );

  assign event_ready = 1'b1;
  assign synaptic_value = event_polarity ? positive_weight_i : negative_weight_i;

  lif_neuron #(
    .WIDTH(ACC_WIDTH), .APPROX_LSB_BITS(APPROX_LSB_BITS),
    .LEAK_SHIFT(LEAK_SHIFT), .THRESHOLD(THRESHOLD)
  ) neuron (
    .clk_i, .rst_ni, .enable_i(event_valid && event_ready),
    .clear_i(slice_end), .approximate_i(approximate_active_o),
    .synaptic_i(synaptic_value), .membrane_o(membrane),
    .spike_o(neuron_spike), .saturated_o(neuron_saturated)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      spike_valid_o <= 1'b0;
      spike_x_o <= '0;
      spike_y_o <= '0;
      spike_slice_id_o <= '0;
      saturation_count_o <= '0;
      pending_x <= '0;
      pending_y <= '0;
      pending_slice_id <= '0;
    end else begin
      spike_valid_o <= neuron_spike;
      if (event_valid && event_ready) begin
        pending_x <= event_x;
        pending_y <= event_y;
        pending_slice_id <= event_slice_id;
      end
      if (neuron_spike) begin
        spike_x_o <= pending_x;
        spike_y_o <= pending_y;
        spike_slice_id_o <= pending_slice_id;
      end
      if (neuron_saturated)
        saturation_count_o <= saturation_count_o + 1'b1;
    end
  end
endmodule
