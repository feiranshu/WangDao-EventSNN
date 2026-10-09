package snn_pkg;
  typedef enum logic [1:0] {
    AER_EVENT       = 2'b00,
    AER_SLICE_START = 2'b01,
    AER_SLICE_END   = 2'b10,
    AER_RESET       = 2'b11
  } aer_packet_type_t;

  typedef enum logic [1:0] {
    CONTROL_EXACT  = 2'b00,
    CONTROL_APPROX = 2'b01,
    CONTROL_AUTO   = 2'b10
  } control_mode_t;

  localparam int AER_TYPE_LSB = 0;
  localparam int AER_X_LSB = 2;
  localparam int AER_Y_LSB = 11;
  localparam int AER_POLARITY_BIT = 19;
  localparam int AER_TBIN_LSB = 20;
  localparam int AER_SLICE_ID_LSB = 24;
  localparam int AER_RESERVED_LSB = 40;
endpackage

