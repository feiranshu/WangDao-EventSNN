module axi_lite_regs #(
  parameter int ADDR_WIDTH = 8
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
  input  logic                  s_axi_awvalid,
  output logic                  s_axi_awready,
  input  logic [31:0]           s_axi_wdata,
  input  logic [3:0]            s_axi_wstrb,
  input  logic                  s_axi_wvalid,
  output logic                  s_axi_wready,
  output logic [1:0]            s_axi_bresp,
  output logic                  s_axi_bvalid,
  input  logic                  s_axi_bready,
  input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
  input  logic                  s_axi_arvalid,
  output logic                  s_axi_arready,
  output logic [31:0]           s_axi_rdata,
  output logic [1:0]            s_axi_rresp,
  output logic                  s_axi_rvalid,
  input  logic                  s_axi_rready,
  output logic                  enable_o,
  output logic [1:0]            control_mode_o,
  output logic [31:0]           density_low_o,
  output logic [31:0]           density_high_o,
  output logic                  soft_reset_pulse_o,
  output logic                  clear_errors_pulse_o,
  input  logic                  busy_i,
  input  logic                  fifo_overflow_i,
  input  logic                  illegal_packet_i,
  input  logic                  approximate_active_i,
  input  logic [63:0]           event_count_i,
  input  logic [63:0]           cycle_count_i,
  input  logic [63:0]           cache_hit_count_i,
  input  logic [63:0]           external_read_count_i,
  input  logic [63:0]           saturation_count_i
);
  localparam logic [7:0] REG_CONTROL = 8'h00;
  localparam logic [7:0] REG_STATUS = 8'h04;
  localparam logic [7:0] REG_DENSITY_LOW = 8'h08;
  localparam logic [7:0] REG_DENSITY_HIGH = 8'h0C;
  localparam logic [7:0] REG_EVENT_LO = 8'h10;
  localparam logic [7:0] REG_EVENT_HI = 8'h14;
  localparam logic [7:0] REG_CYCLE_LO = 8'h18;
  localparam logic [7:0] REG_CYCLE_HI = 8'h1C;
  localparam logic [7:0] REG_CACHE_HIT_LO = 8'h20;
  localparam logic [7:0] REG_CACHE_HIT_HI = 8'h24;
  localparam logic [7:0] REG_EXT_READ_LO = 8'h28;
  localparam logic [7:0] REG_EXT_READ_HI = 8'h2C;
  localparam logic [7:0] REG_SAT_LO = 8'h30;
  localparam logic [7:0] REG_SAT_HI = 8'h34;

  logic aw_held, w_held;
  logic [ADDR_WIDTH-1:0] held_awaddr;
  logic [31:0] held_wdata;
  logic [3:0] held_wstrb;
  logic [31:0] control_register;
  integer byte_index;

  assign s_axi_awready = !aw_held && !s_axi_bvalid;
  assign s_axi_wready = !w_held && !s_axi_bvalid;
  assign s_axi_bresp = 2'b00;
  assign s_axi_arready = !s_axi_rvalid;
  assign s_axi_rresp = 2'b00;
  assign enable_o = control_register[4];
  assign control_mode_o = control_register[1:0];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      aw_held <= 1'b0;
      w_held <= 1'b0;
      s_axi_bvalid <= 1'b0;
      held_awaddr <= '0;
      held_wdata <= '0;
      held_wstrb <= '0;
      control_register <= '0;
      density_low_o <= 32'd6000;
      density_high_o <= 32'd14000;
      soft_reset_pulse_o <= 1'b0;
      clear_errors_pulse_o <= 1'b0;
    end else begin
      soft_reset_pulse_o <= 1'b0;
      clear_errors_pulse_o <= 1'b0;
      if (s_axi_awready && s_axi_awvalid) begin
        held_awaddr <= s_axi_awaddr;
        aw_held <= 1'b1;
      end
      if (s_axi_wready && s_axi_wvalid) begin
        held_wdata <= s_axi_wdata;
        held_wstrb <= s_axi_wstrb;
        w_held <= 1'b1;
      end
      if (aw_held && w_held && !s_axi_bvalid) begin
        case (held_awaddr[7:0])
          REG_CONTROL: begin
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
              if (held_wstrb[byte_index])
                control_register[byte_index*8 +: 8] <= held_wdata[byte_index*8 +: 8];
            soft_reset_pulse_o <= held_wdata[2] && held_wstrb[0];
            clear_errors_pulse_o <= held_wdata[3] && held_wstrb[0];
          end
          REG_DENSITY_LOW: begin
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
              if (held_wstrb[byte_index])
                density_low_o[byte_index*8 +: 8] <= held_wdata[byte_index*8 +: 8];
          end
          REG_DENSITY_HIGH: begin
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
              if (held_wstrb[byte_index])
                density_high_o[byte_index*8 +: 8] <= held_wdata[byte_index*8 +: 8];
          end
          default: begin end
        endcase
        aw_held <= 1'b0;
        w_held <= 1'b0;
        s_axi_bvalid <= 1'b1;
      end else if (s_axi_bvalid && s_axi_bready) begin
        s_axi_bvalid <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      s_axi_rvalid <= 1'b0;
      s_axi_rdata <= '0;
    end else begin
      if (s_axi_arready && s_axi_arvalid) begin
        case (s_axi_araddr[7:0])
          REG_CONTROL: s_axi_rdata <= control_register;
          REG_STATUS: s_axi_rdata <= {28'b0, approximate_active_i, illegal_packet_i, fifo_overflow_i, busy_i};
          REG_DENSITY_LOW: s_axi_rdata <= density_low_o;
          REG_DENSITY_HIGH: s_axi_rdata <= density_high_o;
          REG_EVENT_LO: s_axi_rdata <= event_count_i[31:0];
          REG_EVENT_HI: s_axi_rdata <= event_count_i[63:32];
          REG_CYCLE_LO: s_axi_rdata <= cycle_count_i[31:0];
          REG_CYCLE_HI: s_axi_rdata <= cycle_count_i[63:32];
          REG_CACHE_HIT_LO: s_axi_rdata <= cache_hit_count_i[31:0];
          REG_CACHE_HIT_HI: s_axi_rdata <= cache_hit_count_i[63:32];
          REG_EXT_READ_LO: s_axi_rdata <= external_read_count_i[31:0];
          REG_EXT_READ_HI: s_axi_rdata <= external_read_count_i[63:32];
          REG_SAT_LO: s_axi_rdata <= saturation_count_i[31:0];
          REG_SAT_HI: s_axi_rdata <= saturation_count_i[63:32];
          default: s_axi_rdata <= '0;
        endcase
        s_axi_rvalid <= 1'b1;
      end else if (s_axi_rvalid && s_axi_rready) begin
        s_axi_rvalid <= 1'b0;
      end
    end
  end
endmodule

