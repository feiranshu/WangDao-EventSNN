module aer_fifo #(
  parameter int DEPTH = 16
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clear_errors_i,
  input  logic        s_valid_i,
  output logic        s_ready_o,
  input  logic [63:0] s_data_i,
  input  logic        s_last_i,
  output logic        m_valid_o,
  input  logic        m_ready_i,
  output logic [63:0] m_data_o,
  output logic        m_last_o,
  output logic        overflow_sticky_o,
  output logic [$clog2(DEPTH+1)-1:0] level_o
);
  localparam int PTR_WIDTH = $clog2(DEPTH);
  logic [63:0] data_mem [0:DEPTH-1];
  logic        last_mem [0:DEPTH-1];
  logic [PTR_WIDTH-1:0] write_ptr, read_ptr;
  logic push, pop;

  initial begin
    if (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0)
      $error("aer_fifo DEPTH must be a power of two >= 2");
  end

  assign m_valid_o = (level_o != '0);
  assign s_ready_o = (level_o < DEPTH) || (m_valid_o && m_ready_i);
  assign push = s_valid_i && s_ready_o;
  assign pop = m_valid_o && m_ready_i;
  assign m_data_o = data_mem[read_ptr];
  assign m_last_o = last_mem[read_ptr];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      write_ptr <= '0;
      read_ptr <= '0;
      level_o <= '0;
      overflow_sticky_o <= 1'b0;
    end else begin
      if (clear_errors_i)
        overflow_sticky_o <= 1'b0;
      else if (s_valid_i && !s_ready_o)
        overflow_sticky_o <= 1'b1;

      if (push) begin
        data_mem[write_ptr] <= s_data_i;
        last_mem[write_ptr] <= s_last_i;
        write_ptr <= write_ptr + 1'b1;
      end
      if (pop)
        read_ptr <= read_ptr + 1'b1;

      case ({push, pop})
        2'b10: level_o <= level_o + 1'b1;
        2'b01: level_o <= level_o - 1'b1;
        default: level_o <= level_o;
      endcase
    end
  end
endmodule

