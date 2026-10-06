module state_tile_cache #(
  parameter int ADDR_WIDTH = 20,
  parameter int DATA_WIDTH = 20,
  parameter int LINES = 32
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  invalidate_i,
  input  logic                  request_valid_i,
  output logic                  request_ready_o,
  input  logic [ADDR_WIDTH-1:0] request_addr_i,
  output logic                  response_valid_o,
  input  logic                  response_ready_i,
  output logic                  response_hit_o,
  output logic [DATA_WIDTH-1:0] response_data_o,
  output logic                  miss_valid_o,
  input  logic                  miss_ready_i,
  output logic [ADDR_WIDTH-1:0] miss_addr_o,
  input  logic                  refill_valid_i,
  input  logic [DATA_WIDTH-1:0] refill_data_i,
  input  logic                  update_valid_i,
  input  logic [ADDR_WIDTH-1:0] update_addr_i,
  input  logic [DATA_WIDTH-1:0] update_data_i,
  output logic [63:0]           hit_count_o,
  output logic [63:0]           miss_count_o
);
  localparam int INDEX_WIDTH = $clog2(LINES);
  localparam int TAG_WIDTH = ADDR_WIDTH - INDEX_WIDTH;
  typedef enum logic [1:0] {IDLE, LOOKUP, WAIT_REFILL, RESPOND} state_t;

  logic [DATA_WIDTH-1:0] data_mem [0:LINES-1];
  logic [TAG_WIDTH-1:0] tag_mem [0:LINES-1];
  logic                 valid_mem [0:LINES-1];
  logic [ADDR_WIDTH-1:0] pending_addr;
  logic [INDEX_WIDTH-1:0] pending_index;
  logic [TAG_WIDTH-1:0] pending_tag;
  logic miss_request_sent;
  state_t state;
  integer line;

  initial begin
    if (LINES < 2 || (LINES & (LINES - 1)) != 0 || TAG_WIDTH <= 0)
      $error("state_tile_cache requires a power-of-two line count and a wider address");
  end

  assign request_ready_o = (state == IDLE);
  assign response_valid_o = (state == RESPOND);
  assign miss_valid_o = (state == WAIT_REFILL) && !miss_request_sent;
  assign miss_addr_o = pending_addr;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      state <= IDLE;
      response_hit_o <= 1'b0;
      response_data_o <= '0;
      hit_count_o <= '0;
      miss_count_o <= '0;
      miss_request_sent <= 1'b0;
      for (line = 0; line < LINES; line = line + 1)
        valid_mem[line] <= 1'b0;
    end else begin
      if (invalidate_i) begin
        for (line = 0; line < LINES; line = line + 1)
          valid_mem[line] <= 1'b0;
      end

      if (update_valid_i) begin
        data_mem[update_addr_i[INDEX_WIDTH-1:0]] <= update_data_i;
        tag_mem[update_addr_i[INDEX_WIDTH-1:0]] <= update_addr_i[ADDR_WIDTH-1:INDEX_WIDTH];
        valid_mem[update_addr_i[INDEX_WIDTH-1:0]] <= 1'b1;
      end

      case (state)
        IDLE: begin
          if (request_valid_i) begin
            pending_addr <= request_addr_i;
            pending_index <= request_addr_i[INDEX_WIDTH-1:0];
            pending_tag <= request_addr_i[ADDR_WIDTH-1:INDEX_WIDTH];
            miss_request_sent <= 1'b0;
            state <= LOOKUP;
          end
        end
        LOOKUP: begin
          if (valid_mem[pending_index] && tag_mem[pending_index] == pending_tag) begin
            response_hit_o <= 1'b1;
            response_data_o <= data_mem[pending_index];
            hit_count_o <= hit_count_o + 1'b1;
            state <= RESPOND;
          end else begin
            miss_count_o <= miss_count_o + 1'b1;
            state <= WAIT_REFILL;
          end
        end
        WAIT_REFILL: begin
          if (miss_ready_i)
            miss_request_sent <= 1'b1;
          if (refill_valid_i) begin
            data_mem[pending_index] <= refill_data_i;
            tag_mem[pending_index] <= pending_tag;
            valid_mem[pending_index] <= 1'b1;
            response_hit_o <= 1'b0;
            response_data_o <= refill_data_i;
            miss_request_sent <= 1'b0;
            state <= RESPOND;
          end
        end
        RESPOND: begin
          if (response_ready_i)
            state <= IDLE;
        end
      endcase
    end
  end
endmodule
