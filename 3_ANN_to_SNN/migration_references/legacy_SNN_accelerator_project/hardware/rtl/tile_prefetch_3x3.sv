module tile_prefetch_3x3 #(
  parameter int X_WIDTH = 6,
  parameter int Y_WIDTH = 5,
  parameter int MAX_TILE_X = 39,
  parameter int MAX_TILE_Y = 31
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               start_i,
  input  logic [X_WIDTH-1:0] center_x_i,
  input  logic [Y_WIDTH-1:0] center_y_i,
  output logic               busy_o,
  output logic               tile_valid_o,
  input  logic               tile_ready_i,
  output logic [X_WIDTH-1:0] tile_x_o,
  output logic [Y_WIDTH-1:0] tile_y_o
);
  logic [3:0] index;
  logic [X_WIDTH-1:0] center_x;
  logic [Y_WIDTH-1:0] center_y;
  integer signed dx, dy, candidate_x, candidate_y;
  logic current_in_range;

  always_comb begin
    dx = (index % 3) - 1;
    dy = (index / 3) - 1;
    candidate_x = $signed({1'b0, center_x}) + dx;
    candidate_y = $signed({1'b0, center_y}) + dy;
    current_in_range =
      candidate_x >= 0 && candidate_x <= MAX_TILE_X &&
      candidate_y >= 0 && candidate_y <= MAX_TILE_Y;
    tile_valid_o = busy_o && current_in_range;
    tile_x_o = candidate_x[X_WIDTH-1:0];
    tile_y_o = candidate_y[Y_WIDTH-1:0];
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      busy_o <= 1'b0;
      index <= '0;
      center_x <= '0;
      center_y <= '0;
    end else if (start_i && !busy_o) begin
      busy_o <= 1'b1;
      index <= '0;
      center_x <= center_x_i;
      center_y <= center_y_i;
    end else if (busy_o) begin
      if (!current_in_range || (tile_valid_o && tile_ready_i)) begin
        if (index == 8) begin
          busy_o <= 1'b0;
          index <= '0;
        end else begin
          index <= index + 1'b1;
        end
      end
    end
  end
endmodule

