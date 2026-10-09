module lif_neuron #(
  parameter int WIDTH = 20,
  parameter int APPROX_LSB_BITS = 4,
  parameter int LEAK_SHIFT = 3,
  parameter logic signed [WIDTH-1:0] THRESHOLD = 20'sd512,
  parameter logic signed [WIDTH-1:0] RESET_POTENTIAL = '0
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  input  logic                    enable_i,
  input  logic                    clear_i,
  input  logic                    approximate_i,
  input  logic signed [WIDTH-1:0] synaptic_i,
  output logic signed [WIDTH-1:0] membrane_o,
  output logic                    spike_o,
  output logic                    saturated_o
);
  logic signed [WIDTH-1:0] leaked_membrane;
  logic signed [WIDTH-1:0] updated_membrane;
  logic add_saturated;

  always_comb begin
    if (LEAK_SHIFT == 0)
      leaked_membrane = membrane_o;
    else
      leaked_membrane = membrane_o - (membrane_o >>> LEAK_SHIFT);
  end

  sat_add_dual_mode #(
    .WIDTH(WIDTH),
    .APPROX_LSB_BITS(APPROX_LSB_BITS)
  ) add_i (
    .a_i(leaked_membrane),
    .b_i(synaptic_i),
    .approximate_i(approximate_i),
    .sum_o(updated_membrane),
    .saturated_o(add_saturated)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni || clear_i) begin
      membrane_o <= RESET_POTENTIAL;
      spike_o <= 1'b0;
      saturated_o <= 1'b0;
    end else begin
      spike_o <= 1'b0;
      saturated_o <= 1'b0;
      if (enable_i) begin
        saturated_o <= add_saturated;
        if (updated_membrane >= THRESHOLD) begin
          membrane_o <= RESET_POTENTIAL;
          spike_o <= 1'b1;
        end else begin
          membrane_o <= updated_membrane;
        end
      end
    end
  end
endmodule

