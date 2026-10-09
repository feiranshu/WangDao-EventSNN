`timescale 1ns/1ps

// One patch in flight: 512 feature bytes -> 16 signed Q12 logits.
// The readout consumes all 96 second-layer spikes in time/channel order.
module proposal_engine #(
    parameter integer APPROX_LOW_BITS = 0,
    parameter bit SKIP_ZERO_GROUPS = 0
) (
    input logic clk,
    input logic rst,
    input logic [7:0] s_data,
    input logic s_valid,
    output logic s_ready,
    output logic [31:0] m_data,
    output logic m_valid,
    input logic m_ready,
    output logic [7:0] m_tree_saturations,
    output logic [2:0] m_membrane_saturations
);
    logic draining;
    logic [8:0] input_count;
    logic [3:0] output_count;
    logic first_ready, link_valid, link_ready;
    logic [31:0] link_data;
    logic [7:0] link_tree_saturations;
    logic link_membrane_saturated;

    assign s_ready = !rst && !draining && first_ready;
    two_layer_engine #(.APPROX_LOW_BITS(APPROX_LOW_BITS), .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)) hidden (
        .clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid && !draining),
        .s_ready(first_ready), .m_data(link_data), .m_valid(link_valid), .m_ready(link_ready),
        .m_tree_saturations(link_tree_saturations), .m_membrane_saturated(link_membrane_saturated)
    );
    readout_layer_engine #(.APPROX_LOW_BITS(APPROX_LOW_BITS)) readout (
        .clk(clk), .rst(rst), .s_data(link_data[24]), .s_valid(link_valid), .s_ready(link_ready),
        .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready),
        .m_tree_saturations(m_tree_saturations), .m_membrane_saturations(m_membrane_saturations)
    );
    always_ff @(posedge clk) begin
        if (rst) begin
            draining <= 0;
            input_count <= 0;
            output_count <= 0;
        end else begin
            if (s_valid && s_ready) begin
                if (input_count == 511) begin
                    input_count <= 0;
                    draining <= 1;
                end else input_count <= input_count + 1'b1;
            end
            if (m_valid && m_ready) begin
                if (output_count == 15) begin
                    output_count <= 0;
                    draining <= 0;
                end else output_count <= output_count + 1'b1;
            end
        end
    end
endmodule
