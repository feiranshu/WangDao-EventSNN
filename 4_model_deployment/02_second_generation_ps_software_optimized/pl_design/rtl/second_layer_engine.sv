`timescale 1ns/1ps

// One transaction: 192 binary spikes -> 96 membrane/spike words (time, channel).
// The frozen second layer uses exact synapses in both modes. Binary products
// select a signed weight or zero; no zero-skipping schedule is used.
module second_layer_engine #(
    parameter integer APPROX_LOW_BITS = 0
) (
    input logic clk,
    input logic rst,
    input logic s_data,
    input logic s_valid,
    output logic s_ready,
    output logic [31:0] m_data,
    output logic m_valid,
    input logic m_ready,
    output logic [7:0] m_tree_saturations,
    output logic m_membrane_saturated
);
    typedef enum logic [2:0] {LOAD, PRODUCT, GROUP_SUM, FINAL_SUM,
                             ALIGN, UPDATE, EMIT} state_t;
    state_t state;
    logic [7:0] input_index;
    logic [1:0] time_step;
    logic [4:0] channel;
    logic [2:0] group_index;
    logic [3:0] shifts [0:23];
    logic signed [23:0] membranes [0:23];
    logic signed [23:0] products [0:7];
    logic signed [23:0] partials [0:5];
    logic signed [23:0] group_sum, final_sum, accumulated, current;
    logic signed [23:0] membrane_out;
    logic [3:0] group_saturations;
    logic [2:0] final_saturations;
    logic [7:0] running_saturations;
    logic spike, membrane_saturated;
    integer reset_channel;

    initial begin
        if ((APPROX_LOW_BITS != 0) && (APPROX_LOW_BITS != 2))
            $error("Frozen second layer supports only exact or k=2 LOA");
        $readmemh("layer2_shifts.txt", shifts);
    end

    assign s_ready = !rst && (state == LOAD);
    assign m_valid = !rst && (state == EMIT);

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : banks
            localparam BANK_FILE = (lane == 0) ? "layer2_bank0.txt" :
                (lane == 1) ? "layer2_bank1.txt" : (lane == 2) ? "layer2_bank2.txt" :
                (lane == 3) ? "layer2_bank3.txt" : (lane == 4) ? "layer2_bank4.txt" :
                (lane == 5) ? "layer2_bank5.txt" : (lane == 6) ? "layer2_bank6.txt" :
                "layer2_bank7.txt";
            logic features [0:23];
            logic signed [7:0] weights [0:143];
            logic signed [7:0] selected_weight;
            initial $readmemh(BANK_FILE, weights);
            always_comb begin
                selected_weight = features[time_step * 6 + group_index] ?
                    weights[channel * 6 + group_index] : 8'sd0;
            end
            always_ff @(posedge clk) begin
                if (!rst) begin
                    if (s_valid && s_ready && (input_index[2:0] == lane))
                        features[input_index[7:3]] <= s_data;
                    if (state == PRODUCT)
                        products[lane] <= {{16{selected_weight[7]}}, selected_weight};
                end
            end
        end
    endgenerate

    fixed_add_tree #(.INPUTS(8), .APPROX_LOW_BITS(0)) group_tree (
        .values(products), .result(group_sum), .saturation_count(group_saturations)
    );
    fixed_add_tree #(.INPUTS(6), .APPROX_LOW_BITS(0)) final_tree (
        .values(partials), .result(final_sum), .saturation_count(final_saturations)
    );
    lif_neuron #(.APPROX_LOW_BITS(APPROX_LOW_BITS)) neuron (
        .membrane_in(membranes[channel]), .current_in(current), .fire_enable(1'b1),
        .membrane_out(membrane_out), .spike(spike), .saturated(membrane_saturated)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= LOAD;
            input_index <= 0;
            time_step <= 0;
            channel <= 0;
            group_index <= 0;
            running_saturations <= 0;
            m_data <= 0;
            m_tree_saturations <= 0;
            m_membrane_saturated <= 0;
            for (reset_channel = 0; reset_channel < 24; reset_channel = reset_channel + 1)
                membranes[reset_channel] <= 0;
        end else begin
            case (state)
                LOAD: if (s_valid && s_ready) begin
                    if (input_index == 191) begin
                        input_index <= 0;
                        time_step <= 0;
                        channel <= 0;
                        group_index <= 0;
                        running_saturations <= 0;
                        for (reset_channel = 0; reset_channel < 24; reset_channel = reset_channel + 1)
                            membranes[reset_channel] <= 0;
                        state <= PRODUCT;
                    end else input_index <= input_index + 1'b1;
                end
                PRODUCT: state <= GROUP_SUM;
                GROUP_SUM: begin
                    partials[group_index] <= group_sum;
                    running_saturations <= running_saturations + group_saturations;
                    if (group_index == 5) state <= FINAL_SUM;
                    else begin
                        group_index <= group_index + 1'b1;
                        state <= PRODUCT;
                    end
                end
                FINAL_SUM: begin
                    accumulated <= final_sum;
                    m_tree_saturations <= running_saturations + final_saturations;
                    state <= ALIGN;
                end
                ALIGN: begin
                    // Frozen binary-current bounds are checked before ROM export.
                    current <= accumulated <<< shifts[channel];
                    state <= UPDATE;
                end
                UPDATE: begin
                    membranes[channel] <= membrane_out;
                    m_data <= {7'b0, spike, membrane_out};
                    m_membrane_saturated <= membrane_saturated;
                    state <= EMIT;
                end
                EMIT: if (m_ready) begin
                    group_index <= 0;
                    running_saturations <= 0;
                    if (channel == 23) begin
                        channel <= 0;
                        if (time_step == 3) state <= LOAD;
                        else begin
                            time_step <= time_step + 1'b1;
                            state <= PRODUCT;
                        end
                    end else begin
                        channel <= channel + 1'b1;
                        state <= PRODUCT;
                    end
                end
                default: state <= LOAD;
            endcase
        end
    end
endmodule
