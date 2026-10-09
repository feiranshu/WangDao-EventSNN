`timescale 1ns/1ps

// One transaction: 512 input bytes -> 192 membrane/spike words (time, channel).
// Every tree boundary preserves the frozen 128-input balanced reduction order.
module first_layer_engine #(
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
    output logic m_membrane_saturated
);
    typedef enum logic [2:0] {LOAD, PRODUCT, GROUP_SUM, FINAL_SUM,
                             ALIGN, UPDATE, EMIT} state_t;
    state_t state;
    logic [8:0] input_index;
    logic [1:0] time_step;
    logic [5:0] channel;
    logic [3:0] group_index;
    logic [63:0] group_nonzero;
    logic [1:0] shifts [0:47];
    logic signed [23:0] membranes [0:47];
    logic signed [23:0] products [0:7];
    logic signed [23:0] partials [0:15];
    logic signed [23:0] group_sum, final_sum, accumulated, current;
    logic signed [23:0] shift_one, shift_two, membrane_out;
    logic [3:0] group_saturations;
    logic [4:0] final_saturations;
    logic [7:0] running_saturations;
    logic spike, membrane_saturated;
    integer reset_channel;

    initial begin
        if ((APPROX_LOW_BITS != 0) && (APPROX_LOW_BITS != 2))
            $error("Frozen first layer supports only exact or k=2 LOA");
        $readmemh("layer1_shifts.txt", shifts);
    end

    assign s_ready = !rst && (state == LOAD);
    assign m_valid = !rst && (state == EMIT);

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : banks
            localparam BANK_FILE = (lane == 0) ? "layer1_bank0.txt" :
                (lane == 1) ? "layer1_bank1.txt" : (lane == 2) ? "layer1_bank2.txt" :
                (lane == 3) ? "layer1_bank3.txt" : (lane == 4) ? "layer1_bank4.txt" :
                (lane == 5) ? "layer1_bank5.txt" : (lane == 6) ? "layer1_bank6.txt" :
                "layer1_bank7.txt";
            logic [7:0] features [0:63];
            logic signed [7:0] weights [0:767];
            logic signed [8:0] positive_feature;
            logic signed [16:0] product_wide;
            initial $readmemh(BANK_FILE, weights);
            always_comb begin
                positive_feature = $signed({1'b0, features[{time_step, group_index}]});
                product_wide = positive_feature * weights[{channel, group_index}];
            end
            always_ff @(posedge clk) begin
                if (!rst) begin
                    if (s_valid && s_ready && (input_index[2:0] == lane))
                        features[input_index[8:3]] <= s_data;
                    if (state == PRODUCT)
                        products[lane] <= {{7{product_wide[16]}}, product_wide};
                end
            end
        end
    endgenerate

    fixed_add_tree #(.INPUTS(8), .APPROX_LOW_BITS(APPROX_LOW_BITS)) group_tree (
        .values(products), .result(group_sum), .saturation_count(group_saturations)
    );
    fixed_add_tree #(.INPUTS(16), .APPROX_LOW_BITS(APPROX_LOW_BITS)) final_tree (
        .values(partials), .result(final_sum), .saturation_count(final_saturations)
    );
    signed_round_shift #(.RIGHT_SHIFT(1)) align_one (
        .value_in(accumulated), .value_out(shift_one)
    );
    signed_round_shift #(.RIGHT_SHIFT(2)) align_two (
        .value_in(accumulated), .value_out(shift_two)
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
            group_nonzero <= 0;
            running_saturations <= 0;
            m_data <= 0;
            m_tree_saturations <= 0;
            m_membrane_saturated <= 0;
            for (reset_channel = 0; reset_channel < 48; reset_channel = reset_channel + 1)
                membranes[reset_channel] <= 0;
        end else begin
            case (state)
                LOAD: if (s_valid && s_ready) begin
                    // Each group's first accepted byte overwrites the previous patch's flag.
                    if (SKIP_ZERO_GROUPS) begin
                        if (input_index[2:0] == 0)
                            group_nonzero[input_index[8:3]] <= (s_data != 0);
                        else
                            group_nonzero[input_index[8:3]] <= group_nonzero[input_index[8:3]] | (s_data != 0);
                    end
                    if (input_index == 511) begin
                        input_index <= 0;
                        time_step <= 0;
                        channel <= 0;
                        group_index <= 0;
                        running_saturations <= 0;
                        for (reset_channel = 0; reset_channel < 48; reset_channel = reset_channel + 1)
                            membranes[reset_channel] <= 0;
                        state <= PRODUCT;
                    end else input_index <= input_index + 1'b1;
                end
                PRODUCT: begin
                    if (SKIP_ZERO_GROUPS && !group_nonzero[{time_step, group_index}]) begin
                        // Preserve the slot and final-tree order; zero subtrees have zero saturations.
                        partials[group_index] <= 0;
                        if (group_index == 15) state <= FINAL_SUM;
                        else group_index <= group_index + 1'b1;
                    end else state <= GROUP_SUM;
                end
                GROUP_SUM: begin
                    partials[group_index] <= group_sum;
                    running_saturations <= running_saturations + group_saturations;
                    if (group_index == 15) state <= FINAL_SUM;
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
                    case (shifts[channel])
                        0: current <= accumulated;
                        1: current <= shift_one;
                        2: current <= shift_two;
                        default: current <= 'x;
                    endcase
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
                    if (channel == 47) begin
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
