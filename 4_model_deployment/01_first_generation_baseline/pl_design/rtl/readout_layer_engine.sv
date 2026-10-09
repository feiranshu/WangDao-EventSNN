`timescale 1ns/1ps

// 96 binary inputs -> 16 signed Q12 mean-membrane logits.
// The four membranes are summed exactly in 26 bits; readout never fires/resets.
module readout_layer_engine #(
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
    output logic [2:0] m_membrane_saturations
);
    typedef enum logic [2:0] {LOAD, PRODUCT, GROUP_SUM, FINAL_SUM,
                             ALIGN, UPDATE, EMIT} state_t;
    state_t state;
    logic [6:0] input_index;
    logic [1:0] time_step;
    logic [3:0] channel;
    logic [1:0] group_index;
    logic [3:0] shifts [0:15];
    logic signed [23:0] membranes [0:15];
    logic signed [23:0] products [0:7];
    logic signed [23:0] partials [0:2];
    logic signed [23:0] group_sum, final_sum, accumulated, current;
    logic signed [23:0] membrane_out;
    logic [3:0] group_saturations;
    logic [1:0] final_saturations;
    logic [7:0] running_saturations;
    logic spike, membrane_saturated;
    integer reset_channel;
    logic signed [25:0] readout_sums [0:15];
    logic [7:0] tree_totals [0:15];
    logic [2:0] membrane_totals [0:15];
    logic [7:0] step_tree_saturations;
    // Internal observation points sampled on trace_valid; no flow-control role.
    logic trace_valid;
    logic signed [25:0] previous_sum, next_sum, mean_value;
    assign next_sum = previous_sum + {{2{membrane_out[23]}}, membrane_out};
    signed_round_shift #(.WIDTH(26), .RIGHT_SHIFT(2)) mean_rounding (
        .value_in(next_sum), .value_out(mean_value)
    );

    initial begin
        if ((APPROX_LOW_BITS != 0) && (APPROX_LOW_BITS != 2))
            $error("Frozen readout layer supports only exact or k=2 LOA");
        $readmemh("layer3_shifts.txt", shifts);
    end

    assign s_ready = !rst && (state == LOAD);
    assign m_valid = !rst && (state == EMIT) && (time_step == 3);
    assign trace_valid = !rst && (state == UPDATE);

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : banks
            localparam BANK_FILE = (lane == 0) ? "layer3_bank0.txt" :
                (lane == 1) ? "layer3_bank1.txt" : (lane == 2) ? "layer3_bank2.txt" :
                (lane == 3) ? "layer3_bank3.txt" : (lane == 4) ? "layer3_bank4.txt" :
                (lane == 5) ? "layer3_bank5.txt" : (lane == 6) ? "layer3_bank6.txt" :
                "layer3_bank7.txt";
            logic features [0:11];
            logic signed [7:0] weights [0:47];
            logic signed [7:0] selected_weight;
            initial $readmemh(BANK_FILE, weights);
            always_comb begin
                selected_weight = features[time_step * 3 + group_index] ?
                    weights[channel * 3 + group_index] : 8'sd0;
            end
            always_ff @(posedge clk) begin
                if (!rst) begin
                    if (s_valid && s_ready && (input_index[2:0] == lane))
                        features[input_index[6:3]] <= s_data;
                    if (state == PRODUCT)
                        products[lane] <= {{16{selected_weight[7]}}, selected_weight};
                end
            end
        end
    endgenerate

    fixed_add_tree #(.INPUTS(8), .APPROX_LOW_BITS(APPROX_LOW_BITS)) group_tree (
        .values(products), .result(group_sum), .saturation_count(group_saturations)
    );
    fixed_add_tree #(.INPUTS(3), .APPROX_LOW_BITS(APPROX_LOW_BITS)) final_tree (
        .values(partials), .result(final_sum), .saturation_count(final_saturations)
    );
    lif_neuron #(.APPROX_LOW_BITS(APPROX_LOW_BITS)) neuron (
        .membrane_in(membranes[channel]), .current_in(current), .fire_enable(1'b0),
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
            m_membrane_saturations <= 0;
            previous_sum <= 0;
            for (reset_channel = 0; reset_channel < 16; reset_channel = reset_channel + 1) begin
                membranes[reset_channel] <= 0;
                readout_sums[reset_channel] <= 0;
                tree_totals[reset_channel] <= 0;
                membrane_totals[reset_channel] <= 0;
            end
        end else begin
            case (state)
                LOAD: if (s_valid && s_ready) begin
                    if (input_index == 95) begin
                        input_index <= 0;
                        time_step <= 0;
                        channel <= 0;
                        group_index <= 0;
                        running_saturations <= 0;
                        for (reset_channel = 0; reset_channel < 16; reset_channel = reset_channel + 1) begin
                            membranes[reset_channel] <= 0;
                            readout_sums[reset_channel] <= 0;
                            tree_totals[reset_channel] <= 0;
                            membrane_totals[reset_channel] <= 0;
                        end
                        state <= PRODUCT;
                    end else input_index <= input_index + 1'b1;
                end
                PRODUCT: state <= GROUP_SUM;
                GROUP_SUM: begin
                    partials[group_index] <= group_sum;
                    running_saturations <= running_saturations + group_saturations;
                    if (group_index == 2) state <= FINAL_SUM;
                    else begin
                        group_index <= group_index + 1'b1;
                        state <= PRODUCT;
                    end
                end
                FINAL_SUM: begin
                    accumulated <= final_sum;
                    step_tree_saturations <= running_saturations + final_saturations;
                    state <= ALIGN;
                end
                ALIGN: begin
                    // Frozen binary-current bounds are checked before ROM export.
                    current <= accumulated <<< shifts[channel];
                    // Register the indexed sum alongside the aligned current.
                    previous_sum <= readout_sums[channel];
                    state <= UPDATE;
                end
                UPDATE: begin
                    membranes[channel] <= membrane_out;
                    readout_sums[channel] <= next_sum;
                    tree_totals[channel] <= tree_totals[channel] + step_tree_saturations;
                    membrane_totals[channel] <= membrane_totals[channel] + membrane_saturated;
                    if (time_step == 3) begin
                        m_data <= {{8{mean_value[23]}}, mean_value[23:0]};
                        m_tree_saturations <= tree_totals[channel] + step_tree_saturations;
                        m_membrane_saturations <= membrane_totals[channel] + membrane_saturated;
                    end
                    state <= EMIT;
                end
                EMIT: if ((time_step != 3) || m_ready) begin
                    group_index <= 0;
                    running_saturations <= 0;
                    if (channel == 15) begin
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
