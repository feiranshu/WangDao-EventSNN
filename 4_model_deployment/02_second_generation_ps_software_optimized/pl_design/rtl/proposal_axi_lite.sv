`timescale 1ns/1ps

// PS-facing first integration wrapper for the Proposal Engine.
//
// Address map (32-bit AXI4-Lite, byte addressing):
//   0x000  control: bit 0 starts one patch, bit 1 clears the error flag
//   0x004  status:  bit 0 busy, bit 1 done, bit 2 error, bits 4:3 state
//   0x008  configuration: APPROX_LOW_BITS and SKIP_ZERO_GROUPS
//   0x100-0x2ff  512 input bytes, written as 32-bit little-endian words
//   0x900-0x93f  sixteen 32-bit output logits
//
// This wrapper deliberately keeps the first board path simple.  It does not
// contain a DMA engine or an event-camera interface; Linux writes one complete
// patch, starts the engine, and reads the sixteen results.
module proposal_axi_lite #(
    parameter integer APPROX_LOW_BITS = 0,
    parameter bit SKIP_ZERO_GROUPS = 0,
    parameter integer ADDR_WIDTH = 12
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 99999001" *)
    input logic s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aresetn, POLARITY ACTIVE_LOW" *)
    input logic s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, FREQ_HZ 99999001" *)
    input logic [ADDR_WIDTH-1:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input logic s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input logic [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input logic [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input logic s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input logic s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input logic [ADDR_WIDTH-1:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input logic s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input logic s_axi_rready
);
    localparam logic [ADDR_WIDTH-1:0] CONTROL_ADDR = 12'h000;
    localparam logic [ADDR_WIDTH-1:0] STATUS_ADDR = 12'h004;
    localparam logic [ADDR_WIDTH-1:0] CONFIG_ADDR = 12'h008;
    localparam logic [ADDR_WIDTH-1:0] INPUT_BASE = 12'h100;
    localparam logic [ADDR_WIDTH-1:0] INPUT_END = 12'h300;
    localparam logic [ADDR_WIDTH-1:0] OUTPUT_BASE = 12'h900;
    localparam logic [ADDR_WIDTH-1:0] OUTPUT_END = 12'h940;

    typedef enum logic [1:0] {IDLE = 2'd0, ENGINE_RESET = 2'd1,
                              FEED = 2'd2, DRAIN = 2'd3} state_t;
    state_t state;
    logic busy, done, error_flag;
    logic [8:0] feed_count;
    logic [3:0] output_count;
    logic [7:0] input_mem [0:511];
    logic [31:0] output_mem [0:15];

    logic aw_hold, w_hold;
    logic [ADDR_WIDTH-1:0] awaddr_hold;
    logic [31:0] wdata_hold;
    logic [3:0] wstrb_hold;
    logic write_commit;
    logic [ADDR_WIDTH-1:0] write_addr;
    logic [31:0] write_data;
    logic [3:0] write_strb;
    logic read_fire;
    logic [31:0] read_data_next;
    logic engine_rst;
    logic engine_s_ready;
    logic engine_m_valid;
    logic [7:0] engine_tree_saturations;
    logic [2:0] engine_membrane_saturations;
    logic [7:0] engine_s_data;
    logic [31:0] engine_m_data;

    integer reset_index;
    integer byte_index;

    assign s_axi_awready = !aw_hold && !s_axi_bvalid;
    assign s_axi_wready = !w_hold && !s_axi_bvalid;
    wire aw_fire = s_axi_awvalid && s_axi_awready;
    wire w_fire = s_axi_wvalid && s_axi_wready;
    assign write_commit = !s_axi_bvalid &&
        (aw_hold || aw_fire) && (w_hold || w_fire);
    assign write_addr = aw_hold ? awaddr_hold : s_axi_awaddr;
    assign write_data = w_hold ? wdata_hold : s_axi_wdata;
    assign write_strb = w_hold ? wstrb_hold : s_axi_wstrb;

    assign s_axi_bresp = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign read_fire = s_axi_arvalid && s_axi_arready;
    assign s_axi_rresp = 2'b00;

    assign engine_rst = !s_axi_aresetn || (state == ENGINE_RESET);
    assign engine_s_data = input_mem[feed_count];

    proposal_engine #(
        .APPROX_LOW_BITS(APPROX_LOW_BITS),
        .SKIP_ZERO_GROUPS(SKIP_ZERO_GROUPS)
    ) engine (
        .clk(s_axi_aclk),
        .rst(engine_rst),
        .s_data(engine_s_data),
        .s_valid(state == FEED),
        .s_ready(engine_s_ready),
        .m_data(engine_m_data),
        .m_valid(engine_m_valid),
        .m_ready(state == DRAIN),
        .m_tree_saturations(engine_tree_saturations),
        .m_membrane_saturations(engine_membrane_saturations)
    );

    always_comb begin
        read_data_next = 32'h00000000;
        if (s_axi_araddr == STATUS_ADDR) begin
            read_data_next = {24'h0, state, error_flag, done, busy};
        end else if (s_axi_araddr == CONFIG_ADDR) begin
            read_data_next = {29'h0, SKIP_ZERO_GROUPS, APPROX_LOW_BITS[1:0]};
        end else if ((s_axi_araddr >= OUTPUT_BASE) &&
                     (s_axi_araddr < OUTPUT_END) &&
                     (s_axi_araddr[1:0] == 2'b00)) begin
            read_data_next = output_mem[(s_axi_araddr - OUTPUT_BASE) >> 2];
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_hold <= 1'b0;
            w_hold <= 1'b0;
            awaddr_hold <= '0;
            wdata_hold <= '0;
            wstrb_hold <= '0;
            s_axi_bvalid <= 1'b0;
        end else begin
            if (aw_fire) begin
                aw_hold <= 1'b1;
                awaddr_hold <= s_axi_awaddr;
            end
            if (w_fire) begin
                w_hold <= 1'b1;
                wdata_hold <= s_axi_wdata;
                wstrb_hold <= s_axi_wstrb;
            end
            if (write_commit) begin
                aw_hold <= 1'b0;
                w_hold <= 1'b0;
                s_axi_bvalid <= 1'b1;
                if ((write_addr >= INPUT_BASE) &&
                    (write_addr < INPUT_END) &&
                    (write_addr[1:0] == 2'b00)) begin
                    for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                        if (write_strb[byte_index])
                            input_mem[(write_addr - INPUT_BASE) + byte_index] <=
                                write_data[(byte_index * 8) +: 8];
                end else if ((write_addr == CONTROL_ADDR) && write_strb[0]) begin
                    // Control bits are handled by the engine state machine.
                end else if ((write_addr != CONTROL_ADDR) &&
                             (write_addr != STATUS_ADDR) &&
                             (write_addr != CONFIG_ADDR)) begin
                    // Invalid-write reporting is handled by the engine state machine.
                end
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= 32'h0;
        end else begin
            if (read_fire) begin
                s_axi_rdata <= read_data_next;
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error_flag <= 1'b0;
            feed_count <= 9'd0;
            output_count <= 4'd0;
            for (reset_index = 0; reset_index < 16; reset_index = reset_index + 1)
                output_mem[reset_index] <= 32'h0;
        end else begin
            if (write_commit && (write_addr == CONTROL_ADDR) && write_strb[0]) begin
                if (write_data[1]) error_flag <= 1'b0;
                if (write_data[0]) begin
                    if (state == IDLE) begin
                        state <= ENGINE_RESET;
                        busy <= 1'b1;
                        done <= 1'b0;
                        feed_count <= 9'd0;
                        output_count <= 4'd0;
                    end else begin
                        error_flag <= 1'b1;
                    end
                end
            end else if (write_commit && (write_addr != CONTROL_ADDR) &&
                         (write_addr != STATUS_ADDR) &&
                         (write_addr != CONFIG_ADDR) &&
                         !((write_addr >= INPUT_BASE) &&
                           (write_addr < INPUT_END) &&
                           (write_addr[1:0] == 2'b00))) begin
                error_flag <= 1'b1;
            end

            case (state)
                IDLE: begin
                    // Wait for a control write that starts the next patch.
                end
                ENGINE_RESET: begin
                    state <= FEED;
                    feed_count <= 9'd0;
                end
                FEED: begin
                    if (engine_s_ready) begin
                        if (feed_count == 9'd511) begin
                            feed_count <= 9'd0;
                            output_count <= 4'd0;
                            state <= DRAIN;
                        end else begin
                            feed_count <= feed_count + 1'b1;
                        end
                    end
                end
                DRAIN: begin
                    if (engine_m_valid) begin
                        output_mem[output_count] <= engine_m_data;
                        if (output_count == 4'd15) begin
                            output_count <= 4'd0;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            output_count <= output_count + 1'b1;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
