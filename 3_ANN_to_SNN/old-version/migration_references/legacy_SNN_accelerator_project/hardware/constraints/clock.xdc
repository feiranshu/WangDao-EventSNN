create_clock -period 6.667 -name pl_clk [get_ports clk_i]
set_clock_uncertainty 0.200 [get_clocks pl_clk]

