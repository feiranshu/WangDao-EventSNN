create_clock -name core_clk -period 6.667 [get_ports clk_i]
set_clock_uncertainty 0.200 [get_clocks core_clk]

