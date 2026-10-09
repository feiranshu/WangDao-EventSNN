if {![info exists ::env(LIF_RTL_ROOT)]} {
  error "LIF_RTL_ROOT must point to the staged ASCII-only project path"
}
if {$argc != 2} {
  error "usage: vivado -source synth_common.tcl -tclargs <top> <report_name>"
}

set root [file normalize $::env(LIF_RTL_ROOT)]
set top [lindex $argv 0]
set report_name [lindex $argv 1]
set rtl_dir [file join $root rtl]
set report_dir [file join $root reports $report_name]
file mkdir $report_dir

set_param general.maxThreads 1
read_verilog -sv [list \
  [file join $rtl_dir dual_mode_sat_add.sv] \
  [file join $rtl_dir lif_neuron_dual_mode.sv] \
  [file join $rtl_dir int8_mac_accumulator.sv] \
  [file join $rtl_dir pot_shift_accumulator.sv] \
  [file join $rtl_dir int8_lif_baseline_top.sv] \
  [file join $rtl_dir pot_lif_pe_top.sv]]
read_xdc [file join $root constraints clock.xdc]

synth_design -top $top -part xczu3eg-sfvc784-1-i -mode out_of_context -flatten_hierarchy rebuilt
write_checkpoint -force [file join $report_dir post_synth.dcp]
report_utilization -hierarchical -file [file join $report_dir utilization_synth.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
  -file [file join $report_dir timing_synth.rpt]

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $report_dir post_route.dcp]

report_utilization -hierarchical -file [file join $report_dir utilization_route.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
  -file [file join $report_dir timing_route.rpt]
report_power -file [file join $report_dir power_route.rpt]
report_drc -file [file join $report_dir drc_route.rpt]

set lut_count [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}]]
set ff_count [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set dsp_count [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP*}]]
set bram_count [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ BLOCKRAM*}]]
set timing_path [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK $timing_path]
set period 6.667
set data_delay [expr {$period - $wns}]
if {$data_delay > 0} {
  set fmax_mhz [expr {1000.0 / $data_delay}]
} else {
  set fmax_mhz 0.0
}

set metrics [open [file join $report_dir metrics.json] w]
puts $metrics "{"
puts $metrics "  \"top\": \"$top\","
puts $metrics "  \"part\": \"xczu3eg-sfvc784-1-i\","
puts $metrics "  \"target_clock_mhz\": 150.0,"
puts $metrics "  \"wns_ns\": $wns,"
puts $metrics "  \"estimated_fmax_mhz\": $fmax_mhz,"
puts $metrics "  \"lut\": $lut_count,"
puts $metrics "  \"ff\": $ff_count,"
puts $metrics "  \"dsp\": $dsp_count,"
puts $metrics "  \"bram\": $bram_count"
puts $metrics "}"
close $metrics

exit
