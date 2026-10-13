set script_dir [file dirname [file normalize [info script]]]
set project_root [file normalize [file join $script_dir ../..]]
set rtl_dir [file join $project_root hardware rtl]
set result_dir [file join $project_root results generated vivado]
file mkdir $result_dir

read_verilog -sv [glob [file join $rtl_dir *.sv]]
read_xdc [file join $project_root hardware constraints clock.xdc]
synth_design -top snn_accel_core -part xczu3eg-sfvc784-1-i -flatten_hierarchy rebuilt
opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $result_dir utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
  -file [file join $result_dir timing_summary.rpt]
report_power -file [file join $result_dir power.rpt]
write_checkpoint -force [file join $result_dir snn_accel_core_routed.dcp]

