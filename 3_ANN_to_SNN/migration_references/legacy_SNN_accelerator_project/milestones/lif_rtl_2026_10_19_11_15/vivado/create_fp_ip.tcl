if {![info exists ::env(LIF_RTL_ROOT)]} {
  error "LIF_RTL_ROOT must point to the staged ASCII-only project path"
}

set root [file normalize $::env(LIF_RTL_ROOT)]
set ip_dir [file join $root ip]
set project_dir [file join $root fp_ip_project]
file mkdir $ip_dir

create_project -force fp_ip_project $project_dir -part xczu3eg-sfvc784-1-i
set_property target_language Verilog [current_project]

proc create_arithmetic_ip {name operation precision ip_dir} {
  create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 \
    -module_name $name -dir $ip_dir
  set properties [list \
    CONFIG.Operation_Type $operation \
    CONFIG.A_Precision_Type $precision \
    CONFIG.Flow_Control {NonBlocking} \
    CONFIG.Has_RESULT_TREADY {false} \
    CONFIG.Has_ARESETn {true}]
  if {$operation ne "Compare"} {
    lappend properties CONFIG.Result_Precision_Type $precision
  }
  if {$operation eq "Multiply"} {
    lappend properties CONFIG.C_Mult_Usage {Full_Usage}
  }
  if {$operation eq "Add_Subtract"} {
    lappend properties CONFIG.Add_Sub_Value {Add}
  }
  if {$operation eq "Compare"} {
    lappend properties CONFIG.C_Compare_Operation {Greater_Than_Or_Equal}
  }
  set_property -dict $properties [get_ips $name]
  generate_target all [get_ips $name]
}

foreach specification {
  {fp32_mul_exact Multiply Single}
  {fp32_add_exact Add_Subtract Single}
  {fp32_cmp_ge_exact Compare Single}
  {fp16_mul_exact Multiply Half}
  {fp16_add_exact Add_Subtract Half}
  {fp16_cmp_ge_exact Compare Half}
} {
  create_arithmetic_ip \
    [lindex $specification 0] \
    [lindex $specification 1] \
    [lindex $specification 2] \
    $ip_dir
}

set report [open [file join $ip_dir ip_configuration.txt] w]
foreach ip [lsort [get_ips]] {
  puts $report [format "%s operation=%s precision=%s latency=%s" \
    $ip \
    [get_property CONFIG.Operation_Type $ip] \
    [get_property CONFIG.A_Precision_Type $ip] \
    [get_property CONFIG.C_Latency $ip]]
}
close $report
close_project
exit
