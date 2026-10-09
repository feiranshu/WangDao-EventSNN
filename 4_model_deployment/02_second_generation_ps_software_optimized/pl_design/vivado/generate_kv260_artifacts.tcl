if {![info exists ::env(WANGDAO_REPO_ROOT)]} {
    error "WANGDAO_REPO_ROOT is not set"
}

set repo_root $::env(WANGDAO_REPO_ROOT)
set project_dir [file join $repo_root hardware vivado wangdao_kv260]
set project_file [file join $project_dir wangdao_kv260.xpr]
set export_dir [file join $project_dir export]
set handoff_dir [file join $repo_root hardware kv260 artifacts]

if {![file exists $project_file]} {
    error "Vivado project not found: $project_file"
}

file mkdir $export_dir
file mkdir $handoff_dir
open_project $project_file

set impl_run [get_runs impl_1]
if {[llength $impl_run] != 1} {
    error "Implementation run impl_1 was not found"
}

set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE 1 $impl_run
launch_runs $impl_run -to_step write_bitstream -jobs 2
wait_on_run $impl_run

set run_status [get_property STATUS $impl_run]
puts "WANGDAO_IMPL_STATUS $run_status"
if {![string match "*Complete*" $run_status]} {
    error "Implementation/bitstream run did not complete: $run_status"
}

set run_dir [get_property DIRECTORY $impl_run]
set bit_files [glob -nocomplain [file join $run_dir *.bit]]
set bin_files [glob -nocomplain [file join $run_dir *.bin]]
if {[llength $bit_files] != 1} {
    error "Expected one bitstream in $run_dir, found [llength $bit_files]"
}
if {[llength $bin_files] != 1} {
    error "Expected one bitstream binary in $run_dir, found [llength $bin_files]"
}

set export_bit [file join $export_dir wangdao.bit]
set export_bit_bin [file join $export_dir wangdao.bit.bin]
file copy -force [lindex $bit_files 0] $export_bit
file copy -force [lindex $bin_files 0] $export_bit_bin

open_run $impl_run
set export_xsa [file join $export_dir wangdao.xsa]
write_hw_platform -fixed -include_bit -force -file $export_xsa

# Keep the small, checked-in handoff directory synchronized with the latest
# implementation output. Vivado caches and the project itself remain local.
file copy -force $export_bit_bin [file join $handoff_dir wangdao_pl_flat.bit.bin]
file copy -force $export_bit_bin [file join $handoff_dir wangdao.bit.bin]
file copy -force $export_xsa [file join $handoff_dir wangdao.xsa]
set flat_export_dir [file join $export_dir wangdao_pl_flat]
foreach handoff_name {wangdao_pl_flat.dtbo shell.json} {
    set handoff_source [file join $flat_export_dir $handoff_name]
    if {[file exists $handoff_source]} {
        file copy -force $handoff_source [file join $handoff_dir $handoff_name]
    }
}

puts "WANGDAO_ARTIFACT bit=$export_bit"
puts "WANGDAO_ARTIFACT bit_bin=$export_bit_bin"
puts "WANGDAO_ARTIFACT xsa=$export_xsa"
puts "WANGDAO_HANDOFF_DIR $handoff_dir"

close_design
close_project
