# Copy Proposal-SNN ROM files into the active Vivado synthesis run directory.
#
# The RTL intentionally uses short $readmemh filenames so the same sources
# work with the repository's XSim harness.  Vivado OOC and top-level synthesis
# execute from their own run directories, so the files must be present there
# before synth_design elaborates the initial blocks.

# Do not normalize these paths.  On Windows Vivado 2025.2.1 can normalize a
# path below a Desktop junction incorrectly (dropping the Desktop component).
# Direct paths and relative file operations remain valid in the run process.
set script_dir [file dirname [info script]]
set hardware_dir [file dirname $script_dir]
set vector_dir [file join $hardware_dir vectors]
set run_dir [pwd]

set required_files {}
foreach layer {1 2 3} {
    lappend required_files "layer${layer}_shifts.txt"
    for {set bank 0} {$bank < 8} {incr bank} {
        lappend required_files "layer${layer}_bank${bank}.txt"
    }
}

foreach file_name $required_files {
    set source_file [file join $vector_dir $file_name]
    if {![file isfile $source_file]} {
        error "Wangdao vector file is missing: $source_file"
    }
    # The run process has already changed to its own run directory.  Keep the
    # destination relative so Tcl does not re-normalize the Windows path.
    file copy -force $source_file $file_name
    if {![file isfile $file_name]} {
        error "Wangdao vector file was not copied: $file_name (run_dir=$run_dir)"
    }
}

puts "WANGDAO_VECTOR_PREPARE_OK run_dir=$run_dir files=[llength $required_files]"
