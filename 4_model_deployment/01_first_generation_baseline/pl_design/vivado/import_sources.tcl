# Import the repository RTL, memory files, and AXI-Lite smoke test into the
# manually created KV260 project.  Run from Vivado with Tools -> Run Tcl Script.

# Avoid file normalize here.  In this Windows checkout Vivado 2025.2.1 can
# drop the Desktop component while normalizing paths below a Desktop junction.
# Nested dirname calls also avoid leaving `..` components in Tcl file paths.
set script_dir [file dirname [info script]]
set repo_root [file dirname [file dirname $script_dir]]
set project_file [file join $script_dir wangdao_kv260 wangdao_kv260.xpr]

if {[llength [get_projects -quiet]] == 0} {
    if {![file exists $project_file]} {
        error "KV260 project not found: $project_file"
    }
    open_project $project_file
}

# Resolve the hook next to the opened project rather than from a normalized
# source-script path.  This keeps the property absolute even when the user
# invokes this script with a relative `source` command.
set project_dir [get_property DIRECTORY [current_project]]
set vector_prep_script [file join [file dirname $project_dir] prepare_memory_files.tcl]

set design_files [concat \
    [glob -nocomplain -types f [file join $repo_root hardware rtl *.sv]] \
    [glob -nocomplain -types f [file join $repo_root hardware rtl *.v]]]
if {[llength $design_files] > 0} {
    add_files -fileset [get_filesets sources_1] -norecurse $design_files
}

set memory_files [glob -nocomplain -types f [file join $repo_root hardware vectors *.txt]]
if {[llength $memory_files] > 0} {
    add_files -fileset [get_filesets sources_1] -norecurse $memory_files
    add_files -fileset [get_filesets sim_1] -norecurse $memory_files
    foreach memory_file $memory_files {
        set memory_obj [get_files -quiet $memory_file]
        if {[llength $memory_obj] > 0} {
            set_property file_type {Memory Initialization Files} $memory_obj
        }
    }
}

set simulation_file [file join $repo_root hardware tb tb_proposal_axi_lite.sv]
if {[file exists $simulation_file]} {
    add_files -fileset [get_filesets sim_1] -norecurse $simulation_file
    set_property top tb_proposal_axi_lite [get_filesets sim_1]
}

# Keep the hook in the project utility fileset as well as on each run.  This
# makes the project archive/self-contained while the hook itself resolves the
# repository vector directory from its own location.
if {[file isfile $vector_prep_script] && [llength [get_filesets -quiet utils_1]] > 0} {
    set hook_file [get_files -quiet $vector_prep_script]
    if {[llength $hook_file] == 0} {
        add_files -fileset [get_filesets utils_1] -norecurse $vector_prep_script
    }
}

update_compile_order -fileset [get_filesets sources_1]
update_compile_order -fileset [get_filesets sim_1]

# OOC module-reference synthesis and the top-level synthesis run use private
# working directories.  Register the same pre-hook on every synthesis run so
# all $readmemh initial blocks see exactly the repository ROM files.  Restrict
# this to synthesis runs; implementation runs do not execute synth_design.
set configured_synth_runs {}
if {[file isfile $vector_prep_script]} {
    foreach run [get_runs -quiet] {
        set run_name [get_property NAME $run]
        if {$run_name eq "synth_1" || [string match "*_synth_*" $run_name]} {
            set_property STEPS.SYNTH_DESIGN.TCL.PRE $vector_prep_script $run
            lappend configured_synth_runs $run_name
        }
    }
}
puts "WANGDAO_IMPORT_OK project=[get_property DIRECTORY [current_project]] design_files=[llength $design_files] memory_files=[llength $memory_files] vector_prep=[file isfile $vector_prep_script] synth_runs=[llength $configured_synth_runs]"
