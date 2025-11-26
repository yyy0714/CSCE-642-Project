open_project p1

open_solution -flow_target vitis s1

add_files "bicg.cpp"

set_top kernel

set_part xcu200-fsgd2104-2-e

create_clock -period 250MHz

source pragmas.tcl

csynth_design

#export_design -flow impl

close_project

exit
