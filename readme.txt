First get models_mesa_wfreq and grid_nofreqs (on external HD also
Alesandro's Desktop in /data/walter), which contains MESA histories and
gyre summaries. Then also extract grid_RGB_MAZYAp.tar.gz (on OneDrive).

Order in which to run checking scripts:
If checking an AIMS binary grid file:
1) ./grid_check.py storage_files_for_grids/files_grid_RGB_MAZYAp /data/walter/models_mesa_wfreq/ -n 24
2) ./rerun_gyre.py  (config in script)
3) replace_bad_freq.py (config in script)
4) AIMS.py (with new binary_grid name, mode = "write_grid", and set_mode_n_to_radial_n = True).

If checking a MESA grid:
1) ./check_gyre_mesa_grid.py  (config in script)
2) ./create_aims_grid_input.py (config in script)
3) AIMS.py (with new binary_grid name, mode = "write_grid", and set_mode_n_to_radial_n = True).

# grid_check.py
Originally wanted to be able to check grids using only data in grid file.
However, this was not possible due to many false positives and negatives.
Decided to instead use the input gyre summaries from grid creation by
checking for Delta n_pg > 1, as this indicates modes skipped in gyre.
Will create a file which contains indeces of the bad models which is used
by rerun_gyre.py. Best to run this iteractively as it has a make_plot
function with which to create a diagnostic plot.

# rerun_gyre.py
Rerun gyre for models which have missing modes. This script keeps increasing
gyre's frequency resolution up to a threshold before continuing to the next
model.

# replace_bad_freq.py
Mostly converting gyre summaries to simple/CLES format used by AIMS and putting
them where AIMS can find them.

# check_gyre_mesa_grid.py
Performs the same checks as in grid_check.py but without the need to already have
and AIMS binary grid file.

# create_aims_grid_input.py
Creates a binary grid input file for AIMS.

# AIMS.py
Recreate grid with set_mode_n_to_radial_n = True. This sets non-radial modes'
order to the corresponding radial mode order. This is fine to do as the mode
orders are only used for interpolation between models.
