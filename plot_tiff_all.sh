#!/bin/bash

# Example of calling the Python script with different parameters
python plot_tiff_all.py "TKM" "ssp126" "2041_2070" "True"
python plot_tiff_all.py "TKM" "ssp585" "2041_2070" "False"

python plot_tiff_all.py "TKM" "ssp126" "2071_2100" "False"
python plot_tiff_all.py "TKM" "ssp585" "2071_2100" "False"
