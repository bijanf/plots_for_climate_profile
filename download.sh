#!/bin/bash
set -e
# Base URL for CHELSA data
base_url="https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies"

# Directory to store downloaded and cropped files
output_dir="./downloaded_files"
mkdir -p "$output_dir"
variable="tas"  # Temperature
# Bounding box coordinates for cropping
lat_min=33
lat_max=56.5
lon_min=44
lon_max=91
# Format for GDAL's projwin: minX maxY maxX minY
bbox="$lon_min $lat_max $lon_max $lat_min"

# Download and crop function
download_and_crop() {
    local url=$1
    local filename=$(basename "$url")
    local filepath="$output_dir/$filename"
    local cropped_filepath="$output_dir/cropped_$filename"

    # Download file
    echo "Downloading $url..."
    wget -q -O "$filepath" "$url" && echo "Downloaded to $filepath"

    # Crop file
    if [[ -f "$filepath" ]]; then
        echo "Cropping $filename to fit the bounding box..."
        gdal_translate -projwin $bbox "$filepath" "$cropped_filepath" && echo "Cropped and saved to $cropped_filepath"
    else
        echo "Failed to download $filename or file does not exist."
    fi
}

# Example download: Historical data (adjust URLs as needed)
historical_months=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
for month in ${historical_months[@]}; do
    url="${base_url}/1981-2010/${variable}/CHELSA_${variable}_${month}_1981-2010_V.2.1.tif"
    download_and_crop "$url"
done

# Future scenarios (conceptual example, adjust as necessary)
# Assuming a simplified pattern for demonstration purposes
models=("gfdl-esm4" "ipsl-cm6a-lr" "mri-esm2-0" "ukesm1-0-ll" "mpi-esm1-2-hr")
scenarios=("ssp126" "ssp585")
time_slices=("2011-2040" "2041-2070" "2071-2100")
for month in {01..12}; do
for model in ${models[@]}; do
    for scenario in ${scenarios[@]}; do
        for time_slice in ${time_slices[@]}; do
            # Adjust this URL pattern according to the actual file naming conventions
            # This is a placeholder example and will not directly work without modification
            ensemble_member="r1i1p1f1"
            [[ "$model" == "ukesm1-0-ll" ]] && ensemble_member="r1i1p1f1"
            url="${base_url}/${time_slice}/${model^^}/${scenario}/${variable}/CHELSA_${model}_${ensemble_member}_w5e5_${scenario}_${variable}_${month}_${time_slice//-/_}_norm.tif"
            download_and_crop "$url"
	    rm ${output_dir}/CHELSA_*.tif
        done
    done
done
done #month

echo "All processes complete."
