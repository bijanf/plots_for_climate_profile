# Climate Data Visualization Tool

This repository contains a Python script and a Bash script designed to visualize climate data specifically for Cenral Asian countries. The Python script generates maps with temperature data overlays, and the Bash script assists with the downloading of necessary shapefiles.

## Contents

- `plot_01.py`: A Python script that calculates 30-year average temperatures, creates masks based on the country shapefile, and plots temperature maps for different climate scenarios.
- `download.sh`: A Bash script to automate the downloading of shapefiles required by the Python script.

## Requirements

- Python 3.x
- Required Python packages: `xarray`, `geopandas`, `matplotlib`, `cartopy`, `numpy`, `requests`, `rasterio`, and `affine`. Install these packages using `pip`:

  ```sh
  pip install xarray geopandas matplotlib cartopy numpy requests rasterio affine
  ```

- Bash shell (for running the Bash script).

## Usage

1. Run the Bash script to download the required shapefiles:

   ```sh
   ./download.sh
   ```

2. Execute the Python script to generate the temperature maps:

   ```sh
   python plot_01.py
   ```

3. The output will be saved as `temperature_maps.png` in the current directory.

## Contributing

Feel free to fork this repository, make changes, and open a pull request to contribute to this project.

## License

MIT License

Copyright (c) [2024] [Bijan Fallah]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

