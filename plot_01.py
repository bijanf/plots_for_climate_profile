import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
import rasterio.features
from affine import Affine
import os
from matplotlib.colors import BoundaryNorm
from cartopy.feature import ShapelyFeature

country = "UZB"

# Function to download and extract shapefiles
def download_and_unzip_shapefiles(url, extract_to='.'):
    response = requests.get(url)
    if response.status_code == 200:
        zipfile = ZipFile(BytesIO(response.content))
        zipfile.extractall(extract_to)
        print("Files extracted to", extract_to)
    else:
        print("Error while downloading shapefile. Status Code: ", response.status_code)

# Download Turkmenistan shapefiles
gadm_url = 'https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_'+country+'_shp.zip'
download_and_unzip_shapefiles(gadm_url)

# Load the shapefile for Turkmenistan
shapefile_path = './gadm36_'+country+'_0.shp'
turkmenistan = gpd.read_file(shapefile_path)

# Ensure the CRS for Turkmenistan is WGS84 (epsg:4326)
turkmenistan = turkmenistan.to_crs(epsg=4326)



# Load the shapefile for Turkmenistan
shapefile_path_01 = './gadm36_'+country+'_1.shp'
turkmenistan_01 = gpd.read_file(shapefile_path_01)

# Ensure the CRS for Turkmenistan is WGS84 (epsg:4326)
turkmenistan_01 = turkmenistan_01.to_crs(epsg=4326)




# Function to calculate the 30-year average for a given NetCDF file and time slice
def calculate_30yr_average(nc_file, time_slice):
    ds = xr.open_dataset(nc_file, decode_times=True)
    tas_avg = ds['tas'].sel(time=time_slice).mean(dim='time')
    return tas_avg

# Define time slices
time_slices = {

    "observed": slice("1979-01-01", "2010-12-31"),
    "historical": slice("1979-01-01", "2010-12-31"),
    "ssp126_near": slice("2031-01-01", "2060-12-31"),
    "ssp126_far": slice("2071-01-01", "2100-12-31"),
    "ssp585_near": slice("2031-01-01", "2060-12-31"),
    "ssp585_far": slice("2071-01-01", "2100-12-31"),
}
nc_files = {
    "observed": "./kfo_ens_obs.nc",
    "historical": "./kfo_ens_historical.nc",
    "ssp126_near": "./kfo_ens_ssp126.nc",
    "ssp126_far": "./kfo_ens_ssp126.nc",
    "ssp585_near": "./kfo_ens_ssp585.nc",
    "ssp585_far": "./kfo_ens_ssp585.nc",
}

# Function to create a mask from the Turkmenistan shapefile
def create_mask(lon, lat, shapefile):
    # Create a transform for the mask
    transform = Affine.translation(lon[0] - np.diff(lon).mean() / 2,
                                   lat[0] - np.diff(lat).mean() / 2) * \
                Affine.scale(np.diff(lon).mean(), np.diff(lat).mean())
    mask = rasterio.features.geometry_mask(shapefile.geometry,
                                           out_shape=(len(lat), len(lon)),
                                           transform=transform, invert=True)
    return mask


# Calculate the 30-year averages for each time slice and scenario
# Calculate the 30-year averages for each time slice and scenario
averages = {}
for scenario, time_slice in time_slices.items():
    tas_avg = calculate_30yr_average(nc_files[scenario], time_slice)
    lon = tas_avg.coords['lon'].values
    lat = tas_avg.coords['lat'].values
    mask = create_mask(lon,lat, turkmenistan)
    tas_avg_masked = tas_avg.where(xr.DataArray(mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}))
    averages[scenario] = tas_avg_masked

# Determine global min and max temperature for color scale normalization
global_min = min([data.min().values for data in averages.values()])
global_max = max([data.max().values for data in averages.values()])

# Prepare the plots
fig, axs = plt.subplots(3, 2, figsize=(10, 10),
                        subplot_kw={'projection': ccrs.PlateCarree()})
# This will be used to place the colorbar
fig.subplots_adjust(bottom=0.1, top=1, left=0.0, right=1, wspace=0.0, hspace=0.0)
axs = axs.ravel()
#titles = [
#    "Observed (1979-2010)",
#    "Historical (1979-2010)",
#    "SSP126 Near Future (2031-2060)",
#    "SSP126 Far Future (2071-2100)",
#    "SSP585 Near Future (2031-2060)",
#    "SSP585 Far Future (2071-2100)"
#]
titles = [
    "Observed          ",
    "Historical        ",
    "SSP126 Near Future",
    "SSP126 Far Future ",
    "SSP585 Near Future",
    "SSP585 Far Future "
]

# Plot the data for each scenario
bounds = turkmenistan.total_bounds




n_colors = 7  # The number of discrete colors you want


boundaries = np.linspace(global_min, global_max, n_colors + 1)
boundaries = np.round(boundaries).astype(int)  # Round to nearest integer and convert to int
norm = BoundaryNorm(boundaries, n_colors, clip=True)
# Generate a colormap with n_colors + 1 levels
cmap = plt.get_cmap('Reds', n_colors + 1)

for ax, scenario, title in zip(axs, averages, titles):
    data = averages[scenario]
    # Create boundaries for the data of this specific scenario




    # Plot using the rounded boundaries for the norm
    p = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False,
                  cmap=cmap, norm=norm, vmin=np.round(global_min).astype(int),
                    vmax=np.round(global_max).astype(int))



    # Adding the shapefile
    shape_feature = ShapelyFeature(turkmenistan_01.geometry, ccrs.PlateCarree(),
                                   facecolor='none', edgecolor='black')
    ax.add_feature(shape_feature, linewidth=1)
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=25, fontname='Ubuntu', weight='light' )
    # Removing the spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Adding the colorba# Optionally, remove the axis labels and ticks if you want a cleaner look




print(turkmenistan.total_bounds)
# Add colorbar
cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust these values as needed
cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Temperature (°C)', fontsize=25, fontname='Ubuntu', weight='light')
# Increase the fontsize of the colorbar's tick labels
cbar.ax.tick_params(labelsize=25)
 # Adjust the fontsize a# Set the fontname for each tick label individually
for label in cbar.ax.get_xticklabels():
    label.set_fontname('Ubuntu')
    label.set_fontsize(25)
    label.set_weight('light')



plt.savefig("./temperature_maps"+country+".png", dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()

os.system("rm  ./gadm36_*.*")

import matplotlib.font_manager
fonts = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
for font in fonts:
    print(font)
