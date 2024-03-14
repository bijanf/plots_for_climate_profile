import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import requests
from io import BytesIO
from zipfile import ZipFile
import rasterio.features
from affine import Affine
import os
from matplotlib.colors import BoundaryNorm
from cartopy.feature import ShapelyFeature
import sys
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable, get_cmap

# Default values can be set if no command line arguments are provided
default_country = "KAZ"
default_scenario = "ssp126"
default_years_range_future = "2041_2070"
vmin=1
vmax=6.5
N = 11
#cmaps='Reds'
cmaps='Blues'
variable="pr"
country = sys.argv[1] if len(sys.argv) > 1 else default_country
scenario = sys.argv[2] if len(sys.argv) > 2 else default_scenario
years_range_future = sys.argv[3] if len(sys.argv) > 3 else default_years_range_future
# Parsing command line arguments
# Assuming the first three arguments are country, scenario, and years_range_future
plot_historical_flag = sys.argv[4] if len(sys.argv) > 4 else "False"

# Convert the string flag to a boolean
plot_historical = plot_historical_flag.lower() in ['true', '1', 't', 'y', 'yes']


# Function to download and extract shapefiles
def download_and_unzip_shapefiles(url, extract_to='.'):
    response = requests.get(url)
    if response.status_code == 200:
        zipfile = ZipFile(BytesIO(response.content))
        zipfile.extractall(extract_to)
        print("Files extracted to", extract_to)
    else:
        print("Error while downloading shapefile. Status Code: ",
              response.status_code)

# Download Turkmenistan shapefiles
gadm_url = 'https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_'+country+'_shp.zip'
download_and_unzip_shapefiles(gadm_url)

# Load Turkmenistan shapefile

shapefile_path = './gadm36_'+country+'_0.shp'
shapefile = gpd.read_file(shapefile_path).to_crs(epsg=4326)

shapefile_path1 = './gadm36_'+country+'_1.shp'
shapefile1 = gpd.read_file(shapefile_path1).to_crs(epsg=4326)

scale = 0.1
#offset = -273.15
offset = 0.0
def load_and_adjust_tiff(tiff_path, scale, offset, shapefile):
    with rasterio.open(tiff_path) as src:
        # Mask the raster with the given shapefile, cropping to its extent
        out_image, out_transform = rasterio.mask.mask(src,
                                                       shapefile.geometry, crop=True, nodata=src.nodata)

        # Apply scale and offset if needed
        if src.nodata is not None:
            out_image = np.where(out_image == src.nodata, np.nan, out_image)
        out_image = (out_image * scale) + offset

    return out_image


models = ['gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0', 'ukesm1-0-ll']
years_range_historical = "1981-2010"

# Function to construct file paths for each month and model
def construct_file_paths(base_pattern, years_range, models, month, scenario=""):
    return [base_pattern.format(years_range=years_range, model=model, month=str(month).zfill(2), scenario=scenario) for model in models]

# Future projections file paths for each month and model
future_tiff_base_pattern = 'data/cropped_CHELSA_{model}_r1i1p1f1_w5e5_{scenario}_{variable}_{month}_{years_range}_norm.tif'

# Historical reference file paths for each month
historical_tiff_base_pattern = 'data/cropped_CHELSA_{variable}_{month}_{years_range}_V.2.1.tif'

monthly_anomalies = []
for month in range(1, 13):
    future_datasets_monthly = []
    for model in models:
        future_file_path = construct_file_paths(future_tiff_base_pattern, years_range_future, [model], month, scenario)[0]
        future_dataset = load_and_adjust_tiff(future_file_path, scale, offset, shapefile)
        # After loading a dataset
        if not np.any(future_dataset):  # This checks if the dataset is entirely NaN or empty
            print("Dataset is empty or all NaN")
        future_datasets_monthly.append(future_dataset)


    historical_file_path = construct_file_paths(historical_tiff_base_pattern, years_range_historical, [""], month)[0]

    historical_dataset_monthly = load_and_adjust_tiff(historical_file_path, scale, offset, shapefile)

    # Calculate anomalies for this month across all models
    anomaly_datasets_monthly = [dataset - historical_dataset_monthly for dataset in future_datasets_monthly]
    # Before calculating mean
    if np.isnan(anomaly_datasets_monthly).all():
         print("Warning: attempting to calculate mean of all-NaN slice.")
    ensemble_mean_anomaly_monthly = np.nanmean(anomaly_datasets_monthly, axis=0)
    monthly_anomalies.append(ensemble_mean_anomaly_monthly)

# Calculate yearly average of monthly anomalies
yearly_anomaly = np.nanmean(monthly_anomalies, axis=0)


# Adjust these based on your TIFF metadata
scale = 0.1
offset = -273.15



def get_grid_coordinates(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Get the bounds of the raster
        left, bottom, right, top = src.bounds
        # Generate linear spaces for longitude and latitude
        lon = np.linspace(left, right, src.width + 1)  # One more because these are cell corners
        lat = np.linspace(bottom, top, src.height + 1)  # Same here

        # Meshgrid to create 2D arrays of lon and lat values
        lon, lat = np.meshgrid(lon, lat)

        return lon, lat

# Use this function for one of your TIFFs to generate lon and lat
lon, lat = get_grid_coordinates('data/cropped_CHELSA_gfdl-esm4_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif')


# Assuming you have the geographic extent of your raster data
min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)
min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
extent = [min_lon, max_lon, min_lat, max_lat]


from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Generate 5 colors from the coolwarm colormap
base_cmap = plt.get_cmap(cmaps)

colors = base_cmap(np.linspace(0, 1, N))

# Create a new colormap from these colors
cmap = LinearSegmentedColormap.from_list("custom_coolwarm", colors, N=N)

# Assuming vmin and vmax are defined
boundaries = np.linspace(vmin, vmax, N + 1)  # Creates 5 intervals with 6 boundaries

# Create a BoundaryNorm using these boundaries
norm = BoundaryNorm(boundaries, cmap.N, clip=True)




# Plot the data for each scenario
bounds = shapefile.total_bounds

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

# Use the custom cmap and norm
raster_plot = ax.imshow(yearly_anomaly.squeeze(),
                        cmap=cmap, norm=norm, extent=[bounds[0],
                                                       bounds[2], bounds[1], bounds[3]],
                        transform=ccrs.PlateCarree(), zorder=0)





# Adding the shapefile
shape_feature = ShapelyFeature(shapefile1.geometry, ccrs.PlateCarree(),
                               facecolor='none', edgecolor='black')
ax.add_feature(shape_feature, linewidth=1)

# Adjustments for a cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Add colorbar
#cbar = fig.colorbar(raster_plot, ax=ax, orientation='vertical', shrink=0.8, aspect=20, extend='both')
## Update colorbar label
#cbar.set_label('Temperature Anomaly (°C)', fontsize=12)
filename = f'pngs/ensemble_mean_{scenario}_{years_range_future}_{country}_{variable}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory
os.system('rm -rf ./gadm36_*_*.*')





# Now, separately create and save the colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

#cmap = plt.colormaps['coolwarm']
#norm = Normalize(vmin=vmin, vmax=vmax)

cb1 = ColorbarBase(ax, cmap=cmap,
                   norm=norm,
                   orientation='horizontal')
cb1.set_label('Temperature Anomaly (°C)' , fontsize=20)
cb1.ax.tick_params(labelsize=20)
plt.savefig(f'pngs/colorbar_{variable}.png', dpi=300, bbox_inches='tight')
plt.close(fig)



if plot_historical:

    # Assuming you've already defined a function for loading and adjusting TIFF files, `load_and_adjust_tiff`
    historical_datasets = []

    for month in range(1, 13):
        historical_file_path = construct_file_paths(historical_tiff_base_pattern, years_range_historical, [""], month)[0]
        historical_dataset = load_and_adjust_tiff(historical_file_path, scale, offset, shapefile)
        historical_datasets.append(historical_dataset)

    # Calculate the average across all months
    historical_average = np.nanmean(historical_datasets, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

    # Assuming you want to keep using the coolwarm colormap
    # Adjust vmin and vmax based on the range of your historical data
    vmin_hist = np.nanmin(historical_average)
    vmax_hist = np.nanmax(historical_average)
    cmap_hist = plt.get_cmap('coolwarm')

    norm_hist = Normalize(vmin=vmin_hist, vmax=vmax_hist)
    raster_plot_hist = ax.imshow(historical_average.squeeze(),
                                 cmap=cmap_hist, norm=norm_hist, extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                                 transform=ccrs.PlateCarree(), zorder=0)

    # Add the shapefile, if needed
    shape_feature_hist = ShapelyFeature(shapefile1.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='black')
    ax.add_feature(shape_feature_hist, linewidth=1)

    # Hide the spines and ticks for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for the historical data
    #cbar_hist = fig.colorbar(raster_plot_hist, ax=ax, orientation='vertical', shrink=0.8, aspect=20, extend='both')
    #cbar_hist.set_label('Temperature (°C)', fontsize=12)

    # Save the plot
    filename_hist = f'pngs/absolute_historical_{years_range_historical}_{country}_{variable}.png'
    plt.savefig(filename_hist, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cb1_hist = ColorbarBase(ax, cmap=cmap_hist, norm=norm_hist, orientation='horizontal')
    cb1_hist.set_label('Temperature (°C)' , fontsize=20)
    cb1_hist.ax.tick_params(labelsize=20)
    plt.savefig(f'pangs/colorbar_historical_{variable}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
