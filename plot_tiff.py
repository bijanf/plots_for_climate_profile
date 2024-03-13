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

from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable, get_cmap

country = "KAZ"

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
offset = -273.15
vmin=-2
vmax=2
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

tiff_files = [
    'data/cropped_CHELSA_gfdl-esm4_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif',
    'data/cropped_CHELSA_ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif',
    'data/cropped_CHELSA_mpi-esm1-2-hr_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif',
    'data/cropped_CHELSA_mri-esm2-0_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif',
    'data/cropped_CHELSA_ukesm1-0-ll_r1i1p1f1_w5e5_ssp126_tas_05_2041_2070_norm.tif'
]
historical_tiff_file = 'data/cropped_CHELSA_tas_05_1981-2010_V.2.1.tif'




# Adjust these based on your TIFF metadata
scale = 0.1
offset = -273.15

# Load and adjust each TIFF, then store the result
datasets = [load_and_adjust_tiff(tiff, scale,
                                  offset, shapefile) for tiff in tiff_files]
historical_dataset = load_and_adjust_tiff(historical_tiff_file,
                                          scale, offset, shapefile)
# Calculate anomalies for each future projection dataset
anomaly_datasets = [dataset - historical_dataset for dataset in datasets]
# Calculate ensemble mean of anomalies
ensemble_mean_anomaly = np.nanmean(anomaly_datasets, axis=0)

# Calculate ensemble mean
ensemble_mean = np.nanmean(datasets, axis=0)

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
lon, lat = get_grid_coordinates(tiff_files[0])

# Verify the shape of ensemble_mean matches expectations
print(ensemble_mean.shape)
# Assuming you have the geographic extent of your raster data
min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)
min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
extent = [min_lon, max_lon, min_lat, max_lat]

# Plot the data for each scenario
bounds = shapefile.total_bounds

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

cmap = plt.colormaps['coolwarm']  # Updated colormap access
# Update the plotting code to use ensemble_mean_anomaly
norm = Normalize(vmin=vmin, vmax=vmax)
raster_plot = ax.imshow(ensemble_mean_anomaly.squeeze(),
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
cbar = fig.colorbar(raster_plot, ax=ax, orientation='vertical', shrink=0.8, aspect=20, extend='both')
# Update colorbar label
cbar.set_label('Temperature Anomaly (°C)', fontsize=12)
#cbar.set_label('Temperature (°C)', fontsize=12)
plt.savefig('ensemble_mean.png', dpi=300, bbox_inches='tight')
plt.show()
os.system('rm -rf ./gadm36_*_*.*')
