from PIL import Image
import sys

import matplotlib.pyplot as plt
# Load the images
# Accept the country code from the command line
country = sys.argv[1] if len(sys.argv) > 1 else 'TJK'  # Default to 'TJK' if not specified
image_files = [

    'absolute_historical_1981-2010_'+country+'.png',
    'ensemble_mean_ssp126_2041_2070_'+country+'.png',
    'ensemble_mean_ssp126_2071_2100_'+country+'.png',
    'ensemble_mean_ssp585_2041_2070_'+country+'.png',
    'ensemble_mean_ssp585_2071_2100_'+country+'.png',




    # The sixth subplot will contain the colorbars

]



# Colorbars (assuming horizontal layout)

colorbar_files = [

    'colorbar.png',  # Anomalies colorbar

    'colorbar_historical.png',  # Historical colorbar

]




# Titles for each subplot based on the filenames
titles = [
    "Historical",
    "SSP126 Near Future",
    "SSP126 Far Future",
    "SSP585 Near Future",
    "SSP585 Far Future",

    # The sixth subplot will contain the colorbars, no title needed
]

# Load images using PIL
images = [Image.open(img_file) for img_file in image_files]
colorbars = [Image.open(cbar_file) for cbar_file in colorbar_files]

# Define the figure and subplots layout
fig, axs = plt.subplots(3, 2, figsize=(15, 20))

# Flatten the axes array for easy indexing
axs = axs.flatten()

# Plot each image in a subplot and set titles
for i, ax in enumerate(axs[:-1]):  # Last subplot is reserved for colorbars
    ax.imshow(images[i])
    ax.set_title(titles[i], fontsize=30)
    ax.axis('off')  # Hide axes

# Plot colorbars in the last subplot
for i, cbar in enumerate(colorbars):
    # To position colorbars next to each other, a trick is to create sub-axes
    cbar_ax = axs[-1].inset_axes([0.1, 0.25 * (i + 0.5), .8, 0.3])  # x0, y0, width, height
    cbar_ax.imshow(cbar)
    cbar_ax.axis('off')

# Hide the last subplot's original axes
axs[-1].axis('off')

# Adjust layout to be tight
plt.tight_layout()

# Save the figure with titles
fig.savefig('combined_figure_with_titles_'+country+'.png', dpi=300)
plt.close()

# Return the path to the saved figure with titles
combined_figure_with_titles_path = 'combined_figure_with_titles_'+country+'.png'
combined_figure_with_titles_path
