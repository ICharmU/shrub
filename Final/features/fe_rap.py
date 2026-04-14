import ee
import os
from pathlib import Path
import requests

def get_surrounding_shrub_fraction(shrub_band: ee.Image, radius_meters: int = 500) -> ee.Image:
    """
    Calculates the broad neighborhood shrub fraction.
    Averages the 0-100% cover values over a large area to establish a coarse prior.
    """
    shrub_context = shrub_band.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=radius_meters, units='meters')
    ).rename('shrub_fraction_prior')
    
    return shrub_context

def get_broad_vegetation_composition(veg_image: ee.Image, comp_bands: list, radius_meters: int = 500) -> ee.Image:
    """
    Calculates the neighborhood context for competing vegetation types 
    (trees, grasses) to understand the surrounding ecological matrix.
    """
    broad_veg = veg_image.select(comp_bands)
    
    veg_context = broad_veg.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=radius_meters, units='meters')
    )
    
    # Rename bands dynamically to append '_prior' instead of the default '_mean'
    new_names = [f"{band}_prior" for band in comp_bands]
    return veg_context.rename(new_names)

def extract_coarse_vegetation_prior(
    veg_image: ee.Image, 
    shrub_band_name: str = 'SHR', 
    comp_band_names: list = ['TRE', 'PFG', 'AFG'],
    context_radius: int = 500
) -> ee.Image:
    """
    Main pipeline function tailored for RAP fractional cover datasets.
    """
    # 1. Local/Pixel-level Shrub Presence (The exact 10m pixel fraction)
    local_shrub = veg_image.select(shrub_band_name).rename('local_shrub_presence')
    
    # 2. Neighborhood Shrub Fraction (Coarse Prior)
    shrub_prior = get_surrounding_shrub_fraction(local_shrub, radius_meters=context_radius)
    
    # 3. Broad Vegetation Composition around the area
    broad_veg_prior = get_broad_vegetation_composition(veg_image, comp_bands=comp_band_names, radius_meters=context_radius)
    
    # Combine everything into a single Feature Cube
    vegetation_feature_family = ee.Image.cat([
        local_shrub,
        shrub_prior,
        broad_veg_prior
    ])
    
    return vegetation_feature_family

def save_as_tif(vegetation_feature_family, export_region, out_fname):
    """
    Save ee.Image to .tif

    Note that if the image trying to be saved is too large, the download will fail silently.
    """
    out_tif = os.path.join(os.getcwd(), out_fname)
    # print(f"Starting download to: {out_tif}")

    try:
        download_url = vegetation_feature_family.getDownloadURL({
            'scale': 10,
            'region': export_region,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(download_url)
        
        with open(out_tif, 'wb') as file:
            file.write(response.content)

    except ee.ee_exception.EEException as e:
        print(e)
    except Exception as e:
        print(f"Unknown exception occurred:\n\n{e}")

    # print("Download complete!")

#################
# EXAMPLE
#################
# ee.Authenticate()
# ee.Initialize(project="shrub-488520")

# rap_image = ee.ImageCollection("projects/rap-data-365417/assets/vegetation-cover-10m").first()
# actual_bands = rap_image.bandNames().getInfo()

# veg_features = extract_coarse_vegetation_prior(
#     veg_image=rap_image,
#     shrub_band_name='SHR', 
#     comp_band_names=actual_bands, # Perennial and Annual grasses
#     context_radius=500
# )

# export_region = ee.Geometry.Rectangle([-105.6, 40.35, -105.55, 40.4])
# save_as_tif(veg_features, export_region, Path("Final") / "features" / "processed" / "coarse_vegetation_features.tif")