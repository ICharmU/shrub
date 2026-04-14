import ee
import geemap
import folium
import numpy as np
import math

# EXAMPLE INPUT IMAGE - for testing purposes, do not remove.
# ee.Authenticate()
# ee.Initialize(project='...')
# sample_image = ee.ImageCollection('USGS/3DEP/10m_collection').first()

# ==============================================================================
# INDIVIDUAL TERRAIN FEATURE FUNCTIONS
# ==============================================================================

def get_slope_and_aspect(elevation: ee.Image) -> ee.Image:
    """
    Computes slope and aspect using standard GEE terrain products.
    Aspect is 0-360 degrees clockwise from North.
    """
    # ee.Terrain.products calculates elevation, slope, and aspect simultaneously
    terrain = ee.Terrain.products(elevation)
    return terrain.select(['slope', 'aspect'])

def get_northness_eastness(aspect: ee.Image) -> ee.Image:
    """
    Converts aspect (degrees) to Northness (cos) and Eastness (sin).
    Northness: 1 is North, -1 is South.
    Eastness: 1 is East, -1 is West.
    """
    # Convert degrees to radians
    aspect_rad = aspect.multiply(math.pi / 180.0)
    
    northness = aspect_rad.cos().rename('northness')
    eastness = aspect_rad.sin().rename('eastness')
    
    return ee.Image.cat([northness, eastness])

def get_curvature(elevation: ee.Image) -> ee.Image:
    """
    Approximates terrain shape (curvature) using a Laplacian filter.
    Positive values typically represent concave features (valleys/depressions).
    Negative values typically represent convex features (ridges/peaks).
    """
    laplacian_kernel = ee.Kernel.laplacian8()
    curvature = elevation.convolve(laplacian_kernel).rename('curvature')
    return curvature

def get_ruggedness(elevation: ee.Image, kernel_radius: int = 3) -> ee.Image:
    """
    Calculates a Terrain Ruggedness proxy using focal standard deviation.
    Higher values indicate more rugged, variable terrain.
    """
    ruggedness = elevation.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=kernel_radius, units='pixels')
    ).rename('ruggedness')
    return ruggedness

def get_tpi(elevation: ee.Image, radius_meters: int = 150) -> ee.Image:
    """
    Calculates Topographic Position Index (TPI).
    Positive TPI = hills/ridges. Negative TPI = valleys/canyons.
    Near zero = flat areas or mid-slopes.
    """
    focal_mean = elevation.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=radius_meters, units='meters')
    )
    tpi = elevation.subtract(focal_mean).rename('tpi')
    return tpi

def get_exposure_proxies(elevation: ee.Image, azimuth: int = 270, zenith: int = 45) -> ee.Image:
    """
    Generates hillshade as a proxy for topographic exposure/insolation.
    Note: True Topographic Wetness Index (TWI) requires Flow Accumulation, 
    which is computationally heavy to calculate on-the-fly without a pre-computed 
    flow direction dataset. Hillshade serves as an immediate structural exposure proxy.
    """
    hillshade = ee.Terrain.hillshade(elevation, azimuth, zenith).rename('hillshade_exposure')
    return hillshade

def extract_all_terrain_features(elevation_image: ee.Image) -> ee.Image:
    """
    Main pipeline function. Takes a 3DEP elevation image and returns a 
    multi-band image containing all terrain/site context features.
    """
    # 1. Base Elevation
    elevation = elevation_image.rename('elevation')
    
    # 2. Slope & Aspect
    slope_aspect = get_slope_and_aspect(elevation)
    
    # 3. Northness & Eastness
    north_east = get_northness_eastness(slope_aspect.select('aspect'))
    
    # 4. Terrain Shape (Curvature)
    curvature = get_curvature(elevation)
    
    # 5. Ruggedness
    ruggedness = get_ruggedness(elevation, kernel_radius=3)
    
    # 6. Topographic Position Index (TPI)
    tpi = get_tpi(elevation, radius_meters=150)
    
    # 7. Exposure Proxies
    exposure = get_exposure_proxies(elevation)
    
    # Combine everything into a single Feature Cube
    terrain_feature_family = ee.Image.cat([
        elevation,
        slope_aspect,  # Adds 'slope' and 'aspect' bands
        north_east,    # Adds 'northness' and 'eastness' bands
        curvature,
        ruggedness,
        tpi,
        exposure
    ])

    bands = terrain_feature_family.bandNames().getInfo()

    # use bands to extract feature info.
    return bands, terrain_feature_family

