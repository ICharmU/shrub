import laspy
import pdal
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import json

######################
# SIGNALS
######################
def calculate_roughness_knn(points, k=20):
    """
    Calculates local roughness (Z-standard deviation) using K-Nearest Neighbors.
    
    Parameters:
        points (numpy.ndarray): Nx3 array of X, Y, Z coordinates.
        k (int): Number of nearest neighbors to evaluate.
        
    Returns:
        numpy.ndarray: 1D array of roughness values matching the input point order.
    """
    # 1. Isolate X and Y for a 2.5D surface search 
    # (Use points[:, :3] if you want a true 3D spherical neighborhood)
    xy = points[:, :2]
    z = points[:, 2]
    
    # 2. Build the spatial index
    print("Building KDTree...")
    tree = cKDTree(xy)
    
    # 3. Query the 'k' closest neighbors for every single point
    # workers=-1 tells SciPy to parallelize across all available CPU cores
    print(f"Querying {k} nearest neighbors...")
    distances, indices = tree.query(xy, k=k, workers=-1)
    
    # 4. Extract the Z-elevations for all retrieved neighbors
    # 'indices' is an (N, k) array. This maps our Z array to an (N, k) array of elevations.
    neighbor_z_values = z[indices]
    
    # 5. Vectorized Standard Deviation
    # Calculate the standard deviation across the k-axis (axis=1)
    print("Calculating standard deviation...")
    roughness = np.std(neighbor_z_values, axis=1)
    
    return roughness

def calculate_height_above_ground(input_las, output_las):
    """
    Runs a PDAL pipeline to classify ground points and calculate 
    Height Above Ground (HAG) for a LAS/LAZ file.
    """
    
    # Define the PDAL pipeline as a Python dictionary - C++ compilation
    pipeline_dict = [
        {
            "type": "readers.las",
            "filename": input_las
        },
        {
            # Step 1: Ground Classification
            # SMRF (Simple Morphological Filter) is generally faster and 
            # requires less parameter tuning than CSF for standard terrain.
            "type": "filters.smrf",
            "ignore": "Classification[7:7]", # Ignore noise points if they exist
            "slope": 0.15,
            "window": 18,
            "threshold": 0.5,
            "scalar": 1.25
        },
        {
            # Step 2: Calculate Height Above Ground
            # hag_nn uses nearest-neighbor interpolation from the ground surface.
            "type": "filters.hag_nn"
        },
        {
            # Step 3: Write the output
            "type": "writers.las",
            "filename": output_las,
            # We must explicitly tell the writer to include our new HAG dimension
            "extra_dims": "HeightAboveGround=float32" 
        }
    ]
    
    # Convert the dictionary to a JSON string
    pipeline_json = json.dumps(pipeline_dict)
    
    # Create and execute the PDAL pipeline
    print(f"Starting PDAL pipeline for {input_las}...")
    pipeline = pdal.Pipeline(pipeline_json)
    
    try:
        # execute() returns the number of points processed
        count = pipeline.execute()
        print(f"Successfully processed {count} points.")
        print(f"Saved to {output_las} with new 'HeightAboveGround' dimension.")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

def calculate_surface_angle(input_las, output_las, knn=30):
    """
    Calculates surface normals and derived slope angle for a point cloud.
    
    Parameters:
        input_las (str): Path to input LAS/LAZ file.
        output_las (str): Path to save the processed LAS/LAZ file.
        knn (int): Number of nearest neighbors to use for normal estimation.
    """
    print(f"Reading {input_las}...")
    las = laspy.read(input_las)
    
    # 1. Stack coordinates into an Nx3 NumPy array
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 2. Convert NumPy array to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 3. Estimate Normals
    # We use a KDTree search with a defined number of neighbors (knn)
    print(f"Estimating normals using {knn} nearest neighbors...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    
    # Optional but recommended: Align normals to point generally "up"
    # This helps keep the vectors consistent
    pcd.orient_normals_to_align_with_direction([0., 0., 1.])
    
    # 4. Extract the normals back into a NumPy array
    normals = np.asarray(pcd.normals)
    
    # 5. Calculate the Slope Angle (in degrees)
    print("Calculating surface angles...")
    # Extract the Z component of the normal vectors
    nz = normals[:, 2]
    
    # Calculate arccosine of the absolute Z value (returns radians)
    slope_radians = np.arccos(np.abs(nz))
    
    # Convert radians to degrees for easier interpretation (0 = flat, 90 = vertical)
    slope_degrees = np.degrees(slope_radians)
    
    # 6. Save back to LAS
    print("Writing results to new LAS file...")
    
    # Add new dimensions for the normal vectors (useful for advanced processing)
    # and the calculated slope angle
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="slope_degrees", type=np.float32))
    
    las.normal_x = normals[:, 0]
    las.normal_y = normals[:, 1]
    las.normal_z = normals[:, 2]
    las.slope_degrees = slope_degrees
    
    las.write(output_las)
    print(f"Success! Saved to {output_las}.")

def segment_trees_dbscan(input_las, output_las, eps=0.8, min_points=15, height_threshold=2.0):
    """
    Segments individual tree canopies using 3D DBSCAN clustering.
    
    Parameters:
        input_las (str): Path to input LAS/LAZ (must have HeightAboveGround).
        output_las (str): Path to save the clustered LAS/LAZ.
        eps (float): Maximum distance between two points to be considered neighbors.
        min_points (int): Minimum number of points to form a dense cluster.
        height_threshold (float): Minimum height above ground to be considered canopy.
    """
    print(f"Reading {input_las}...")
    las = laspy.read(input_las)
    
    # 1. Filter the points
    # We only want to cluster points that are high enough to be trees/tall shrubs.
    # We assume you have the 'HeightAboveGround' dimension from the PDAL step.
    # If not, you can swap this to use the raw Z value, but HAG is much better.
    try:
        hag = las.HeightAboveGround
    except AttributeError:
        print("Error: 'HeightAboveGround' dimension not found. Please run PDAL HAG first.")
        return

    # Create a boolean mask: True for points > height_threshold
    canopy_mask = hag > height_threshold
    
    # Extract only the XYZ coordinates of the canopy points
    canopy_points = np.vstack((las.x[canopy_mask], 
                               las.y[canopy_mask], 
                               las.z[canopy_mask])).transpose()
    
    # 2. Convert to Open3D PointCloud
    print(f"Clustering {len(canopy_points)} canopy points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(canopy_points)
    
    # 3. Run DBSCAN
    # This returns an array of cluster labels. 
    # Label -1 means "noise" (points that didn't fit into any dense cluster).
    # Labels 0, 1, 2... are unique tree IDs.
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    max_label = labels

    las.write(output_las)

def classify_vegetation_rules(input_las, output_las):
    """
    Classifies points into Shrubs vs Trees using hardcoded logic rules.
    """
    print(f"Reading {input_las}...")
    las = laspy.read(input_las)
    
    # Ensure HAG exists
    try:
        hag = las.HeightAboveGround
    except AttributeError:
        print("Error: 'HeightAboveGround' not found. Run PDAL HAG first.")
        return

    # 1. Initialize a new classification array (0 = Unclassified)
    # We will use standard LAS classification codes: 3 = Low Veg (Shrub), 5 = High Veg (Tree)
    veg_class = np.zeros(len(las.points), dtype=np.uint8)
    
    # 2. Define the Shrub Mask
    # Rule: Height between 0.5m and 3.0m.
    is_shrub = (hag >= 0.5) & (hag < 3.0)
    
    # 3. Define the Tree Mask
    # Rule: Height > 3.0m OR (Height > 1.5m AND it generated multiple returns)
    # The second condition catches shorter trees with complex, branching canopies.
    is_tree = (hag >= 3.0) | ((hag >= 1.5) & (las.number_of_returns > 1))
    
    # 4. Apply the masks
    # We apply trees second so that if a point meets both criteria, it defaults to tree
    veg_class[is_shrub] = 3
    veg_class[is_tree] = 5
    
    # 5. Save the results
    # We can either overwrite the native LAS 'classification' dimension or create a new one
    las.classification = veg_class
    
    print("Writing results...")
    las.write(output_las)
    print(f"Success! Classification saved to {output_las}.")

####################
# ***part of classifying tree vs shrub
# come back to this during pipeline completion. 
# functions below are meant to match classifications from different bounding boxes by comparing overlapping points.
####################
def get_overlap_mask(points, bbox_overlap):
    """
    Returns a boolean mask of points that fall strictly within the overlap bounding box.
    bbox_overlap format: [min_x, max_x, min_y, max_y]
    """
    min_x, max_x, min_y, max_y = bbox_overlap
    x = points[:, 0]
    y = points[:, 1]
    
    mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    return mask

def align_tile_labels(points_A, labels_A, points_B, labels_B, bbox_overlap):
    """
    Aligns the K-Means labels of Tile B to match the semantic meaning of Tile A 
    using the shared spatial overlap.
    
    Returns:
        numpy.ndarray: The remapped labels for Tile B.
    """
    print("Isolating overlap regions...")
    # 1. Isolate the points that live in the shared boundary
    mask_A = get_overlap_mask(points_A, bbox_overlap)
    mask_B = get_overlap_mask(points_B, bbox_overlap)
    
    overlap_points_A = points_A[mask_A]
    overlap_labels_A = labels_A[mask_A]
    
    overlap_points_B = points_B[mask_B]
    overlap_labels_B = labels_B[mask_B]
    
    if len(overlap_points_A) == 0 or len(overlap_points_B) == 0:
        raise ValueError("No points found inside the provided overlap bounding box.")

    print(f"Found {len(overlap_points_A)} points in Tile A's overlap and {len(overlap_points_B)} in Tile B's.")

    # 2. Match points perfectly using a KD-Tree
    # We query Tile A's tree using Tile B's points to find the exact nearest neighbors
    print("Matching points across tiles...")
    tree_A = cKDTree(overlap_points_A[:, :2]) # Match based on X, Y
    distances, indices_A = tree_A.query(overlap_points_B[:, :2], k=1)
    
    # 3. Build a mapping dictionary based on majority vote
    # We map what Tile B *thinks* it is, to what Tile A *knows* it is
    print("Calculating label consensus...")
    unique_labels_B = np.unique(labels_B)
    label_mapping = {}
    
    for label_b in unique_labels_B:
        # Find all points in the overlap where Tile B predicted 'label_b'
        b_indices_for_this_label = np.where(overlap_labels_B == label_b)[0]
        
        # If this label doesn't exist in the overlap, we can't map it safely
        if len(b_indices_for_this_label) == 0:
            print(f"Warning: Label {label_b} not found in overlap. Leaving as is.")
            label_mapping[label_b] = label_b
            continue
            
        # Get the corresponding indices in Tile A's overlap
        matched_indices_in_A = indices_A[b_indices_for_this_label]
        
        # Get what Tile A labeled those exact same points
        corresponding_labels_in_A = overlap_labels_A[matched_indices_in_A]
        
        # Find the most common label in Tile A for this group (Majority Vote)
        # np.bincount tallies the occurrences, argmax gets the most frequent
        most_frequent_label_A = np.bincount(corresponding_labels_in_A).argmax()
        
        label_mapping[label_b] = most_frequent_label_A
        print(f"Tile B Label {label_b}  --->  Maps to Tile A Label {most_frequent_label_A}")

    # 4. Apply the mapping to the ENTIRETY of Tile B
    print("Applying remapped labels to Tile B...")
    remapped_labels_B = np.copy(labels_B)
    for old_label, new_label in label_mapping.items():
        remapped_labels_B[labels_B == old_label] = new_label
        
    return remapped_labels_B

######################
# HEIGHT RASTERS
######################


######################
# HELPERS
######################
def plot_als(input_path):
    las = laspy.read(input_path)

    subset_size = 50000 
    random_indices = np.random.choice(len(las.x), size=subset_size, replace=False)

    x_als, y_als, z_als = las.x, las.y, las.z
    # 2. We create an interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_als[random_indices],
        y=y_als[random_indices],
        z=z_als[random_indices],
        mode='markers',
        marker=dict(
            size=2,                     
            color=z_als[:subset_size],   # Color points by height (just to make it look nicer :D)
            colorscale='Viridis',        
            opacity=0.8                  
        )
    )])

    # 3. We add a few details and show the figure
    fig.update_layout(
        title="Airborne Laser Scanning (ALS) Point Cloud",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)  
    )

    fig.show()

def process_las_roughness(input_path, output_path, k=30):
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    roughness = calculate_roughness_knn(points, k=k)
    
    las.add_extra_dim(laspy.ExtraBytesParams(name="roughness", type=np.float32))
    las.roughness = roughness
    
    las.write(output_path)
    print(f"Saved processed point cloud to {output_path}")
