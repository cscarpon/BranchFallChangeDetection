import numpy as np
import laspy
from plyfile import PlyData, PlyElement
import open3d as o3d
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import sys

def write_voxel_obj_quads(df, voxel_size, filename):
    """
    Write an OBJ file with quads for each voxel cube.
    
    Parameters:
    - df: DataFrame with 'X_1', 'Y_1', 'Z_1' as voxel centers
    - voxel_size: size of the voxel cube edge
    - filename: output OBJ file path
    """
    half = voxel_size / 2
    verts = []
    faces = []
    vert_idx = 1  # OBJ uses 1-based indexing

    # 8 cube corners in fixed order
    cube_offsets = np.array([
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ]) * half

    # Each face as a quad (indices of the 8 cube vertices)
    cube_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7],  # left
    ]

    for _, row in df.iterrows():
        center = np.array([row['VoxPos_X_1'], row['VoxPos_Y_1'], row['VoxPos_Z_1']])
        cube_vertices = center + cube_offsets
        verts.extend(cube_vertices.tolist())

        base = vert_idx
        for face in cube_faces:
            faces.append([base + i for i in face])
        vert_idx += 8

    # Write OBJ
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {' '.join(map(str, face))}\n")

    print(f"Wrote {len(df)} voxel cubes to {filename} (with quad faces).")

def write_voxel_obj(df, voxel_size, filename):
    """
    Write an OBJ file from a DataFrame of voxel center positions.
    
    Parameters:
    - df: DataFrame with columns 'X_1', 'Y_1', 'Z_1'
    - voxel_size: edge length of each voxel cube
    - filename: output .obj filename
    """
    half = voxel_size / 2
    verts = []
    faces = []
    vert_idx = 1  # .obj indexing starts at 1

    # Define relative offsets for a cube centered at origin
    cube_offsets = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ]) * half

    # Faces (triangles) from the 8 vertices of a cube
    cube_faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ]

    for _, row in df.iterrows():
        center = np.array([row['VoxPos_X_1'], row['VoxPos_Y_1'], row['VoxPos_Z_1']])
        cube_vertices = center + cube_offsets
        verts.extend(cube_vertices.tolist())

        base_idx = vert_idx
        for face in cube_faces:
            faces.append([base_idx + i for i in face])
        vert_idx += 8  # each cube has 8 vertices

    # Write to OBJ
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {' '.join(map(str, face))}\n")

    print(f"OBJ file written to {filename} with {len(df)} voxels.")


def extract_intact_branch_from_axis(axis_id, qsm_df, paths_df):
    axis_cylinders = qsm_df[qsm_df['axis_ID'] == axis_id]

    if axis_cylinders.empty:
        print(f"Axis {axis_id} has no cylinders.")
        return pd.DataFrame()

    cyl_id_start = axis_cylinders['cyl_ID'].min()
    id_paths = paths_df[paths_df['cyl_ID'] == cyl_id_start]['ID_Path'].unique()

    matching_cyls = paths_df[
        (paths_df['ID_Path'].isin(id_paths)) & 
        (paths_df['cyl_ID'] >= cyl_id_start)
    ]['cyl_ID'].unique()

    branch_df = qsm_df[qsm_df['cyl_ID'].isin(matching_cyls)].copy()
    branch_df['branch_ID'] = axis_id
    branch_df['label'] = qsm_df.label
    
    return branch_df


def extract_branch_from_axis(axis_id, qsm_df, paths_df, used_cyl_ids):
    axis_cylinders = qsm_df[qsm_df['axis_ID'] == axis_id]

    if axis_cylinders.empty:
        print(f"Axis {axis_id} has no cylinders.")
        return pd.DataFrame(), used_cyl_ids

    cyl_id_start = axis_cylinders['cyl_ID'].min()
    id_paths = paths_df[paths_df['cyl_ID'] == cyl_id_start]['ID_Path'].unique()

    matching_cyls = paths_df[
        (paths_df['ID_Path'].isin(id_paths)) & 
        (paths_df['cyl_ID'] >= cyl_id_start)
    ]['cyl_ID'].unique()

    # Remove already-used cylinders
    new_cyls = [cyl for cyl in matching_cyls if cyl not in used_cyl_ids]

    if not new_cyls:
        return pd.DataFrame(), used_cyl_ids

    used_cyl_ids.update(new_cyls)  # track them as used

    branch_df = qsm_df[qsm_df['cyl_ID'].isin(new_cyls)].copy()
    branch_df['branch_ID'] = axis_id
    # branch_df['label'] = branch_df['label']  # Keep label if relevant

    return branch_df, used_cyl_ids

def points_in_voxels(points, voxel_df, voxel_size):
    """
    Filter points within voxel boundaries.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Point coordinates in the same physical coordinate system as the voxels.
    voxel_df : pandas.DataFrame
        DataFrame describing voxels. It may contain voxel locations in one of
        these forms:
          - 'X', 'Y', 'Z'
          - 'X_1', 'Y_1', 'Z_1'
          - 'VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1'
          - 'VoxPos_X', 'VoxPos_Y', 'VoxPos_Z'
          - 'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z' (integer indices)
    voxel_size : float
        Edge length of a voxel, in the same units as `points` coordinates.
    """
    voxel_df = voxel_df.copy()

    # First, normalize older naming convention X_1/Y_1/Z_1 → X/Y/Z
    rename_map = {"X_1": "X", "Y_1": "Y", "Z_1": "Z"}
    voxel_df.rename(columns={k: v for k, v in rename_map.items() if k in voxel_df.columns},
                    inplace=True)

    # Decide which coordinate columns to use
    cols_xyz      = ['X', 'Y', 'Z']
    cols_suffix_1 = ['VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1']
    cols_base     = ['VoxPos_X',   'VoxPos_Y',   'VoxPos_Z']
    cols_label    = ['VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z']

    if all(c in voxel_df.columns for c in cols_xyz):
        # Already in physical coordinates
        voxel_coords = voxel_df[cols_xyz].values.astype(float)

    elif all(c in voxel_df.columns for c in cols_suffix_1):
        voxel_coords = voxel_df[cols_suffix_1].values.astype(float)

    elif all(c in voxel_df.columns for c in cols_base):
        voxel_coords = voxel_df[cols_base].values.astype(float)

    elif all(c in voxel_df.columns for c in cols_label):
        # Integer voxel indices → convert to physical coordinates
        voxel_coords = voxel_df[cols_label].values.astype(float) * float(voxel_size)

    else:
        raise KeyError(
            "points_in_voxels: could not find voxel coordinate columns.\n"
            f"Expected one of:\n"
            f"  {cols_xyz}\n"
            f"  {cols_suffix_1}\n"
            f"  {cols_base}\n"
            f"  {cols_label}\n"
            f"but got:\n{list(voxel_df.columns)}"
        )

    # Now do the same bounding-box test as your original implementation
    voxel_min = voxel_coords - voxel_size / 2.0
    voxel_max = voxel_coords + voxel_size / 2.0

    inside_points = []
    inside_indices = []

    for i, point in enumerate(points):
        inside = np.any(
            (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
            (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
            (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
        )
        if inside:
            inside_indices.append(i)
            inside_points.append(point)

    return np.array(inside_indices), np.array(inside_points)

def original_points_in_voxels(tree1_filename, adjacent_df, voxel_size):
    """
    Filter points within voxel boundaries.
    """
    adjacent_df = adjacent_df.rename(columns={"X_1": "X", "Y_1": "Y", "Z_1": "Z"})
    voxel_coords = adjacent_df[['X', 'Y', 'Z']].values
    voxel_min = voxel_coords - voxel_size / 2
    voxel_max = voxel_coords + voxel_size / 2
    
    with laspy.open(tree1_filename) as las_file:
        las = las_file.read()
        tree_points = np.vstack((las.x, las.y, las.z)).T
    
    inside_points = []
    for point in tree_points:
        inside = np.any(
            (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
            (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
            (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
        )
        if inside:
            inside_points.append(point)
    
    return np.array(inside_points)

def dbscan_connectivity(df_diff, voxel_size, min_samples):
    """
    Apply DBSCAN to find clusters of connected voxels.

    df_diff: DataFrame containing voxel positions for the tree that changed.
             Expected to have either:
               - 'VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1'
             or   'VoxPos_X',  'VoxPos_Y',  'VoxPos_Z'
             or   'VoxLabel_X','VoxLabel_Y','VoxLabel_Z'
    """
    # Possible coordinate column sets
    cols_suffix_1 = ['VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1']
    cols_base     = ['VoxPos_X',   'VoxPos_Y',   'VoxPos_Z']
    cols_label    = ['VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z']

    # Decide how to build voxel coordinates
    if all(c in df_diff.columns for c in cols_suffix_1):
        # Physical coords for tree1 with _1 suffix
        voxels = df_diff[cols_suffix_1].values.astype(float)

    elif all(c in df_diff.columns for c in cols_base):
        # Physical coords without suffix
        voxels = df_diff[cols_base].values.astype(float)

    elif all(c in df_diff.columns for c in cols_label):
        # Integer voxel labels – convert to physical coordinates
        # so eps (defined in meters) still makes sense.
        voxels = df_diff[cols_label].values.astype(float) * float(voxel_size)

    else:
        # Helpful debug if the expected columns are missing
        raise KeyError(
            "dbscan_connectivity: could not find voxel coordinate columns.\n"
            f"Expected one of:\n"
            f"  {cols_suffix_1}\n"
            f"  {cols_base}\n"
            f"  {cols_label}\n"
            f"but got:\n{list(df_diff.columns)}"
        )

    # Apply DBSCAN with a radius slightly greater than the voxel diagonal
    eps = voxel_size * np.sqrt(3) + 0.5 * voxel_size  # adjacency in physical space
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(voxels)

    # Avoid pandas SettingWithCopy warnings
    df_diff = df_diff.copy()
    df_diff.loc[:, 'Component'] = db.labels_

    # Number of unique clusters, ignoring noise (label = -1)
    num_components = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # Filter out noise points (optional)
    df_connected = df_diff[df_diff['Component'] != -1].reset_index(drop=True)

    return df_connected, num_components

def load_tree_data(mat, tree_index):
    tree_data = mat['OptQSM'][0][tree_index][2][0][0]

    tree_data_array = np.array([
        tree_data[i].flatten()[0] for i in range(23)
    ]).reshape(1, -1)
    
    columns = ['TotalVolume', 'TrunkVolume', 'BranchVolume', 
               'TreeHeight', 'TrunkLength', 'BranchLength', 
               'TotalLength', 'NumberBranches', 'MaxBranchOrder',
               'TrunkArea', 'BranchArea', 'TotalArea', 'DBHqsm', 'DBHcyl',
               'CrownDiamAve', 'CrownDiamMax', 'CrownAreaConv', 'CrownAreaAlpha',
               'CrownBaseHeight', 'CrownLength', 'CrownRatio', 'CrownVolumeConv',
               'CrownVolumeAlpha']
    
    tree_data_df = pd.DataFrame(tree_data_array, columns=columns)

    return tree_data_df 

def load_cylinders(mat, tree_index):
    cylinders = mat['OptQSM'][0][tree_index][0][0][0]

    # cylinders_array = np.column_stack([cylinder.flatten() for cylinder in cylinders])
    cylinders_array = np.array([cylinders[0].flatten(),
                                cylinders[1].flatten(),
                                cylinders[2][:, 0], 
                                cylinders[2][:, 1], 
                                cylinders[2][:, 2], 
                                cylinders[3][:, 0], 
                                cylinders[3][:, 1], 
                                cylinders[3][:, 2],
                                cylinders[4].flatten(),
                                cylinders[5].flatten(),
                                cylinders[6].flatten(),
                                cylinders[7].flatten(),
                                cylinders[8].flatten(),
                                cylinders[9].flatten(),
                                cylinders[10].flatten(),
                                cylinders[11].flatten(),
                                cylinders[12].flatten()]).T

    cylinders_df = pd.DataFrame(cylinders_array, columns=['radius', 'length', 
                                                          'startx', 'starty', 'startz', 
                                                          'axisx', 'axisy', 'axisz',
                                                          'parent', 'extension', 'added', 'UnmodRadius', 'branch',
                                                          'SurfCov', 'mad', 'BranchOrder', 'PositionInBranch'])

    # Add coordinates of the end of the cylinders
    cylinders_df['endx'] = cylinders_df['startx'] + cylinders_df['axisx'] * cylinders_df['length']
    cylinders_df['endy'] = cylinders_df['starty'] + cylinders_df['axisy'] * cylinders_df['length']
    cylinders_df['endz'] = cylinders_df['startz'] + cylinders_df['axisz'] * cylinders_df['length']
    
    return cylinders_df 

def load_branches(mat, tree_index):
    branches = mat['OptQSM'][0][tree_index][1][0][0]

    branches_array = np.column_stack([branch.flatten() for branch in branches])
    branches_df = pd.DataFrame(branches_array, columns=['order', 'parent', 
                                                         'diameter', 'volume', 'area', 
                                                         'length', 'angle', 'height',
                                                          'azimuth', 'zenith'])

    # Add broken column to branches_df and initialize all branches to not broken at first
    branches_df['broken'] = False
    branches_df['broken'] = branches_df['broken'].astype(bool)

    return branches_df

def original_points_in_voxels_pcd(tree1_filename, adjacent_df, voxel_size):
    """
    Filter points within voxel boundaries for PCD files.
    """
    # Adjust the dataframe to align voxel boundaries
    adjacent_df = adjacent_df.rename(columns={"X_1": "X", "Y_1": "Y", "Z_1": "Z"})
    voxel_coords = adjacent_df[['X', 'Y', 'Z']].values
    voxel_min = voxel_coords - voxel_size / 2
    voxel_max = voxel_coords + voxel_size / 2

    # Load PCD file
    pcd = o3d.io.read_point_cloud(tree1_filename)
    points = np.asarray(pcd.points)

    # Filter points within voxel boundaries
    inside_points = []
    for point in points:
        inside = np.any(
            (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
            (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
            (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
        )
        if inside:
            inside_points.append(point)

    return np.array(inside_points)

def compare_connectivity(voxels, voxel_size):

    voxels = voxels[['VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1']].values
    
    # Tableau pour stocker les connexions (indices des voxels adjacents)
    adjacency_list = []
    adjacent_voxels_indices = set()  # Ensemble pour conserver les indices uniques des voxels adjacents
    
    # Parcourir chaque voxel pour trouver ses voisins
    for i, voxel in enumerate(voxels):
        # Calculer les différences au carré entre le voxel courant et tous les autres voxels
        squared_differences = (voxels - voxel) ** 2
        distances = np.sum(squared_differences, axis=1)
                
        # Vérifier la distance Euclidienne pour trouver les voxels adjacents
        is_adjacent = distances <= (voxel_size ** 2) * 3 + 0.1 * voxel_size
        
        # Trouver les indices des voxels adjacents
        adjacent_indices = np.where(is_adjacent)[0]
                
        # Connectivité
        if len(adjacent_indices) > 5:
            # Ajouter les connexions dans la liste et mettre à jour l'ensemble des indices
            for j in adjacent_indices:
                if i < j:  # Évite les doublons
                    adjacency_list.append((i, j))
                adjacent_voxels_indices.add(i)
                adjacent_voxels_indices.add(j)
    
    # Créer un nouveau DataFrame avec uniquement les voxels adjacents
    df_adjacent = df_diff.iloc[list(adjacent_voxels_indices)].reset_index(drop=True)
    return df_adjacent

# Vectorized voxel-cylinder intersection
def find_inside_cylinders(cylinders_df, voxel_min, voxel_max, v_size):
    # Convert cylinder data to numpy arrays
    starts = cylinders_df[['startx', 'starty', 'startz']].to_numpy()
    ends = cylinders_df[['endx', 'endy', 'endz']].to_numpy()
    
    # Vectorized comparison: Check if starts or ends are within voxel bounds
    starts_inside = np.any(
        (starts[:, None, :] >= voxel_min.values) & (starts[:, None, :] <= voxel_max.values), axis=2
    )
    ends_inside = np.any(
        (ends[:, None, :] >= voxel_min.values) & (ends[:, None, :] <= voxel_max.values), axis=2
    )
    
    # Combine results
    inside_mask = np.any(starts_inside | ends_inside, axis=1)
    return cylinders_df[inside_mask]

# Function to write a PCD file
def write_pcd(file_path, points):
    with open(file_path, 'w') as f:
        # Write the PCD file header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")  # Each coordinate is a 4-byte float
        f.write("TYPE F F F\n")  # Data types are all floats
        f.write("COUNT 1 1 1\n")  # One value per field
        f.write(f"WIDTH {points.shape[0]}\n")  # Number of points
        f.write("HEIGHT 1\n")  # Unordered point cloud (1 row)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")  # Default viewpoint
        f.write(f"POINTS {points.shape[0]}\n")  # Total number of points
        f.write("DATA ascii\n")  # Data format is ASCII
        
        # Write point data
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")  # Save points with 6 decimal precision

def generate_all_cylinder_meshes_from_df(df, num_segments=12):
    starts = df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
    ends = df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
    radii = df["radius_cyl"].to_numpy(dtype=float)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(len(radii)):
        start = starts[i]
        end = ends[i]
        radius = radii[i]

        v, f = generate_cylinder_vertices_faces_adtree(start, end, radius, num_segments)
        f_offset = [[idx + vertex_offset for idx in face] for face in f]

        all_vertices.append(v)
        all_faces.extend(f_offset)
        vertex_offset += v.shape[0]

    return np.vstack(all_vertices), all_faces

def generate_all_cylinder_meshes_adtree(starts, ends, radii, num_segments=12):
    """
    Génère un maillage de plusieurs cylindres à partir de listes de départs, fins et rayons.

    Paramètres :
    - starts : array (N, 3) des points de départ.
    - ends : array (N, 3) des points de fin.
    - radii : array (N,) des rayons.
    - num_segments : nombre de segments pour approximer la base circulaire.

    Retour :
    - vertices : tableau de sommets combinés.
    - faces : liste de faces avec index global ajusté.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(len(radii)):
        start = starts[i]
        end = ends[i]
        radius = radii[i]

        v, f = generate_cylinder_vertices_faces_adtree(start, end, radius, num_segments)

        # Décalage des indices de faces
        f_offset = [[idx + vertex_offset for idx in face] for face in f]
        all_vertices.append(v)
        all_faces.extend(f_offset)

        vertex_offset += v.shape[0]

    return np.vstack(all_vertices), all_faces

def generate_cylinder_vertices_faces_adtree(start, end, radius, num_segments=12):
    """
    Generate vertices and faces for a cylinder with a given start point, end point, radius, and number of segments.

    Parameters:
    - start: Starting point of the cylinder as a numpy array [x, y, z].
    - end: Ending point of the cylinder as a numpy array [x, y, z].
    - radius: Radius of the cylinder.
    - num_segments: Number of segments to approximate the circular base.

    Returns:
    - vertices: Array of vertices (numpy array).
    - faces: List of faces (as indices into the vertices array).
    """
    # Calculate the axis and length
    axis = end - start
    length = np.linalg.norm(axis)
    axis /= length

    # Generate orthogonal vectors for the circular base
    if np.allclose(axis, [0, 0, 1]) or np.allclose(axis, [0, 0, -1]):
        ortho_vector = np.array([1, 0, 0], dtype=float)
    else:
        ortho_vector = np.cross(axis, [0, 0, 1]).astype(float)
    ortho_vector /= np.linalg.norm(ortho_vector)
    ortho_vector2 = np.cross(axis, ortho_vector)

    # Generate circle points on both ends
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle_points = radius * (np.outer(np.cos(theta), ortho_vector) + np.outer(np.sin(theta), ortho_vector2))

    # Translate circle points to start and end positions
    bottom_circle = start + circle_points
    top_circle = end + circle_points

    # Vertices for the cylinder
    vertices = np.vstack([bottom_circle, top_circle])

    # Generate faces
    faces = []
    for i in range(num_segments):
        # Wrap index around the circle
        next_i = (i + 1) % num_segments
        # Side faces (two triangles per segment)
        faces.append([i, next_i, i + num_segments])        # Triangle 1
        faces.append([next_i, next_i + num_segments, i + num_segments])  # Triangle 2
    # Add the top and bottom faces (optional)
    bottom_center = len(vertices)
    top_center = bottom_center + 1
    vertices = np.vstack([vertices, [start], [end]])  # add centers to vertices

    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        # Bottom and top faces
        faces.append([i, next_i, bottom_center])       # Bottom cap
        faces.append([i + num_segments, next_i + num_segments, top_center])  # Top cap

    return vertices, faces

def write_ply_adtree(filename, vertices, faces):
    """
    Write vertices and faces to a PLY file.

    Parameters:
    - filename: The name of the file to write.
    - vertices: Array of vertices.
    - faces: List of faces.
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def create_cylinder_ply_from_skeleton(skeleton, filename, num_segments=12):
    """
    Create a PLY file containing cylinders based on a skeleton.

    Parameters:
    - skeleton: List of vertices, each containing [x, y, z, radius].
    - filename: The name of the PLY file to create.
    - num_segments: Number of segments to approximate the circular base.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(0, skeleton.elements[0].count-1):
        start = np.array(skeleton['vertex'][i].tolist()[:3])
        end = np.array(skeleton['vertex'][i+1].tolist()[:3])
        radius = skeleton['vertex'][i].tolist()[-1]

        vertices, faces = generate_cylinder_vertices_faces(start, end, radius, num_segments)

        # Adjust face indices
        faces = [[v + vertex_offset for v in face] for face in faces]

        all_vertices.extend(vertices)
        all_faces.extend(faces)
        vertex_offset = len(all_vertices)

    write_ply_adtree(filename, np.array(all_vertices), all_faces)

def generate_cylinder_vertices_faces(start, axis, radius, length, num_segments=12):
    """
    Generate vertices and faces for a cylinder with a given start point, axis direction, radius, and length.

    Parameters:
    - start: Starting point of the cylinder as a numpy array [x, y, z].
    - axis: Unit vector indicating the cylinder's axis direction.
    - radius: Radius of the cylinder.
    - length: Length of the cylinder.
    - num_segments: Number of segments to approximate the circular base.

    Returns:
    - vertices: Array of vertices (numpy array).
    - faces: List of faces (as indices into the vertices array).
    """
    # Calculate the end point using the axis and length
    end = start + axis * length

    # Generate orthogonal vectors for the circular base
    if np.allclose(axis, [0, 0, 1]) or np.allclose(axis, [0, 0, -1]):
        ortho_vector = np.array([1, 0, 0], dtype=float)
    else:
        ortho_vector = np.cross(axis, [0, 0, 1]).astype(float)
    ortho_vector /= np.linalg.norm(ortho_vector)
    ortho_vector2 = np.cross(axis, ortho_vector)

    # Generate circle points on both ends
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle_points = radius * (np.outer(np.cos(theta), ortho_vector) + np.outer(np.sin(theta), ortho_vector2))

    # Translate circle points to start and end positions
    bottom_circle = start + circle_points
    top_circle = end + circle_points

    # Vertices for the cylinder
    vertices = np.vstack([bottom_circle, top_circle])

    # Generate faces
    faces = []
    for i in range(num_segments):
        # Wrap index around the circle
        next_i = (i + 1) % num_segments
        # Side faces (two triangles per segment)
        faces.append([i, next_i, i + num_segments])        # Triangle 1
        faces.append([next_i, next_i + num_segments, i + num_segments])  # Triangle 2
    # Add the top and bottom faces (optional)
    bottom_center = len(vertices)
    top_center = bottom_center + 1
    vertices = np.vstack([vertices, [start], [end]])  # add centers to vertices

    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        # Bottom and top faces
        faces.append([i, next_i, bottom_center])       # Bottom cap
        faces.append([i + num_segments, next_i + num_segments, top_center])  # Top cap

    return vertices, faces

def generate_unique_colors(n):
    """Generate `n` distinct colors in (R, G, B) format using HSV space."""
    np.random.seed(42)  # Ensure consistent colors across runs
    
    # Evenly distribute hues in HSV space
    hues = np.linspace(0, 1, num=int(n), endpoint=False)  # Avoid duplicate red at 1.0
    colors = [mcolors.hsv_to_rgb((h, 1, 1)) * 255 for h in hues]  # Convert to RGB
    
    # Convert float values to integers (R, G, B)
    colors = [(int(r), int(g), int(b)) for r, g, b in colors]  
    
    return colors  # Returns a list of (R, G, B) tuples

def write_ply_file(filename, cylinders_df, colors_param=None):
    """Creates a PLY file from cylinders in a dataframe with a specified or default color."""
    
    all_vertices = []
    all_faces = []
    all_colors = []  # Store colors for each vertex
    offset = 0
    
    if colors_param == None:
        colors = {}
        colors = {i: color for i, color in enumerate([
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),   # Magenta
            (255, 120, 255),
            (255, 120, 120),
            (120, 120, 255),
            (120, 120, 120),
            (50, 0, 0),
            (50, 50, 0),
            (50, 50, 50)
            
        ])}        
    
    for idx, row in cylinders_df.iterrows():
    
        vertices, faces = generate_cylinder_vertices_faces(row[['startX', 'startY', 'startZ']].to_numpy(),
                                                           row[['axisx', 'axisy', 'axisz']].to_numpy(),
                                                           row['radius'], row['length'], num_segments=12)
        
        if colors_param == None:
            r, g, b = colors[int(row.radius)]
            # r,g,b = colors_param  
            
        else:
            r,g,b = colors_param    

        
        # Append vertices and corresponding colors
        all_vertices.extend(vertices)
        all_colors.extend([(r, g, b)] * len(vertices))  # Apply the same color to all vertices
        
        # Adjust face indices and append them to the global face list
        faces = [[idx + offset for idx in face] for face in faces]
        all_faces.extend(faces)
        
        # Update offset for the next set of vertices
        offset += len(vertices)
    
    
    
    # Create structured vertex data with color
    vertex_data = np.array([(v[0], v[1], v[2], c[0], c[1], c[2]) for v, c in zip(all_vertices, all_colors)],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    print(vertex_data)
    
    face_data = np.array([(face,) for face in all_faces],
                         dtype=[('vertex_indices', 'i4', (3,))])
    
    # Create PLY elements
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')
    
    # Write PLY file
    PlyData([vertex_element, face_element], text=True).write(filename)
    print("File written:")
    print(filename)

def original_points_in_voxels(tree1_filename, voxels, voxel_size):
    """
    Filter points within voxel boundaries.
    """
    voxels_df = voxels.rename(columns={"X_1": "X", "Y_1": "Y", "Z_1": "Z"})
    voxel_coords = voxels_df[['X', 'Y', 'Z']].values
    voxel_min = voxel_coords - voxel_size / 2
    voxel_max = voxel_coords + voxel_size / 2
    
    with laspy.open(tree1_filename) as las_file:
        las = las_file.read()
        points = np.vstack((las.x, las.y, las.z)).T
    
    inside_points = []
    for point in points:
        inside = np.any(
            (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
            (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
            (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
        )
        if inside:
            inside_points.append(point)
    
    return np.array(inside_points)