import numpy as np
import laspy
from plyfile import PlyData, PlyElement
import open3d as o3d
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import sys

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

def points_in_voxels(points, voxel_df, voxel_size, ref_min_point):
    """
    points: (N,3) world coordinates (your LiDAR points or cylinder endpoints)
    voxel_df: DataFrame with integer voxel indices VoxLabel_X/Y/Z
    voxel_size: voxel edge length, same as in voxelize()
    ref_min_point: np.array shape (3,), same as voxelize() ref_min_point
    """
    labels = voxel_df[["VoxLabel_X", "VoxLabel_Y", "VoxLabel_Z"]].to_numpy().astype(float)

    # Compute voxel min and max in world coordinates
    voxel_min = ref_min_point + labels * voxel_size
    voxel_max = voxel_min + voxel_size  # each voxel is a cube of size voxel_size

    inside_indices = []
    inside_points = []

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

# def points_in_voxels(points, voxel_df, voxel_size, ref_min_point):
#     """
#     points: (N,3) world coordinates (your LiDAR points or cylinder endpoints)
#     voxel_df: DataFrame with integer voxel indices VoxLabel_X/Y/Z
#     voxel_size: voxel edge length, same as in voxelize()
#     ref_min_point: np.array shape (3,), same as voxelize() ref_min_point
#     """
#     labels = voxel_df[["VoxLabel_X", "VoxLabel_Y", "VoxLabel_Z"]].to_numpy().astype(float)

#     # Compute voxel min and max in world coordinates
#     voxel_min = ref_min_point + labels * voxel_size
#     voxel_max = voxel_min + voxel_size  # each voxel is a cube of size voxel_size

#     inside_indices = []
#     inside_points = []

#     for i, point in enumerate(points):
#         inside = np.any(
#             (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
#             (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
#             (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
#         )
#         if inside:
#             inside_indices.append(i)
#             inside_points.append(point)

#     return np.array(inside_indices), np.array(inside_points)

# def points_in_voxels(points, voxel_df, voxel_size):
#     """
#     Filter points within voxel boundaries.

#     Parameters
#     ----------
#     points : array-like, shape (N, 3)
#         Point coordinates in the same physical coordinate system as the voxels.
#     voxel_df : pandas.DataFrame
#         DataFrame describing voxels. It may contain voxel locations in one of
#         these forms:
#           - 'X', 'Y', 'Z'
#           - 'X_1', 'Y_1', 'Z_1'
#           - 'VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1'
#           - 'VoxPos_X', 'VoxPos_Y', 'VoxPos_Z'
#           - 'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z' (integer indices)
#     voxel_size : float
#         Edge length of a voxel, in the same units as `points` coordinates.
#     """
#     voxel_df = voxel_df.copy()

#     # First, normalize older naming convention X_1/Y_1/Z_1 → X/Y/Z
#     rename_map = {"X_1": "X", "Y_1": "Y", "Z_1": "Z"}
#     voxel_df.rename(columns={k: v for k, v in rename_map.items() if k in voxel_df.columns},
#                     inplace=True)

#     # Decide which coordinate columns to use
#     cols_xyz      = ['X', 'Y', 'Z']
#     cols_suffix_1 = ['VoxPos_X_1', 'VoxPos_Y_1', 'VoxPos_Z_1']
#     cols_base     = ['VoxPos_X',   'VoxPos_Y',   'VoxPos_Z']
#     cols_label    = ['VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z']

#     if all(c in voxel_df.columns for c in cols_xyz):
#         # Already in physical coordinates
#         voxel_coords = voxel_df[cols_xyz].values.astype(float)

#     elif all(c in voxel_df.columns for c in cols_suffix_1):
#         voxel_coords = voxel_df[cols_suffix_1].values.astype(float)

#     elif all(c in voxel_df.columns for c in cols_base):
#         voxel_coords = voxel_df[cols_base].values.astype(float)

#     elif all(c in voxel_df.columns for c in cols_label):
#         # Integer voxel indices → convert to physical coordinates
#         voxel_coords = voxel_df[cols_label].values.astype(float) * float(voxel_size)

#     else:
#         raise KeyError(
#             "points_in_voxels: could not find voxel coordinate columns.\n"
#             f"Expected one of:\n"
#             f"  {cols_xyz}\n"
#             f"  {cols_suffix_1}\n"
#             f"  {cols_base}\n"
#             f"  {cols_label}\n"
#             f"but got:\n{list(voxel_df.columns)}"
#         )

#     # Now do the same bounding-box test as your original implementation
#     voxel_min = voxel_coords - voxel_size / 2.0
#     voxel_max = voxel_coords + voxel_size / 2.0

#     inside_points = []
#     inside_indices = []

#     for i, point in enumerate(points):
#         inside = np.any(
#             (voxel_min[:, 0] <= point[0]) & (point[0] <= voxel_max[:, 0]) &
#             (voxel_min[:, 1] <= point[1]) & (point[1] <= voxel_max[:, 1]) &
#             (voxel_min[:, 2] <= point[2]) & (point[2] <= voxel_max[:, 2])
#         )
#         if inside:
#             inside_indices.append(i)
#             inside_points.append(point)

#     return np.array(inside_indices), np.array(inside_points)

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
