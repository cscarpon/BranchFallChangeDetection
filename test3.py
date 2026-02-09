import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, cKDTree
# import matplotlib.pyplot as plt

from VoxelChangeDetector import VoxelChangeDetector
from tools import dbscan_connectivity, points_in_voxels, extract_branch_from_axis

# plt.close('all')

# parts = ["part1", "part2a", "part2b", "part3", "part4", "part5","part6"]
# parts = ["part4", "part5","part6"]
# parts = ["part2a", "part2b"]
# parts = ["part5"]
# parts = ["part1", "part2a"]
parts = ["part2a"]

fallen_branches_df_columns = ["part", "tree_index", "essence",
                              "branch_id", "volume", "length",
                              "surface", "max_radius", "angle", "origin2com"]
tree_level_df_columns = ["part", "tree_index_str", "essence", "lost_volume_ratio"]

# Parameters
# for kd-tree search
radius = 0.25
min_neighbors = 1

# Define voxel size
voxel_size = 0.5

# voxel dbscan
min_samples = 30

current_dir = rf"D:\Karl\hydro\dataset\Working\single_trees"
root_path = rf"{current_dir}\part2a"
print(root_path)

trees_2022 = os.path.join(root_path, "2022")
trees_2023 = os.path.join(root_path, "2023")

sorted_trees_2022 = sorted([f for f in os.listdir(trees_2022) if f.endswith('.xyz')], 
                            key=lambda x: int(x.split(' ')[2]))

# for part in parts: Removed the parts loop
    
fallen_branches_rows_for_part = []
tree_level_rows_for_part = []

# current_dir = os.getcwd()
root_path = rf"{current_dir}\part2a"
trees_2022 = os.path.join(root_path, "2022", "matched_xyz_species")
trees_2023 = os.path.join(root_path, "2023", "matched_xyz_species")

sorted_trees_2022 = sorted([f for f in os.listdir(trees_2022) if f.endswith('.xyz')], 
                            key=lambda x: int(x.split(' ')[2]))
sorted_trees_2023 = sorted([f for f in os.listdir(trees_2023) if f.endswith('.xyz')], 
                            key=lambda x: int(x.split(' ')[2]))

# QSM files
adtreeqsm_folder = os.path.join(root_path, r"2022\adtree_qsm3")

original_sorted_qsms = sorted(
    [f for f in os.listdir(adtreeqsm_folder) if 'qsm_origins' in f and f.endswith('.csv')],
    key=lambda x: int(x.split('_')[0])
)

print(original_sorted_qsms)

archi_sorted_qsm = sorted(
    [f for f in os.listdir(adtreeqsm_folder) if 'qsm_out' in f and f.endswith('.csv')],
    key=lambda x: int(x.split('_')[0])
)

print(archi_sorted_qsm)

# sorted_treedata = sorted(
#     [f for f in os.listdir(adtreeqsm_folder) if 'tree' in f and f.endswith('.csv')],
#     key=lambda x: int(x.split('_')[0])
# )

sorted_paths = sorted(
    [f for f in os.listdir(adtreeqsm_folder) if 'paths' in f and f.endswith('.csv')],
    key=lambda x: int(x.split('_')[0])
)

print(sorted_paths)
# tree_level_data_path = os.path.join(root_path, f"tree_level_data_{part}.csv")
# tree_level_data = pd.read_csv(tree_level_data_path)

tree_indexes = tuple(range(len(sorted_trees_2022)-1))
print(tree_indexes)

# tree_indexes = (0,)

# for tree_index in tree_indexes: Removed the tree index

tree_index= 8  
all_cluster_pts = []
    
tree_filename_2022 = os.path.join(trees_2022, sorted_trees_2022[tree_index])
tree_filename_2023 = os.path.join(trees_2023, sorted_trees_2023[tree_index])
print(f'"{tree_filename_2022}"')
print(f'"{tree_filename_2023}"')

PC = np.loadtxt(tree_filename_2023)
tree_2023 = cKDTree(PC)

essence = sorted_trees_2022[tree_index].split(" ")[4].split(".")[0]
tree_index_str = sorted_trees_2022[tree_index].split(" ")[2]
tree_index_str_2023 = sorted_trees_2022[tree_index].split(" ")[2]

if tree_index_str != tree_index_str_2023:
    print("Problem with files")
    sys.exit()

# # QSM files
# print("Reading QSM files ...")        
original_qsm_file = os.path.join(adtreeqsm_folder, original_sorted_qsms[tree_index])
archi_qsm_file = os.path.join(adtreeqsm_folder, archi_sorted_qsm[tree_index])
path_file = os.path.join(adtreeqsm_folder, sorted_paths[tree_index])

print(path_file)

original_cylinders_df = pd.read_csv(original_qsm_file, 
                            sep=',', 
                            header=None,
                            low_memory=False)
archi_cylinders_df = pd.read_csv(archi_qsm_file, 
                            sep=',', 
                            header=None,
                            low_memory=False)

paths_df = pd.read_csv(path_file, 
                        sep=',', 
                        header=None,
                        low_memory=False)

original_cylinders_df.columns = original_cylinders_df.iloc[0]
original_cylinders_df = original_cylinders_df.drop(0).reset_index(drop=True)

archi_cylinders_df .columns = archi_cylinders_df.iloc[0]
archi_cylinders_df = archi_cylinders_df .drop(0).reset_index(drop=True)

paths_df.columns = paths_df.iloc[0]
paths_df = paths_df.drop(0).reset_index(drop=True)
paths_df.cyl_ID = paths_df.cyl_ID.astype(int)

cylinders_df = archi_cylinders_df.copy()
cylinders_df.radius_cyl = original_cylinders_df.radius_cyl
cylinders_df.length = original_cylinders_df.length
cylinders_df.volume= original_cylinders_df.volume
cylinders_df.cyl_ID = cylinders_df.cyl_ID.astype(int)

cylinders_df["label"] = 0

voxelChangeDetector = VoxelChangeDetector()
df1, df2 = voxelChangeDetector.voxelize_trees(
    tree1_filename=tree_filename_2022, 
    tree2_filename=tree_filename_2023,
    voxel_size=voxel_size
)
# source if it source = 1 its the 2022 / source

voxel_diff_df = voxelChangeDetector.compare_voxels()
print(voxel_diff_df)

# df of voxel coordinates only from 2022 of the branches that have fallen
only1_df = voxel_diff_df[voxel_diff_df.Source == '1']
print(only1_df)

# dbscan to cluster to keep only connected voxels for representative samples
connected_df, num_components = dbscan_connectivity(only1_df, voxel_size, min_samples=min_samples)

ref_min = voxelChangeDetector.ref_min_point

print(connected_df)
print(num_components)

# Extract the start and end points of cylinders as numpy arrays from the voxels
start_points = cylinders_df[["startX", "startY", "startZ"]].astype(float).to_numpy()
end_points = cylinders_df[["endX", "endY", "endZ"]].astype(float).to_numpy()

print(start_points)
print(end_points)

total_branch_volume = 0

# if only1_df.shape[0] > 0: Removed the if statement for if there are connected components
start_indices, s_points = points_in_voxels(start_points, connected_df, voxel_size)
end_indices, e_points = points_in_voxels(end_points, connected_df, voxel_size)


# testing the points in the voxels code 

# def points_in_voxels(points, voxel_df, voxel_size):
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
points = start_points
voxel_df = connected_df
voxel_df = voxel_df.copy()

labels = voxel_df[["VoxLabel_X", "VoxLabel_Y", "VoxLabel_Z"]].to_numpy().astype(float)
voxel_min = ref_min_point + labels * voxel_size
voxel_max = voxel_min + voxel_size  # each voxel is a cube of size voxel_size

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

inside_indices = np.array(inside_indices)
print(inside_indices)
inside_points = np.array(inside_points)


print("voxel world min max:")
print(voxel_min.min(axis=0), voxel_max.max(axis=0))

print("points min max:")
print(points.min(axis=0), points.max(axis=0))

print(start_indices)

inside_indices = np.union1d(start_indices, end_indices)
inside_cylinders = cylinders_df.iloc[inside_indices]

cylinders_df.loc[inside_indices, "label"] = 1
filtered_paths = paths_df[paths_df.cyl_ID.isin(inside_cylinders.cyl_ID)]
used_cyl_ids = set()

start_cyls = []

print("branch extraction")

# you dont know if the cyclinders are in the branch or trunk it uses the axis id from archi to extract individual branches from the cyclinders
# if it finds a smaller branch in a larger branch, it will drop the smaller ones
for axis_id in inside_cylinders['axis_ID'].unique():


    branch_df, used_cyl_ids = extract_branch_from_axis(axis_id, 
                                                        inside_cylinders, 
                                                        filtered_paths, 
                                                        used_cyl_ids)
    
    if not branch_df.empty:
        
        # Extraction des coordonnées start et end (N x 3)
        start_points = branch_df[["startX", "startY", "startZ"]].astype(float).to_numpy()
        end_points = branch_df[["endX", "endY", "endZ"]].astype(float).to_numpy()
        
        # Recherche des voisins dans le nuage de points
        start_neighbors = tree_2023.query_ball_point(start_points, r=radius)
        end_neighbors = tree_2023.query_ball_point(end_points, r=radius)
        
        # Comptage des voisins pour chaque point
        start_counts = np.array([len(pts) for pts in start_neighbors])
        end_counts = np.array([len(pts) for pts in end_neighbors])
        
        # Filtrer les branches dont les deux extrémités sont bien connectées au nuage
        start_ok = start_counts >= min_neighbors
        end_ok = end_counts >= min_neighbors
        
        # supprimer les branches proches de données 2023
        valid_branches = branch_df[~(start_ok | end_ok)]
                            
        total_branch_volume = total_branch_volume + np.sum(valid_branches.volume.astype(float))
        branch_length = valid_branches['length'].astype(float).sum()
        
        if branch_length > 5:
                                                
            # --- Coordinates and centers ---
            starts = valid_branches[['startX', 'startY', 'startZ']].values.astype(float)
            ends = valid_branches[['endX', 'endY', 'endZ']].values.astype(float)
            centers = (starts + ends) / 2  # (N, 3)
                            
            # --- First and last cylinder centers based on cyl_ID ---
            min_cyl_idx = valid_branches['cyl_ID'].astype(int).idxmin()
            max_cyl_idx = valid_branches['cyl_ID'].astype(int).idxmax()
            
            start_first = valid_branches.loc[min_cyl_idx, ['startX', 'startY', 'startZ']].astype(float).values
            end_first = valid_branches.loc[min_cyl_idx, ['endX', 'endY', 'endZ']].astype(float).values
            first_center = (start_first + end_first) / 2
            
            start_last = valid_branches.loc[max_cyl_idx, ['startX', 'startY', 'startZ']].astype(float).values
            end_last = valid_branches.loc[max_cyl_idx, ['endX', 'endY', 'endZ']].astype(float).values
            last_center = (start_last + end_last) / 2
            
            # --- Distance between first and last cylinders ---
            dist_first_last = np.linalg.norm(last_center - first_center)
            
            if dist_first_last < 1:
                continue
            
            # --- Volume and surface ---
            volumes = valid_branches['volume'].astype(float).values
            total_volume = volumes.sum()
            max_radius = valid_branches['radius_cyl'].astype(float).max()
            total_surface = (2 * np.pi * valid_branches['radius_cyl'].astype(float) * branch_length).sum()
            
            # --- Center of mass ---
            com = np.sum(centers * volumes[:, np.newaxis], axis=0) / total_volume
            euclidean_distance = np.linalg.norm(com - first_center)
            
            # --- Branch direction vector ---
            coords = valid_branches[['startX', 'startY', 'startZ', 'endX', 'endY', 'endZ']].values.astype(float)
            v = coords[:, 3:6] - coords[:, 0:3]
            norms = np.linalg.norm(v, axis=1)
            valid = norms > 0
            v_unit = np.zeros_like(v)
            v_unit[valid] = v[valid] / norms[valid][:, np.newaxis]
            
            # Average unit vector (branch axis)
            mean_vec = np.nanmean(v_unit[valid], axis=0)
            mean_vec_norm = mean_vec / np.linalg.norm(mean_vec)
            
            # --- Angle with vertical ---
            avg_angle_deg = np.degrees(np.arccos(mean_vec_norm[2]))
            # print(f"Average branch angle: {avg_angle_deg:.2f} degrees")
            
            # --- Projection of CoM and first cylinder onto branch axis ---
            s_com = com @ mean_vec_norm
            s_first = first_center @ mean_vec_norm
            s_last = centers[-1] @ mean_vec_norm
            
            distance_along_axis = abs(s_com - s_first)
            projected_branch_length = abs(s_last - s_first)
            percent_position = 100 * distance_along_axis / projected_branch_length if projected_branch_length > 0 else np.nan
            
            if percent_position > 200:
                
                print("d :")
                print(dist_first_last)
                                                                        
                # # Centroid of start points
                # centroid = np.nanmean(coords[:, 0:3], axis=0)
                
                # # Plot
                # fig = plt.figure(figsize=(10, 8))
                # ax = fig.add_subplot(111, projection='3d')
                
                # ax.scatter(com[0], com[1], com[2], color='r', s=30)
                
                # ax.scatter(start_first[0], start_first[1], start_first[2], color='g', s=30)
                # ax.scatter(centers[-1][0], centers[-1][1], centers[-1][2], color='k', s=30)
            
                # ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], color='b', s=10)
                
                # # Average direction vector (plotted in red from centroid)
                # ax.quiver(
                #     centroid[0], centroid[1], centroid[2],
                #     mean_vec_norm[0], mean_vec_norm[1], mean_vec_norm[2],
                #     length=0.2, normalize=True, color='red', linewidth=2
                # )
                
                # # Labels
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_title(f'Branch Vectors and Mean Orientation\nAverage Angle = {avg_angle_deg:.2f}\n{percent_position}°')
                # plt.tight_layout()
                # plt.show()
                
                # sys.exit()
            
            new_row = pd.DataFrame([{
                "part": part,
                "tree_index": tree_index_str,
                "essence": essence,
                "branch_id": axis_id,
                "volume": total_volume,
                "length": branch_length,
                "surface": total_surface,
                "max_radius": max_radius,
                "angle":avg_angle_deg,
                "origin2com":percent_position,
            }])
            
            fallen_branches_rows_for_part.append({
                "part": part, "tree_index": tree_index_str, "essence": essence,
                "branch_id": axis_id, "volume": total_volume, "length": branch_length,
                "surface": total_surface, "max_radius": max_radius,
                "angle": avg_angle_deg, "origin2com": percent_position
            })
            
            # np.savetxt(f"branches/cylinders/part_{part}_tree_{tree_index_str}_branch_{axis_id}.txt", starts)
            dest_folder_cylinders = os.path.join(root_path, "2022", "branch", "cylinders", f"tree_{tree_index}")
            os.makedirs(dest_folder_cylinders, exist_ok=True)
            cylinders_filename = f"{part}_tree_{tree_index_str}_branch_{axis_id}.txt"
            np.savetxt(os.path.join(dest_folder_cylinders, cylinders_filename), starts)
            # end of branch loop


lost_volume_ratio = total_branch_volume/np.sum(cylinders_df.volume.astype(float))
print(f"lost volume ratio : {lost_volume_ratio}")

print(f"saving {tree_index_str}_true_qsm.csv")
cylinders_df.to_csv(os.path.join(adtreeqsm_folder, f"{tree_index_str}_true_qsm.csv"), index=False)

tree_level_rows_for_part.append({
    "part": part,
    "tree_index_str": tree_index_str,
    "essence": essence,
    "lost_volume_ratio": lost_volume_ratio
})
    # end of tree loop

data_dir = os.getcwd()

# Create DataFrames from the collected rows for the current part
output_fallen_branches_df_part = pd.DataFrame(fallen_branches_rows_for_part, columns=fallen_branches_df_columns)
output_tree_level_df_part = pd.DataFrame(tree_level_rows_for_part, columns=tree_level_df_columns)

print(f"saving fallen_branch_data_{part}.csv")
output_fallen_branches_df_part.to_csv(os.path.join(data_dir, "data", f"fallen_branch_data_{part}.csv"), index=False)

print(f"saving tree_level_data_{part}.csv")
output_tree_level_df_part.to_csv(os.path.join(data_dir, "data", f"new_volume_tree_level_data_{part}.csv"), index=False)
print("Done.")
# end of part loop