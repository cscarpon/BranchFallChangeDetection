# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:54:29 2025

@author: KAMON37
"""

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
parts = ["part1", "part2b", "part3", "part4", "part5","part6"]
# parts = ["part2a"]

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
min_samples = 10

# Root Drive for the data
current_dir = rf"D:\Karl\hydro\dataset\Working\single_trees"

# Start of the root outdirs, usually the working drive. 
out_dir = rf"D:\Chris\Hydro\Karl\data"

for part in parts:
    dest_part = os.path.join(
        out_dir,
        f"{part}",
        "2022"
    )
    os.makedirs(dest_part, exist_ok=True)
    
    fallen_branches_rows_for_part = []
    tree_level_rows_for_part = []

    # current_dir = os.getcwd()
    root_path = rf"{current_dir}\{part}"
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

    archi_sorted_qsm = sorted(
        [f for f in os.listdir(adtreeqsm_folder) if 'qsm_out' in f and f.endswith('.csv')],
        key=lambda x: int(x.split('_')[0])
    )

    # sorted_treedata = sorted(
    #     [f for f in os.listdir(adtreeqsm_folder) if 'tree' in f and f.endswith('.csv')],
    #     key=lambda x: int(x.split('_')[0])
    # )

    sorted_paths = sorted(
        [f for f in os.listdir(adtreeqsm_folder) if 'paths' in f and f.endswith('.csv')],
        key=lambda x: int(x.split('_')[0])
    )
    
    # tree_level_data_path = os.path.join(root_path, f"tree_level_data_{part}.csv")
    # tree_level_data = pd.read_csv(tree_level_data_path)

    tree_indexes = tuple(range(len(sorted_trees_2022)-1))
    # tree_indexes = (0,)
    
    for tree_index in tree_indexes:
        
        all_cluster_pts = []
        
        print(f"\ntree index : {tree_index}")
        
        tree_filename_2022 = os.path.join(trees_2022, sorted_trees_2022[tree_index])
        tree_filename_2023 = os.path.join(trees_2023, sorted_trees_2023[tree_index])               
        print(f'"{tree_filename_2022}"')
        print(f'"{tree_filename_2023}"')


        # Load 2022 tree and build KDTree
        PC_2022 = np.loadtxt(tree_filename_2022)
        points_2022 = PC_2022[:, :3]  # assuming XYZ in first 3 columns
        tree_2022 = cKDTree(points_2022)
        
        # Load 2023 tree and build KDTree
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

        #Define the minimum reference point:

        ref_min = voxelChangeDetector.ref_min_point

        # source if it source = 1 its the 2022 / source

        voxel_diff_df = voxelChangeDetector.compare_voxels()

        # df of voxel coordinates only from 2022 of the branches that have fallen
        only1_df = voxel_diff_df[voxel_diff_df.Source == '1']
        
        # dbscan to cluster to keep only connected voxels for representative samples
        connected_df, num_components = dbscan_connectivity(only1_df, voxel_size, min_samples=min_samples)

        # Extract the start and end points of cylinders as numpy arrays from the voxels
        start_points = cylinders_df[["startX", "startY", "startZ"]].astype(float).to_numpy()
        end_points = cylinders_df[["endX", "endY", "endZ"]].astype(float).to_numpy()
        
        total_branch_volume = 0
        
        if only1_df.shape[0] > 0:
            start_indices, s_points = points_in_voxels(start_points, connected_df, voxel_size, ref_min)
            end_indices, e_points = points_in_voxels(end_points, connected_df, voxel_size, ref_min)
            
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

                        # Save the branch code                       
                        search_radius = float(valid_branches['radius_cyl'].astype(float).max()) * 1.5

                        # Use all start and end points of the fallen cylinders as query centers
                        sample_points = np.vstack([starts, ends])  # (2N, 3)

                        branch_point_indices = set()
                        neighbor_lists = tree_2022.query_ball_point(sample_points, r=search_radius)

                        for neigh in neighbor_lists:
                            for idx in neigh:
                                branch_point_indices.add(idx)

                        if branch_point_indices:
                            branch_point_indices = np.array(sorted(branch_point_indices), dtype=int)
                            branch_points = points_2022[branch_point_indices, :3]

                        # Change the dest folder to point to the out_dir
                            dest_folder_points = os.path.join(
                                dest_part,
                                "branch_points",
                                f"tree_{tree_index}"
                            )
                            os.makedirs(dest_folder_points, exist_ok=True)

                            xyz_filename = f"{part}_tree_{tree_index_str}_branch_{axis_id}.xyz"
                            np.savetxt(
                                os.path.join(dest_folder_points, xyz_filename),
                                branch_points,
                                fmt="%.6f"
                            )

                            # Optional: also write LAZ using laspy (if you want)
                            import laspy
                            laz_filename = f"{part}_tree_{tree_index_str}_branch_{axis_id}.laz"
                            hdr = laspy.LasHeader(point_format=3, version="1.2")
                            las = laspy.LasData(hdr)
                            las.x = branch_points[:, 0]
                            las.y = branch_points[:, 1]
                            las.z = branch_points[:, 2]
                            las.write(os.path.join(dest_folder_points, laz_filename))
                        
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
                        dest_folder_cylinders = os.path.join(dest_part, "branch", "cylinders", f"tree_{tree_index}")
                        os.makedirs(dest_folder_cylinders, exist_ok=True)
                        cylinders_filename = f"{part}_tree_{tree_index_str}_branch_{axis_id}.txt"
                        np.savetxt(os.path.join(dest_folder_cylinders, cylinders_filename), starts)
                    # end of branch loop
        
        
        lost_volume_ratio = total_branch_volume/np.sum(cylinders_df.volume.astype(float))
        print(f"lost volume ratio : {lost_volume_ratio}")
    
        print(f"saving {tree_index_str}_true_qsm.csv")

        dest_folder_cylinders = os.path.join(
            dest_part,
            "cylinders",
            f"tree_{tree_index}"
        )
        os.makedirs(dest_folder_cylinders, exist_ok=True)

        cylinders_df.to_csv(os.path.join(dest_folder_cylinders, f"{tree_index_str}_true_qsm.csv"), index=False)
        
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
    output_fallen_branches_df_part.to_csv(os.path.join(dest_part, f"fallen_branch_data_{part}.csv"), index=False)

    print(f"saving tree_level_data_{part}.csv")
    output_tree_level_df_part.to_csv(os.path.join(dest_part, f"new_volume_tree_level_data_{part}.csv"), index=False)
    print("Done.")
# end of part loop