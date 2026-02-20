from branches import BranchesChange

if __name__ == "__main__":
    bc = BranchesChange(
        in_dir_src=r"D:\Chris\Hydro\Karl\translation\raw\part1", ### Oldest (source) tree point clouds
        in_dir_target=r"D:\Karl\hydro\dataset\Working\single_trees\part1\2023\matched_xyz_species", ### Newest (target) tree point clouds
        qsm_dir=r"D:\Chris\Hydro\Karl\translation\rTwig\part1", ### rTwig cylinders (from 2022 point clouds)
        out_dir=r"D:\Chris\Hydro\Karl\translation\change", ### Output folder for CSVs and saved XYZ of fallen branches
        radius=0.5,
        min_neighbors=1,
        voxel_size=0.5,
        min_samples=5,
        voxel_dilation=1,
        flush_every=1,
        min_branch_length_m=2.0,
        do_icp=True,
        icp_voxel=0.05,
        icp_max_corr=0.25,
        icp_max_iters=60,
        icp_use_point_to_plane=True,
        connect_to_parent=True,
        capture_descendants=True,
        stop_at_present_descendant=True,
        split_units_by_branch_id=False,   # biggest branches possible
        merge_by_parent_branch=True,      # merge fragmented canopy-edge detections
        save_fallen_xyz=True,
    )

    fallen_df, intact_df, tree_df = bc.run(
        split_units_by_branch_id=False,
        connect_to_parent=True,
        capture_descendants=True,
        merge_by_parent_branch=True,
    )