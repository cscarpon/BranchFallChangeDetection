import numpy as np
import open3d as o3d

### Alpha shape hull and voxel occupancy grid example for single-tree point clouds. ### 


# -------------------------
# Inputs: a single-tree point cloud
# -------------------------
# You should replace this with your own loader / extractor.
# Example:

path = r"D:/Chris/Hydro/Karl/edits/part6/arbre 2022 245 12.31 Unknown.xyz"
# Read XYZ (Open3D will treat it as an uncolored point cloud)
pcd = o3d.io.read_point_cloud(path)
pts = np.asarray(pcd.points)

print("loaded points:", pts.shape)
if pts.size == 0:
    raise RuntimeError("Point cloud loaded with 0 points. File may not be plain XYZ or parsing failed.")



# Clean NaN/inf (common in exported XYZ)
mask = np.isfinite(pts).all(axis=1)
pcd = pcd.select_by_index(np.flatnonzero(mask).astype(np.int32))
pts = np.asarray(pcd.points)

pcd_ds = pcd.voxel_down_sample(voxel_size=0.01)  # adjust if needed
print("downsampled points:", np.asarray(pcd_ds.points).shape)

# ---- Tight hull (alpha shape) ----
alpha = 0.08
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ds, alpha)
mesh.compute_vertex_normals()


mesh = (mesh.remove_duplicated_vertices()
            .remove_degenerate_triangles()
            .remove_duplicated_triangles()
            .remove_non_manifold_edges())
mesh.compute_vertex_normals()
print(mesh)

pcd_show = pcd_ds
pcd_show.paint_uniform_color([0.7,0.7,0.7])

for a in [0.18, 0.5, 0.7, 0.8]:
    m = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ds, a)
    m = (m.remove_duplicated_vertices()
           .remove_degenerate_triangles()
           .remove_duplicated_triangles()
           .remove_non_manifold_edges())
    m.compute_vertex_normals()
    m.paint_uniform_color([0.2,0.8,0.2])
    o3d.visualization.draw_geometries([pcd_show, m], window_name=f"alpha={a}  watertight={m.is_watertight()}")


##### Voxel Occupancy Grid + Matching Cubes #####


def voxel_indices(points: np.ndarray, origin: np.ndarray, vs: float) -> np.ndarray:
    # points: (N,3)
    return np.floor((points - origin) / vs).astype(np.int32)


def dilate_voxels(occ: np.ndarray, r: int) -> np.ndarray:
    # occ: (M,3) int voxel indices, r: dilation radius in voxels
    if r <= 0:
        return occ
    offsets = np.array(
        [(dx, dy, dz)
         for dx in range(-r, r + 1)
         for dy in range(-r, r + 1)
         for dz in range(-r, r + 1)],
        dtype=np.int32
    )
    expanded = (occ[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return expanded


def filter_target_by_source_voxels(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.03,
    dilation_voxels: int = 1,
):
    src = np.asarray(source_pcd.points, dtype=np.float64)
    tgt = np.asarray(target_pcd.points, dtype=np.float64)

    if src.shape[0] == 0 or tgt.shape[0] == 0:
        raise ValueError("source_pcd or target_pcd has 0 points")

    # Set origin to source min bound so indices are stable
    origin = src.min(axis=0)

    # Source occupancy
    src_idx = voxel_indices(src, origin, voxel_size)

    # Unique occupied voxels
    # Use a structured view to allow np.unique over rows
    src_view = src_idx.view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
    occ = np.unique(src_view).view(np.int32).reshape(-1, 3)

    # Optional dilation in voxel space
    occ_dil = dilate_voxels(occ, dilation_voxels)
    occ_dil_view = occ_dil.view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
    occ_dil = np.unique(occ_dil_view).view(np.int32).reshape(-1, 3)

    # Build a hash set for membership testing
    # Pack 3 int32 into one int64-ish key using offsets to avoid negatives
    mins = occ_dil.min(axis=0)
    shifted = occ_dil - mins
    # Choose multipliers large enough (max range + 1)
    ranges = shifted.max(axis=0) + 1
    key_occ = (shifted[:, 0].astype(np.int64) +
               shifted[:, 1].astype(np.int64) * ranges[0].astype(np.int64) +
               shifted[:, 2].astype(np.int64) * (ranges[0].astype(np.int64) * ranges[1].astype(np.int64)))
    occ_set = set(key_occ.tolist())

    # Target indices and membership test
    tgt_idx = voxel_indices(tgt, origin, voxel_size)
    tgt_shift = tgt_idx - mins
    key_tgt = (tgt_shift[:, 0].astype(np.int64) +
               tgt_shift[:, 1].astype(np.int64) * ranges[0].astype(np.int64) +
               tgt_shift[:, 2].astype(np.int64) * (ranges[0].astype(np.int64) * ranges[1].astype(np.int64)))

    keep = np.fromiter((k in occ_set for k in key_tgt), dtype=bool, count=key_tgt.shape[0])

    kept_idx = np.flatnonzero(keep).astype(np.int32)
    target_in_occ = target_pcd.select_by_index(kept_idx)

    # For visualization: voxel centers as points
    centers = (occ_dil.astype(np.float64) + 0.5) * voxel_size + origin
    occ_centers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))

    return target_in_occ, occ_centers


# -------------------------
# Example usage
# -------------------------
# Replace with your own file paths
src_path = r"D:/Chris/Hydro/Karl/edits/part6/arbre 2022 245 12.31 Unknown.xyz"
tgt_path = r"D:/Karl/hydro/dataset/Working/single_trees/part6/2023/matched_xyz_species/arbre 2023 245 12.31 Unknown.xyz" # example

source = o3d.io.read_point_cloud(src_path)
target = o3d.io.read_point_cloud(tgt_path)

# (Optional) Downsample for speed and more uniform occupancy
source_ds = source.voxel_down_sample(0.01)
target_ds = target.voxel_down_sample(0.01)

target_in_occ, occ_centers = filter_target_by_source_voxels(
    source_ds, target_ds,
    voxel_size=0.03,        # 3 cm occupancy
    dilation_voxels=3      # expand by 1 voxel (3 cm) to tolerate misalignment
)

# Colorize for display
source_ds.paint_uniform_color([0.7, 0.7, 0.7])       # grey
target_ds.paint_uniform_color([0.5, 0.5, 0.9])       # blue
target_in_occ.paint_uniform_color([1.0, 0.2, 0.2])   # red
occ_centers.paint_uniform_color([0.2, 0.9, 0.2])     # green

o3d.visualization.draw_geometries(
    [source_ds, target_ds, target_in_occ, occ_centers],
    window_name="Voxel occupancy gating (source occupancy -> filtered target)"
)