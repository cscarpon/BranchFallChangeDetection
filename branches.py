# -*- coding: utf-8 -*-
"""
BranchesChange:
- Optional ICP alignment (2023 -> 2022) using Open3D
- DBH + height metrics written
- Tunable min_branch_length_m
- connect_to_parent: grow upward to larger limb
- capture_descendants: absorb all children below the limb
- split_units_by_branch_id: optional splitting (default False)
- merge_by_parent_branch: merges fragmented units into the largest parent limb (default True)
- Fixes XYZ overwriting by using unique unit_key in filename
- Writes: fallen_branches.csv, intact_branches.csv, tree_level.csv, failures.csv

ADDED (branch position metrics relative to tree):
- branch_base_z_m, branch_base_height_ratio
- branch_com_z_m, branch_com_height_ratio
- branch_base_xy_dist_to_tree_centroid_m, branch_com_xy_dist_to_tree_centroid_m
- branch_base_stem_dist_m, branch_com_stem_dist_m (distance to stem axis in XY)
- outer_canopy_ratio (COM radial distance / crown_r95)
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import open3d as o3d

from VoxelChangeDetector import VoxelChangeDetector
from tools import dbscan_connectivity, points_in_voxels


class BranchesChange:
    def __init__(
        self,
        in_dir_src: str,
        in_dir_target: str,
        qsm_dir: str,
        out_dir: str,
        radius: float = 0.25,
        min_neighbors: int = 1,
        voxel_size: float = 0.5,
        min_samples: int = 10,
        voxel_dilation: int = 0,
        flush_every: int = 1,
        min_branch_length_m: float = 1.0,
        # ICP
        do_icp: bool = True,
        icp_voxel: float = 0.05,
        icp_max_corr: float = 0.25,
        icp_max_iters: int = 60,
        icp_use_point_to_plane: bool = True,
        # Upward/downward logic
        connect_to_parent: bool = True,
        parent_steps_max: int = 200,
        stop_at_present_parent: bool = True,
        capture_descendants: bool = True,
        stop_at_present_descendant: bool = True,
        # NEW
        split_units_by_branch_id: bool = False,
        merge_by_parent_branch: bool = True,
        # Save XYZ for fallen units
        save_fallen_xyz: bool = True,
    ):
        self.in_dir_src = Path(in_dir_src)
        self.in_dir_target = Path(in_dir_target)
        self.qsm_dir = Path(qsm_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.radius = float(radius)
        self.min_neighbors = int(min_neighbors)
        self.voxel_size = float(voxel_size)
        self.min_samples = int(min_samples)
        self.voxel_dilation = int(voxel_dilation)
        self.flush_every = int(flush_every)
        self.min_branch_length_m = float(min_branch_length_m)

        self.do_icp = bool(do_icp)
        self.icp_voxel = float(icp_voxel)
        self.icp_max_corr = float(icp_max_corr)
        self.icp_max_iters = int(icp_max_iters)
        self.icp_use_point_to_plane = bool(icp_use_point_to_plane)

        self.connect_to_parent = bool(connect_to_parent)
        self.parent_steps_max = int(parent_steps_max)
        self.stop_at_present_parent = bool(stop_at_present_parent)

        self.capture_descendants = bool(capture_descendants)
        self.stop_at_present_descendant = bool(stop_at_present_descendant)

        self.split_units_by_branch_id = bool(split_units_by_branch_id)
        self.merge_by_parent_branch = bool(merge_by_parent_branch)
        self.save_fallen_xyz = bool(save_fallen_xyz)

        self.fallen_out = self.out_dir / "fallen_branches.csv"
        self.intact_out = self.out_dir / "intact_branches.csv"
        self.tree_out = self.out_dir / "tree_level.csv"
        self.fail_out = self.out_dir / "failures.csv"

        # ----------------------------
        # ADDED: Branch position metrics columns
        # ----------------------------
        self.pos_cols = [
            "branch_base_z_m",
            "branch_base_height_ratio",
            "branch_com_z_m",
            "branch_com_height_ratio",
            "branch_base_xy_dist_to_tree_centroid_m",
            "branch_com_xy_dist_to_tree_centroid_m",
            "branch_base_stem_dist_m",
            "branch_com_stem_dist_m",
            "outer_canopy_ratio",
        ]

        # Branch outputs include parent/child stats for transparency
        self.branch_cols = [
            "tree_index", "tree_folder",
            "parent_branch_id",
            "n_child_branches", "child_branch_ids",
            "volume", "length", "surface", "max_radius",
            "angle", "origin2com",
            "dir_dx", "dir_dy", "dir_dz",
            "azimuth_deg", "inclination_deg",
            "dbh_m", "height_m",
            "lost_volume_ratio",
            "unit_key",     # stable unique key per saved unit
            "saved_xyz",
        ] + self.pos_cols

        self.tree_cols = [
            "tree_index", "tree_folder",
            "lost_volume_ratio",
            "dbh_m", "dbh_n_cyl",
            "height_m",
            "n_cylinders",
            "n_fallen_units",
            "n_fallen_parent_branches",
            "icp_fitness", "icp_inlier_rmse",
        ]

        self.fail_cols = ["tree_index", "tree_folder", "err"]

        pd.DataFrame(columns=self.branch_cols).to_csv(self.fallen_out, index=False)
        pd.DataFrame(columns=self.branch_cols).to_csv(self.intact_out, index=False)
        pd.DataFrame(columns=self.tree_cols).to_csv(self.tree_out, index=False)
        pd.DataFrame(columns=self.fail_cols).to_csv(self.fail_out, index=False)

        print("[INFO] Writing incrementally to:")
        print(" ", self.fallen_out)
        print(" ", self.intact_out)
        print(" ", self.tree_out)
        print(" ", self.fail_out)

    # ----------------------------
    # IO helpers
    # ----------------------------
    @staticmethod
    def _tree_index_str(filename: str) -> str:
        parts = filename.split(" ")
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format for tree index: {filename}")
        return parts[2]

    @staticmethod
    def _read_xyz_first3(path: str) -> np.ndarray:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python", dtype=str)
        if df.shape[1] < 3:
            raise ValueError(f"XYZ file has < 3 columns: {path}")
        xyz = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return xyz

    @staticmethod
    def _safe_numeric_xyz(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        ok = np.isfinite(arr).all(axis=1)
        return arr[ok]

    @staticmethod
    def _to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        return p

    @staticmethod
    def _write_xyz(path: Path, xyz: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.asarray(xyz, dtype=float), fmt="%.6f")

    def _icp_align_2023_to_2022(self, pts_2022: np.ndarray, pts_2023: np.ndarray):
        if pts_2022.shape[0] < 50 or pts_2023.shape[0] < 50:
            T = np.eye(4, dtype=float)
            return pts_2023, T, np.nan, np.nan

        src = self._to_o3d(pts_2023)
        tgt = self._to_o3d(pts_2022)

        v = float(self.icp_voxel)
        max_corr = float(self.icp_max_corr)

        src_d = src.voxel_down_sample(v) if v > 0 else src
        tgt_d = tgt.voxel_down_sample(v) if v > 0 else tgt

        if self.icp_use_point_to_plane:
            rad = max(2.5 * v, 0.05) if v > 0 else 0.10
            tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
            src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(self.icp_max_iters))
        init = np.eye(4, dtype=float)

        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d,
            max_correspondence_distance=max_corr,
            init=init,
            estimation_method=est,
            criteria=criteria,
        )

        T = np.asarray(reg.transformation, dtype=float)
        pts_2023_aligned = (pts_2023 @ T[:3, :3].T) + T[:3, 3]
        return pts_2023_aligned, T, float(reg.fitness), float(reg.inlier_rmse)

    # ----------------------------
    # rTwig parsing
    # ----------------------------
    def _load_rtwig_cylinders(self, tree_index_str: str) -> pd.DataFrame:
        patt = str(self.qsm_dir / f"arbre 2022 {tree_index_str} *" / "*_branches_cylinders_corrected.csv")
        hits = glob.glob(patt)
        if not hits:
            raise FileNotFoundError(f"Missing rTwig cylinders_corrected for tree {tree_index_str}: {patt}")

        cyl_path = hits[0]
        cyl = pd.read_csv(cyl_path)

        rename = {
            "start_x": "startX", "start_y": "startY", "start_z": "startZ",
            "end_x": "endX", "end_y": "endY", "end_z": "endZ",
            "radius": "radius_cyl",
            "id": "cyl_ID",
            "parent": "parent_ID",
        }
        cyl = cyl.rename(columns={k: v for k, v in rename.items() if k in cyl.columns})

        required = [
            "startX", "startY", "startZ", "endX", "endY", "endZ",
            "cyl_ID", "parent_ID", "radius_cyl", "length", "branch", "branch_order"
        ]
        missing = [c for c in required if c not in cyl.columns]
        if missing:
            raise ValueError(f"rTwig cylinder file missing columns {missing} in {cyl_path}")

        cyl["cyl_ID"] = cyl["cyl_ID"].astype(int)
        cyl["parent_ID"] = pd.to_numeric(cyl["parent_ID"], errors="coerce").fillna(-1).astype(int)
        cyl["radius_cyl"] = pd.to_numeric(cyl["radius_cyl"], errors="coerce")
        cyl["length"] = pd.to_numeric(cyl["length"], errors="coerce")
        cyl["branch"] = pd.to_numeric(cyl["branch"], errors="coerce").astype("Int64")
        cyl["branch_order"] = pd.to_numeric(cyl["branch_order"], errors="coerce").astype("Int64")

        if "volume_m3" in cyl.columns:
            cyl["volume"] = pd.to_numeric(cyl["volume_m3"], errors="coerce")
        elif "volume" in cyl.columns:
            cyl["volume"] = pd.to_numeric(cyl["volume"], errors="coerce")
        else:
            cyl["volume"] = np.pi * (cyl["radius_cyl"].to_numpy(dtype=float) ** 2) * cyl["length"].to_numpy(dtype=float)

        cyl["label"] = 0
        cyl.attrs["rtwig_folder"] = Path(cyl_path).parent.name
        return cyl

    # ----------------------------
    # ADDED: Branch position metrics helpers
    # ----------------------------
    @staticmethod
    def _pca_axis(points_xyz: np.ndarray):
        pts = np.asarray(points_xyz, dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            return np.full(3, np.nan), np.full(3, np.nan)
        mu = pts.mean(axis=0)
        X = pts - mu
        C = (X.T @ X) / max(1, X.shape[0] - 1)
        w, v = np.linalg.eigh(C)
        axis = v[:, np.argmax(w)]
        n = float(np.linalg.norm(axis))
        if n <= 0 or not np.isfinite(n):
            return mu, np.full(3, np.nan)
        return mu, axis / n

    @staticmethod
    def _tree_xy_centroid_from_cyl(cylinders_df: pd.DataFrame) -> np.ndarray:
        s = cylinders_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
        e = cylinders_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
        centers = 0.5 * (s + e)
        xy = centers[:, :2]
        ok = np.isfinite(xy).all(axis=1)
        return xy[ok].mean(axis=0) if np.any(ok) else np.array([np.nan, np.nan], dtype=float)

    @staticmethod
    def _crown_r95_xy(points_2022: np.ndarray, tree_xy_centroid: np.ndarray) -> float:
        pts = np.asarray(points_2022, dtype=float)
        if pts.shape[0] == 0 or not np.isfinite(tree_xy_centroid).all():
            return np.nan
        d = np.linalg.norm(pts[:, :2] - tree_xy_centroid[None, :], axis=1)
        d = d[np.isfinite(d)]
        return float(np.quantile(d, 0.95)) if d.size else np.nan

    @staticmethod
    def _dist_point_to_line_xy(p_xy: np.ndarray, line_p_xy: np.ndarray, line_u_xy: np.ndarray) -> float:
        p = np.asarray(p_xy, dtype=float)
        a = np.asarray(line_p_xy, dtype=float)
        u = np.asarray(line_u_xy, dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(a).all() or not np.isfinite(u).all():
            return np.nan
        nu = float(np.linalg.norm(u))
        if nu <= 0 or not np.isfinite(nu):
            return np.nan
        u = u / nu
        ap = p - a
        return float(abs(ap[0] * u[1] - ap[1] * u[0]))

    def _ordered_centers(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.zeros((0, 3), dtype=float)

        if "base_distance" in df.columns and df["base_distance"].notna().any():
            dd = pd.to_numeric(df["base_distance"], errors="coerce").to_numpy()
            dfo = df.iloc[np.argsort(dd)].copy()
        elif "growth_length" in df.columns and df["growth_length"].notna().any():
            dd = pd.to_numeric(df["growth_length"], errors="coerce").to_numpy()
            dfo = df.iloc[np.argsort(dd)].copy()
        else:
            dfo = df.sort_values("cyl_ID").copy()

        s = dfo[["startX", "startY", "startZ"]].to_numpy(dtype=float)
        e = dfo[["endX", "endY", "endZ"]].to_numpy(dtype=float)
        centers = 0.5 * (s + e)
        centers = centers[np.isfinite(centers).all(axis=1)]
        return centers

    def _branch_position_metrics(
        self,
        unit_df: pd.DataFrame,
        z0: float,
        height_m: float,
        tree_xy_centroid: np.ndarray,
        crown_r95: float,
        stem_p0_xyz: np.ndarray,
        stem_u_xyz: np.ndarray,
    ) -> dict:
        centers = self._ordered_centers(unit_df)
        if centers.shape[0] == 0:
            return {k: np.nan for k in self.pos_cols}

        base = centers[0]

        # COM: volume-weighted over cylinder centers (fallback unweighted)
        vols = pd.to_numeric(unit_df["volume"], errors="coerce").to_numpy(dtype=float)
        vols = np.where(np.isfinite(vols) & (vols > 0), vols, 0.0)
        if vols.size == centers.shape[0] and float(vols.sum()) > 0:
            com = (centers * vols[:, None]).sum(axis=0) / float(vols.sum())
        else:
            com = centers.mean(axis=0)

        base_z = float(base[2])
        com_z = float(com[2])

        if np.isfinite(z0) and np.isfinite(height_m) and height_m > 0:
            base_hr = float((base_z - z0) / height_m)
            com_hr = float((com_z - z0) / height_m)
        else:
            base_hr, com_hr = np.nan, np.nan

        if np.isfinite(tree_xy_centroid).all():
            base_xy_d = float(np.linalg.norm(base[:2] - tree_xy_centroid))
            com_xy_d = float(np.linalg.norm(com[:2] - tree_xy_centroid))
        else:
            base_xy_d, com_xy_d = np.nan, np.nan

        if np.isfinite(stem_p0_xyz).all() and np.isfinite(stem_u_xyz).all():
            base_stem_d = self._dist_point_to_line_xy(base[:2], stem_p0_xyz[:2], stem_u_xyz[:2])
            com_stem_d = self._dist_point_to_line_xy(com[:2], stem_p0_xyz[:2], stem_u_xyz[:2])
        else:
            base_stem_d, com_stem_d = np.nan, np.nan

        if np.isfinite(crown_r95) and crown_r95 > 0 and np.isfinite(com_xy_d):
            outer = float(com_xy_d / crown_r95)
        else:
            outer = np.nan

        return {
            "branch_base_z_m": base_z,
            "branch_base_height_ratio": base_hr,
            "branch_com_z_m": com_z,
            "branch_com_height_ratio": com_hr,
            "branch_base_xy_dist_to_tree_centroid_m": base_xy_d,
            "branch_com_xy_dist_to_tree_centroid_m": com_xy_d,
            "branch_base_stem_dist_m": base_stem_d,
            "branch_com_stem_dist_m": com_stem_d,
            "outer_canopy_ratio": outer,
        }

    # ----------------------------
    # Metrics helpers
    # ----------------------------
    def _branch_direction(self, df: pd.DataFrame):
        if df.empty:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        if "base_distance" in df.columns and df["base_distance"].notna().any():
            dd = pd.to_numeric(df["base_distance"], errors="coerce")
            dfo = df.iloc[np.argsort(dd.to_numpy())].copy()
        elif "growth_length" in df.columns and df["growth_length"].notna().any():
            dd = pd.to_numeric(df["growth_length"], errors="coerce")
            dfo = df.iloc[np.argsort(dd.to_numpy())].copy()
        else:
            dfo = df.sort_values("cyl_ID").copy()

        s = dfo[["startX", "startY", "startZ"]].to_numpy(dtype=float)
        e = dfo[["endX", "endY", "endZ"]].to_numpy(dtype=float)
        centers = 0.5 * (s + e)

        v = centers[-1] - centers[0]
        dx, dy, dz = float(v[0]), float(v[1]), float(v[2])

        az = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        horiz = np.hypot(dx, dy)
        inc = np.degrees(np.arctan2(dz, horiz))
        return (dx, dy, dz, float(az), float(inc))

    @staticmethod
    def _origin2com(df: pd.DataFrame) -> float:
        if df.empty:
            return np.nan

        s2 = df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
        e2 = df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
        centers = 0.5 * (s2 + e2)

        if "base_distance" in df.columns and df["base_distance"].notna().any():
            ordv = pd.to_numeric(df["base_distance"], errors="coerce").to_numpy()
            order = np.argsort(ordv)
        elif "growth_length" in df.columns and df["growth_length"].notna().any():
            ordv = pd.to_numeric(df["growth_length"], errors="coerce").to_numpy()
            order = np.argsort(ordv)
        else:
            order = np.argsort(pd.to_numeric(df["cyl_ID"], errors="coerce").to_numpy())

        centers_o = centers[order]
        first_center = centers_o[0]
        last_center = centers_o[-1]
        axis_vec = last_center - first_center
        axis_len = float(np.linalg.norm(axis_vec))
        if axis_len <= 0:
            return np.nan

        axis_u = axis_vec / axis_len
        vols = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)
        vols = np.where(np.isfinite(vols) & (vols > 0), vols, 0.0)
        vtot = float(np.sum(vols))
        if vtot <= 0:
            return np.nan

        com = (centers * vols[:, None]).sum(axis=0) / vtot
        s_first = float(first_center @ axis_u)
        s_last = float(last_center @ axis_u)
        s_com = float(com @ axis_u)
        denom2 = abs(s_last - s_first)
        return float(100.0 * abs(s_com - s_first) / denom2) if denom2 > 0 else np.nan

    def _surface_area(self, df: pd.DataFrame) -> float:
        if "surface_area_m2" in df.columns:
            return float(pd.to_numeric(df["surface_area_m2"], errors="coerce").sum())
        r = pd.to_numeric(df["radius_cyl"], errors="coerce").to_numpy(dtype=float)
        L = pd.to_numeric(df["length"], errors="coerce").to_numpy(dtype=float)
        return float(np.nansum(2.0 * np.pi * r * L))

    def _angle_from_vertical(self, df: pd.DataFrame) -> float:
        if "angle_from_vertical_deg" not in df.columns:
            return np.nan
        w = pd.to_numeric(df["length"], errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(df["angle_from_vertical_deg"], errors="coerce").to_numpy(dtype=float)
        sw = float(np.nansum(w))
        return float(np.nansum(a * w) / sw) if sw > 0 else np.nan

    def _save_unit_xyz(
        self,
        tree_index_str: str,
        unit_key: str,
        parent_branch_id: int,
        points_2022: np.ndarray,
        tree_2022_kd: cKDTree,
        unit_df: pd.DataFrame,
    ) -> str:
        if not self.save_fallen_xyz:
            return ""

        max_radius = float(pd.to_numeric(unit_df["radius_cyl"], errors="coerce").max())
        search_radius = 1.5 * max_radius

        s2 = unit_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
        e2 = unit_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
        sample = np.vstack([s2, e2])

        neigh = tree_2022_kd.query_ball_point(sample, r=float(search_radius))
        idxs = set()
        for n in neigh:
            idxs.update(n)
        if not idxs:
            return ""

        pts = points_2022[np.array(sorted(idxs), dtype=int), :3]
        out_subdir = self.out_dir / "lost_branches_xyz" / f"tree_{tree_index_str}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Unique per unit_key, no overwriting across fragmented units
        out_path = out_subdir / f"tree_{tree_index_str}_parent_{parent_branch_id}_unit_{unit_key}.xyz"
        np.savetxt(out_path, pts, fmt="%.6f")
        return str(out_path)

    # ----------------------------
    # Core per-tree routine
    # ----------------------------
    def process_one_tree(self, tree_filename_2022: str, tree_filename_2023: str, tree_index_str: str):
        # Load point clouds
        points_2022 = self._safe_numeric_xyz(self._read_xyz_first3(tree_filename_2022))
        points_2023 = self._safe_numeric_xyz(self._read_xyz_first3(tree_filename_2023))
        if points_2022.shape[0] == 0 or points_2023.shape[0] == 0:
            raise ValueError("Empty point cloud after numeric filtering")

        height_m = float(np.max(points_2022[:, 2]) - np.min(points_2022[:, 2]))

        # ICP align 2023 -> 2022
        icp_fitness, icp_rmse = np.nan, np.nan
        points_2023_used = points_2023
        tree_filename_2023_used = tree_filename_2023
        if self.do_icp:
            pts_aligned, _, icp_fitness, icp_rmse = self._icp_align_2023_to_2022(points_2022, points_2023)
            aligned_dir = self.out_dir / "_aligned_xyz" / f"tree_{tree_index_str}"
            aligned_2023_path = aligned_dir / f"tree_{tree_index_str}_2023_aligned.xyz"
            self._write_xyz(aligned_2023_path, pts_aligned)
            points_2023_used = pts_aligned
            tree_filename_2023_used = str(aligned_2023_path)

        tree_2022_kd = cKDTree(points_2022)
        tree_2023_kd = cKDTree(points_2023_used)

        # Load cylinders
        cylinders_df = self._load_rtwig_cylinders(tree_index_str)
        tree_folder = cylinders_df.attrs.get("rtwig_folder", "")

        # Identify trunk branch (rTwig convention: branch_order == 0)
        trunk_branch = cylinders_df.loc[cylinders_df["branch_order"] == 0, "branch"].dropna().unique()
        if len(trunk_branch) != 1:
            raise RuntimeError(f"Expected exactly one trunk branch, got {trunk_branch}")
        trunk_branch = int(trunk_branch[0])

        # DBH (meters) from cylinders at z0+1.3
        z0 = float(np.nanmin(np.r_[cylinders_df["startZ"].to_numpy(dtype=float),
                                  cylinders_df["endZ"].to_numpy(dtype=float)]))
        z_dbh = z0 + 1.3
        zmin = np.minimum(cylinders_df["startZ"].to_numpy(dtype=float), cylinders_df["endZ"].to_numpy(dtype=float))
        zmax = np.maximum(cylinders_df["startZ"].to_numpy(dtype=float), cylinders_df["endZ"].to_numpy(dtype=float))
        hit = cylinders_df[(zmin <= z_dbh) & (zmax >= z_dbh)].copy()
        if hit.shape[0] == 0:
            dbh_m, dbh_n_cyl = np.nan, 0
        else:
            d = 2.0 * hit["radius_cyl"].to_numpy(dtype=float)
            dbh_m = float(d[0]) if d.size == 1 else float(2.0 * np.sqrt(np.sum((d / 2.0) ** 2)))
            dbh_n_cyl = int(hit.shape[0])

        # ----------------------------
        # ADDED: Tree reference frame for branch position metrics
        # ----------------------------
        tree_xy_centroid = self._tree_xy_centroid_from_cyl(cylinders_df)
        crown_r95 = self._crown_r95_xy(points_2022, tree_xy_centroid)

        trunk_df = cylinders_df[cylinders_df["branch"].astype("Int64") == trunk_branch].copy()
        if not trunk_df.empty:
            ts = trunk_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
            te = trunk_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
            trunk_centers = 0.5 * (ts + te)
            stem_p0_xyz, stem_u_xyz = self._pca_axis(trunk_centers)
        else:
            stem_p0_xyz, stem_u_xyz = np.full(3, np.nan), np.full(3, np.nan)

        # Graph maps
        row_map = cylinders_df.set_index("cyl_ID")
        parent_map = cylinders_df.set_index("cyl_ID")["parent_ID"].to_dict()

        children_map = {}
        for cid, pid in zip(cylinders_df["cyl_ID"].astype(int).tolist(), cylinders_df["parent_ID"].astype(int).tolist()):
            if pid < 0:
                continue
            children_map.setdefault(pid, []).append(cid)

        def _is_cyl_present_in_2023(cyl_row: pd.Series) -> bool:
            s = np.array([cyl_row["startX"], cyl_row["startY"], cyl_row["startZ"]], dtype=float)
            e = np.array([cyl_row["endX"],   cyl_row["endY"],   cyl_row["endZ"]], dtype=float)
            sc = len(tree_2023_kd.query_ball_point(s, r=self.radius))
            ec = len(tree_2023_kd.query_ball_point(e, r=self.radius))
            return (sc >= self.min_neighbors) or (ec >= self.min_neighbors)

        def _components_by_parent_graph(df: pd.DataFrame) -> list[pd.DataFrame]:
            if df.empty:
                return []
            ids = set(df["cyl_ID"].astype(int).tolist())
            adj = {i: [] for i in ids}
            for cid, pid in zip(df["cyl_ID"].astype(int).tolist(), df["parent_ID"].astype(int).tolist()):
                if pid in ids:
                    adj[cid].append(pid)
                    adj[pid].append(cid)
            seen, comps = set(), []
            for start in ids:
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                comp_ids = []
                while stack:
                    u = stack.pop()
                    comp_ids.append(u)
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)
                comps.append(df[df["cyl_ID"].astype(int).isin(comp_ids)].copy())
            return comps

        def _grow_upward(unit_df: pd.DataFrame) -> pd.DataFrame:
            if unit_df.empty:
                return unit_df
            unit_ids = set(unit_df["cyl_ID"].astype(int).tolist())
            frontier = list(unit_ids)
            steps = 0
            while frontier and steps < self.parent_steps_max:
                cid = frontier.pop()
                pid = int(parent_map.get(cid, -1))
                if pid < 0 or pid in unit_ids or pid not in row_map.index:
                    steps += 1
                    continue
                prow = row_map.loc[pid]
                try:
                    if int(prow.get("branch", -999)) == trunk_branch:
                        break
                except Exception:
                    pass
                if self.stop_at_present_parent:
                    try:
                        if _is_cyl_present_in_2023(prow):
                            break
                    except Exception:
                        pass
                unit_ids.add(pid)
                frontier.append(pid)
                steps += 1
            return cylinders_df[cylinders_df["cyl_ID"].astype(int).isin(unit_ids)].copy()

        def _grow_downward(unit_df: pd.DataFrame) -> pd.DataFrame:
            if unit_df.empty:
                return unit_df
            unit_ids = set(unit_df["cyl_ID"].astype(int).tolist())
            queue = list(unit_ids)
            while queue:
                cid = int(queue.pop())
                for child in children_map.get(cid, []):
                    if child in unit_ids or child not in row_map.index:
                        continue
                    crow = row_map.loc[int(child)]
                    try:
                        if int(crow.get("branch", -999)) == trunk_branch:
                            continue
                    except Exception:
                        pass
                    if self.stop_at_present_descendant:
                        try:
                            if _is_cyl_present_in_2023(crow):
                                continue
                        except Exception:
                            pass
                    unit_ids.add(int(child))
                    queue.append(int(child))
            return cylinders_df[cylinders_df["cyl_ID"].astype(int).isin(unit_ids)].copy()

        def _parent_branch_id(unit_df: pd.DataFrame) -> int:
            if unit_df.empty:
                return -1
            # parent defined as branch id at most proximal cylinder
            if "base_distance" in unit_df.columns and unit_df["base_distance"].notna().any():
                prox_idx = pd.to_numeric(unit_df["base_distance"], errors="coerce").to_numpy().argmin()
            elif "growth_length" in unit_df.columns and unit_df["growth_length"].notna().any():
                prox_idx = pd.to_numeric(unit_df["growth_length"], errors="coerce").to_numpy().argmin()
            else:
                prox_idx = pd.to_numeric(unit_df["cyl_ID"], errors="coerce").to_numpy().argmin()
            try:
                b = int(pd.to_numeric(unit_df.iloc[prox_idx]["branch"], errors="coerce"))
                return b
            except Exception:
                bids = (
                    unit_df.loc[unit_df["branch"].notna(), "branch"]
                    .astype("Int64").dropna().astype(int).unique().tolist()
                )
                return int(bids[0]) if bids else -1

        def _child_branch_stats(unit_df: pd.DataFrame, parent_bid: int):
            bids = (
                unit_df.loc[unit_df["branch"].notna(), "branch"]
                .astype("Int64").dropna().astype(int).unique().tolist()
            )
            child_ids = [b for b in bids if b != parent_bid]
            return int(len(child_ids)), ",".join(map(str, child_ids))

        # ----------------------------
        # Voxel differencing
        # ----------------------------
        vcd = VoxelChangeDetector()
        vcd.voxelize_trees(
            tree1_filename=tree_filename_2022,
            tree2_filename=tree_filename_2023_used,
            voxel_size=self.voxel_size,
        )
        ref_min = vcd.ref_min_point
        _ = vcd.compare_voxels()

        df1, df2 = vcd.dataframe_1, vcd.dataframe_2
        if df1 is None or df2 is None:
            raise RuntimeError("Voxelization did not produce df1/df2")

        only1_df = VoxelChangeDetector.missing_tree1_voxels(df1, df2, dilation_voxels=self.voxel_dilation)

        fallen_units = []
        total_branch_volume = 0.0

        if only1_df.shape[0] > 0:
            connected_df, _ = dbscan_connectivity(only1_df, self.voxel_size, min_samples=self.min_samples)

            start_pts = cylinders_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
            end_pts = cylinders_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)

            start_idx, _ = points_in_voxels(start_pts, connected_df, self.voxel_size, ref_min)
            end_idx, _ = points_in_voxels(end_pts, connected_df, self.voxel_size, ref_min)

            inside_idx = np.union1d(start_idx, end_idx)
            inside_cyl = cylinders_df.iloc[inside_idx].copy()
            inside_cyl = inside_cyl[inside_cyl["branch"].astype("Int64") != trunk_branch]
            cylinders_df.loc[inside_idx, "label"] = 1

            # Endpoint presence test against 2023
            s_all = inside_cyl[["startX", "startY", "startZ"]].to_numpy(dtype=float)
            e_all = inside_cyl[["endX", "endY", "endZ"]].to_numpy(dtype=float)

            start_neighbors = tree_2023_kd.query_ball_point(s_all, r=self.radius)
            end_neighbors = tree_2023_kd.query_ball_point(e_all, r=self.radius)

            start_counts = np.fromiter((len(v) for v in start_neighbors), dtype=int)
            end_counts = np.fromiter((len(v) for v in end_neighbors), dtype=int)

            start_ok = start_counts >= self.min_neighbors
            end_ok = end_counts >= self.min_neighbors

            valid_missing_cyl = inside_cyl[~(start_ok | end_ok)].copy()

            if not valid_missing_cyl.empty:
                comps = _components_by_parent_graph(valid_missing_cyl)

                # Optional splitting by rTwig branch id
                units = []
                if self.split_units_by_branch_id and "branch" in valid_missing_cyl.columns:
                    for comp in comps:
                        if comp["branch"].notna().any():
                            for bid, sub in comp.groupby(comp["branch"].astype("Int64")):
                                if pd.isna(bid):
                                    continue
                                units.append(sub.copy())
                        else:
                            units.append(comp)
                else:
                    units = comps

                # Expand each unit to largest possible limb
                expanded_units = []
                for u in units:
                    if self.connect_to_parent:
                        u = _grow_upward(u)
                    if self.capture_descendants:
                        u = _grow_downward(u)
                    expanded_units.append(u)

                # Merge fragmented units by parent branch id (largest limb per path)
                if self.merge_by_parent_branch:
                    merged = {}
                    for u in expanded_units:
                        pb = _parent_branch_id(u)
                        if pb < 0:
                            pb = -1
                        ids = set(u["cyl_ID"].astype(int).tolist())
                        merged.setdefault(pb, set()).update(ids)

                    final_units = []
                    for pb, ids in merged.items():
                        dfu = cylinders_df[cylinders_df["cyl_ID"].astype(int).isin(ids)].copy()
                        # ensure trunk excluded
                        dfu = dfu[dfu["branch"].astype("Int64") != trunk_branch]
                        if not dfu.empty:
                            final_units.append(dfu)
                else:
                    final_units = expanded_units

                # Convert to rows
                for u in final_units:
                    if u.empty:
                        continue

                    branch_length = float(pd.to_numeric(u["length"], errors="coerce").sum())
                    if branch_length <= self.min_branch_length_m:
                        continue

                    total_volume = float(pd.to_numeric(u["volume"], errors="coerce").sum())
                    total_branch_volume += total_volume

                    max_radius = float(pd.to_numeric(u["radius_cyl"], errors="coerce").max())
                    surface = self._surface_area(u)
                    angle = self._angle_from_vertical(u)
                    origin2com = self._origin2com(u)
                    dx, dy, dz, azimuth_deg, inclination_deg = self._branch_direction(u)

                    pb = _parent_branch_id(u)
                    n_child, child_ids = _child_branch_stats(u, pb)

                    # stable unit key: parent branch + min cyl id (unique across merges)
                    min_cyl = int(pd.to_numeric(u["cyl_ID"], errors="coerce").min())
                    unit_key = f"pb{pb}_c{min_cyl}"

                    saved_xyz = self._save_unit_xyz(
                        tree_index_str=tree_index_str,
                        unit_key=unit_key,
                        parent_branch_id=pb,
                        points_2022=points_2022,
                        tree_2022_kd=tree_2022_kd,
                        unit_df=u,
                    )

                    # ADDED: position metrics
                    pos = self._branch_position_metrics(
                        unit_df=u,
                        z0=z0,
                        height_m=height_m,
                        tree_xy_centroid=tree_xy_centroid,
                        crown_r95=crown_r95,
                        stem_p0_xyz=stem_p0_xyz,
                        stem_u_xyz=stem_u_xyz,
                    )

                    row = {
                        "tree_index": tree_index_str,
                        "tree_folder": tree_folder,
                        "parent_branch_id": int(pb),
                        "n_child_branches": int(n_child),
                        "child_branch_ids": child_ids,
                        "volume": float(total_volume),
                        "length": float(branch_length),
                        "surface": float(surface),
                        "max_radius": float(max_radius),
                        "angle": float(angle) if np.isfinite(angle) else np.nan,
                        "origin2com": float(origin2com) if np.isfinite(origin2com) else np.nan,
                        "dir_dx": float(dx),
                        "dir_dy": float(dy),
                        "dir_dz": float(dz),
                        "azimuth_deg": float(azimuth_deg),
                        "inclination_deg": float(inclination_deg),
                        "dbh_m": float(dbh_m) if np.isfinite(dbh_m) else np.nan,
                        "height_m": float(height_m),
                        "unit_key": unit_key,
                        "saved_xyz": saved_xyz,
                    }
                    row.update(pos)
                    fallen_units.append(row)

        denom = float(pd.to_numeric(cylinders_df["volume"], errors="coerce").sum())
        lost_volume_ratio = (total_branch_volume / denom) if denom > 0 else np.nan

        # Intact branches: all rTwig branches excluding trunk and fallen parent branches
        fallen_parent_ids = set(int(r["parent_branch_id"]) for r in fallen_units if int(r["parent_branch_id"]) >= 0)

        all_branch_ids = (
            cylinders_df.loc[cylinders_df["branch"].notna(), "branch"]
            .astype("Int64").dropna().astype(int).unique().tolist()
        )
        all_branch_ids = [b for b in all_branch_ids if b != trunk_branch]
        intact_ids = [b for b in all_branch_ids if b not in fallen_parent_ids]

        intact_rows = []
        for bid in intact_ids:
            bdf = cylinders_df[cylinders_df["branch"].astype("Int64") == bid].copy()
            if bdf.empty:
                continue
            blen = float(pd.to_numeric(bdf["length"], errors="coerce").sum())
            if blen <= self.min_branch_length_m:
                continue

            bvol = float(pd.to_numeric(bdf["volume"], errors="coerce").sum())
            bmaxr = float(pd.to_numeric(bdf["radius_cyl"], errors="coerce").max())
            bsurf = self._surface_area(bdf)
            bang = self._angle_from_vertical(bdf)
            bcom = self._origin2com(bdf)
            dx, dy, dz, az, inc = self._branch_direction(bdf)

            # ADDED: position metrics for intact branch (as a unit)
            pos = self._branch_position_metrics(
                unit_df=bdf,
                z0=z0,
                height_m=height_m,
                tree_xy_centroid=tree_xy_centroid,
                crown_r95=crown_r95,
                stem_p0_xyz=stem_p0_xyz,
                stem_u_xyz=stem_u_xyz,
            )

            row = {
                "tree_index": tree_index_str,
                "tree_folder": tree_folder,
                "parent_branch_id": int(bid),
                "n_child_branches": 0,
                "child_branch_ids": "",
                "volume": float(bvol),
                "length": float(blen),
                "surface": float(bsurf),
                "max_radius": float(bmaxr),
                "angle": float(bang) if np.isfinite(bang) else np.nan,
                "origin2com": float(bcom) if np.isfinite(bcom) else np.nan,
                "dir_dx": float(dx),
                "dir_dy": float(dy),
                "dir_dz": float(dz),
                "azimuth_deg": float(az),
                "inclination_deg": float(inc),
                "dbh_m": float(dbh_m) if np.isfinite(dbh_m) else np.nan,
                "height_m": float(height_m),
                "unit_key": f"intact_pb{bid}",
                "saved_xyz": "",
            }
            row.update(pos)
            intact_rows.append(row)

        tree_metrics = {
            "tree_index": tree_index_str,
            "tree_folder": tree_folder,
            "lost_volume_ratio": lost_volume_ratio,
            "dbh_m": dbh_m,
            "dbh_n_cyl": dbh_n_cyl,
            "height_m": height_m,
            "n_cylinders": int(len(cylinders_df)),
            "n_fallen_units": int(len(fallen_units)),
            "n_fallen_parent_branches": int(len(set(int(r["parent_branch_id"]) for r in fallen_units if int(r["parent_branch_id"]) >= 0))),
            "icp_fitness": icp_fitness,
            "icp_inlier_rmse": icp_rmse,
        }

        return lost_volume_ratio, fallen_units, intact_rows, tree_metrics

    # ----------------------------
    # Runner
    # ----------------------------
    def run(
        self,
        split_units_by_branch_id: bool | None = None,
        connect_to_parent: bool | None = None,
        capture_descendants: bool | None = None,
        merge_by_parent_branch: bool | None = None,
    ):
        if split_units_by_branch_id is not None:
            self.split_units_by_branch_id = bool(split_units_by_branch_id)
        if connect_to_parent is not None:
            self.connect_to_parent = bool(connect_to_parent)
        if capture_descendants is not None:
            self.capture_descendants = bool(capture_descendants)
        if merge_by_parent_branch is not None:
            self.merge_by_parent_branch = bool(merge_by_parent_branch)

        src_2022 = self.in_dir_src
        candidate = self.in_dir_src / "2022" / "matched_xyz_species"
        if candidate.exists():
            src_2022 = candidate

        tgt_2023 = self.in_dir_target
        if not src_2022.exists():
            raise FileNotFoundError(f"2022 source folder not found: {src_2022}")
        if not tgt_2023.exists():
            raise FileNotFoundError(f"2023 target folder not found: {tgt_2023}")

        src_files = [p for p in src_2022.iterdir() if p.is_file() and p.suffix.lower() == ".xyz"]
        tgt_files = [p for p in tgt_2023.iterdir() if p.is_file() and p.suffix.lower() == ".xyz"]
        if not src_files or not tgt_files:
            raise FileNotFoundError("Missing .xyz files in source or target folder")

        src_by_idx = {self._tree_index_str(p.name): p for p in src_files}
        tgt_by_idx = {self._tree_index_str(p.name): p for p in tgt_files}

        common = sorted(set(src_by_idx.keys()) & set(tgt_by_idx.keys()), key=lambda x: int(x))
        if not common:
            raise RuntimeError("No matching tree indices between 2022 and 2023 folders")

        # optional resume on tree_out
        done = set()
        try:
            prev = pd.read_csv(self.tree_out, usecols=["tree_index"], dtype={"tree_index": str})
            done = set(prev["tree_index"].dropna().astype(str).tolist())
        except Exception:
            done = set()
        todo = [idx for idx in common if idx not in done]
        if not todo:
            print("[run] Nothing to do")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        fallen_buf, intact_buf, tree_buf, fail_buf = [], [], [], []

        def _flush():
            nonlocal fallen_buf, intact_buf, tree_buf, fail_buf

            if tree_buf:
                df = pd.DataFrame(tree_buf)
                for c in self.tree_cols:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[self.tree_cols]
                df.to_csv(self.tree_out, mode="a", header=False, index=False)
                tree_buf = []

            def _write_branch(buf, out_path):
                if not buf:
                    return
                dfb = pd.DataFrame(buf)
                for c in self.branch_cols:
                    if c not in dfb.columns:
                        dfb[c] = np.nan
                dfb = dfb[self.branch_cols]
                dfb.to_csv(out_path, mode="a", header=False, index=False)

            _write_branch(fallen_buf, self.fallen_out)
            fallen_buf = []
            _write_branch(intact_buf, self.intact_out)
            intact_buf = []

            if fail_buf:
                dff = pd.DataFrame(fail_buf)
                for c in self.fail_cols:
                    if c not in dff.columns:
                        dff[c] = np.nan
                dff = dff[self.fail_cols]
                dff.to_csv(self.fail_out, mode="a", header=False, index=False)
                fail_buf = []

        print(
            f"[run] split_units_by_branch_id={self.split_units_by_branch_id} "
            f"connect_to_parent={self.connect_to_parent} capture_descendants={self.capture_descendants} "
            f"merge_by_parent_branch={self.merge_by_parent_branch} min_branch_length_m={self.min_branch_length_m}"
        )

        for i, idx in enumerate(todo):
            f2022 = src_by_idx[idx]
            f2023 = tgt_by_idx[idx]
            print(f"\n[Tree {i+1}/{len(todo)}] idx={idx}")
            print("  2022:", f2022)
            print("  2023:", f2023)

            try:
                lost_ratio, fallen_rows, intact_rows, tree_metrics = self.process_one_tree(
                    tree_filename_2022=str(f2022),
                    tree_filename_2023=str(f2023),
                    tree_index_str=idx,
                )

                # attach lost_ratio
                for r in fallen_rows:
                    r["lost_volume_ratio"] = lost_ratio
                for r in intact_rows:
                    r["lost_volume_ratio"] = lost_ratio

                fallen_buf.extend(fallen_rows)
                intact_buf.extend(intact_rows)
                tree_buf.append({k: tree_metrics.get(k, np.nan) for k in self.tree_cols})

                print(f"  lost_volume_ratio: {lost_ratio:.6f}")
                print(f"  fallen units saved: {tree_metrics['n_fallen_units']}  unique parent branches: {tree_metrics['n_fallen_parent_branches']}")
                print(f"  intact branches saved: {len(intact_rows)}")
                print(f"  icp_fitness: {tree_metrics.get('icp_fitness')}  icp_rmse: {tree_metrics.get('icp_inlier_rmse')}")

            except Exception as e:
                fail_buf.append({"tree_index": idx, "tree_folder": "", "err": str(e)})
                print(f"[SKIP] idx={idx}: {e}")

            fe = int(getattr(self, "flush_every", 1))
            if fe <= 1 or ((i + 1) % fe == 0):
                _flush()

        _flush()
        return pd.read_csv(self.fallen_out), pd.read_csv(self.intact_out), pd.read_csv(self.tree_out)


# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     bc = BranchesChange(
#         in_dir_src=r"D:\Chris\Hydro\Karl\translation\raw\part1", ### Oldest (source) tree point clouds
#         in_dir_target=r"D:\Karl\hydro\dataset\Working\single_trees\part1\2023\matched_xyz_species", ### Newest (target) tree point clouds
#         qsm_dir=r"D:\Chris\Hydro\Karl\translation\rTwig\part1", ### rTwig cylinders (from 2022 point clouds)
#         out_dir=r"D:\Chris\Hydro\Karl\translation\change", ### Output folder for CSVs and saved XYZ of fallen branches
#         radius=0.5,
#         min_neighbors=1,
#         voxel_size=0.5,
#         min_samples=5,
#         voxel_dilation=1,
#         flush_every=1,
#         min_branch_length_m=2.0,
#         do_icp=True,
#         icp_voxel=0.05,
#         icp_max_corr=0.25,
#         icp_max_iters=60,
#         icp_use_point_to_plane=True,
#         connect_to_parent=True,
#         capture_descendants=True,
#         stop_at_present_descendant=True,
#         split_units_by_branch_id=False,   # biggest branches possible
#         merge_by_parent_branch=True,      # merge fragmented canopy-edge detections
#         save_fallen_xyz=True,
#     )

#     fallen_df, intact_df, tree_df = bc.run(
#         split_units_by_branch_id=False,
#         connect_to_parent=True,
#         capture_descendants=True,
#         merge_by_parent_branch=True,
#     )
