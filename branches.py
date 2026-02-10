# -*- coding: utf-8 -*-
"""
Created on 2026-02-09

@author: cscarpon
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
    ):
        self.in_dir_src = Path(in_dir_src)
        self.in_dir_target = Path(in_dir_target)
        self.qsm_dir = Path(qsm_dir)
        self.out_dir = Path(out_dir)

        self.radius = float(radius)
        self.min_neighbors = int(min_neighbors)
        self.voxel_size = float(voxel_size)
        self.min_samples = int(min_samples)
        self.flush_every = int(flush_every)
        self.voxel_dilation = int(voxel_dilation)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # output paths
        self.fallen_out = self.out_dir / "fallen_branches.csv"
        self.tree_out = self.out_dir / "tree_level.csv"
        self.fail_out = self.out_dir / "failures.csv"

        # Fixed column sets so incremental writes are stable
        self.fallen_cols = [
            "tree_index", "tree_folder", "branch_id",
            "volume", "length", "surface", "max_radius",
            "angle", "origin2com",
            "dir_dx", "dir_dy", "dir_dz",
            "azimuth_deg", "inclination_deg",
            "dbh_m", "height_m",
            "lost_volume_ratio",
            "saved_xyz",
        ]

        self.tree_cols = [
            "tree_index", "tree_folder",
            "lost_volume_ratio",
            "dbh_m", "dbh_n_cyl",
            "height_m",
            "n_cylinders", "n_fallen_branches",
        ]

        self.fail_cols = ["tree_index", "tree_folder", "err"]

        # Initialize files with headers (overwrite existing)
        pd.DataFrame(columns=self.fallen_cols).to_csv(self.fallen_out, index=False)
        pd.DataFrame(columns=self.tree_cols).to_csv(self.tree_out, index=False)
        pd.DataFrame(columns=self.fail_cols).to_csv(self.fail_out, index=False)

        print("[INFO] Writing incrementally to:")
        print(" ", self.fallen_out)
        print(" ", self.tree_out)
        print(" ", self.fail_out)

    # ----------------------------
    # Input parsing
    # ----------------------------
    @staticmethod
    def _tree_index_str(filename: str) -> str:
        parts = filename.split(" ")
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format for tree index: {filename}")
        return parts[2]

    @staticmethod
    def _read_xyz_first3(path: str) -> np.ndarray:
        # robust reader: handles junk/header/mixed whitespace
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python", dtype=str)
        if df.shape[1] < 3:
            raise ValueError(f"XYZ file has < 3 columns: {path}")
        xyz = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return xyz

    def _load_rtwig_cylinders(self, tree_index_str: str) -> pd.DataFrame:
        patt = str(self.qsm_dir / f"arbre 2022 {tree_index_str} *" / "*_branches_cylinders_corrected.csv")
        hits = glob.glob(patt)
        if not hits:
            raise FileNotFoundError(f"Missing rTwig cylinders_corrected for tree {tree_index_str}: {patt}")

        cyl_path = hits[0]
        cyl = pd.read_csv(cyl_path)

        # normalize columns
        rename = {
            "start_x": "startX", "start_y": "startY", "start_z": "startZ",
            "end_x": "endX", "end_y": "endY", "end_z": "endZ",
            "radius": "radius_cyl",
            "id": "cyl_ID",
            "parent": "parent_ID",
        }
        cyl = cyl.rename(columns={k: v for k, v in rename.items() if k in cyl.columns})

        required = ["startX", "startY", "startZ", "endX", "endY", "endZ", "cyl_ID", "parent_ID", "radius_cyl", "length", "branch", "branch_order"]
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
    # Core per-tree routine
    # ----------------------------
    def process_one_tree(self, tree_filename_2022: str, tree_filename_2023: str, tree_index_str: str):
        """
        Per-tree workflow:
        - read XYZ (robust) and build KD trees
        - load rTwig cylinders_corrected for this tree_index
        - compute trunk (branch_order==0), DBH, height
        - voxel differencing (Source==1)
        - map diff voxels -> cylinders (start/end in voxels)
        - identify missing cylinders by KD connectivity to 2023
        - DECOMPOSE missing cylinders by parent-child graph components
        - split components by rTwig branch id so sub-branches are not suppressed
        - compute metrics, direction, optional save branch points
        Returns:
        lost_volume_ratio, cylinders_df, fallen_branch_rows, tree_metrics_dict
        """

        # ----------------------------
        # helpers (local, no class changes required)
        # ----------------------------
        def _components_by_parent_graph(df: pd.DataFrame) -> list[pd.DataFrame]:
            # df must contain cyl_ID, parent_ID
            if df.empty:
                return []

            ids = set(df["cyl_ID"].astype(int).tolist())
            adj = {i: [] for i in ids}

            parent_vals = pd.to_numeric(df["parent_ID"], errors="coerce").fillna(-1).astype(int).values
            cyl_vals = df["cyl_ID"].astype(int).values

            for cid, pid in zip(cyl_vals, parent_vals):
                if pid in ids:
                    adj[cid].append(pid)
                    adj[pid].append(cid)

            seen = set()
            comps = []
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

        def _safe_numeric_xyz(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float64)
            ok = np.isfinite(arr).all(axis=1)
            return arr[ok]

        # ----------------------------
        # load point clouds + KD trees
        # ----------------------------
        points_2022 = self._read_xyz_first3(tree_filename_2022)
        points_2022 = _safe_numeric_xyz(points_2022)
        if points_2022.shape[0] == 0:
            raise ValueError(f"No finite XYZ points in 2022 file: {tree_filename_2022}")
        tree_2022 = cKDTree(points_2022)

        points_2023 = self._read_xyz_first3(tree_filename_2023)
        points_2023 = _safe_numeric_xyz(points_2023)
        if points_2023.shape[0] == 0:
            raise ValueError(f"No finite XYZ points in 2023 file: {tree_filename_2023}")
        tree_2023 = cKDTree(points_2023)

        # Tree height from points
        zmin_2022 = float(np.min(points_2022[:, 2]))
        zmax_2022 = float(np.max(points_2022[:, 2]))
        height_m = zmax_2022 - zmin_2022

        # ----------------------------
        # load rTwig cylinders
        # ----------------------------
        cylinders_df = self._load_rtwig_cylinders(tree_index_str)
        tree_folder = cylinders_df.attrs.get("rtwig_folder", "")

        # trunk identification (rTwig convention: branch_order==0)
        trunk_branch = cylinders_df.loc[cylinders_df["branch_order"] == 0, "branch"].dropna().unique()
        if len(trunk_branch) != 1:
            raise RuntimeError(f"Expected exactly one trunk branch, got {trunk_branch}")
        trunk_branch = int(trunk_branch[0])

        # ----------------------------
        # DBH from cylinders (meters)
        # ----------------------------
        z0 = float(np.nanmin(np.r_[cylinders_df["startZ"].to_numpy(dtype=float),
                                cylinders_df["endZ"].to_numpy(dtype=float)]))
        z_dbh = z0 + 1.3

        zmin = np.minimum(cylinders_df["startZ"].to_numpy(dtype=float), cylinders_df["endZ"].to_numpy(dtype=float))
        zmax = np.maximum(cylinders_df["startZ"].to_numpy(dtype=float), cylinders_df["endZ"].to_numpy(dtype=float))
        hit = cylinders_df[(zmin <= z_dbh) & (zmax >= z_dbh)].copy()

        if hit.shape[0] == 0:
            dbh_m = np.nan
            dbh_n_cyl = 0
        else:
            d = 2.0 * hit["radius_cyl"].to_numpy(dtype=float)
            dbh_m = float(d[0]) if d.size == 1 else float(2.0 * np.sqrt(np.sum((d / 2.0) ** 2)))
            dbh_n_cyl = int(hit.shape[0])

        # ----------------------------
        # voxel change detection (your existing approach)
        # ----------------------------
        vcd = VoxelChangeDetector()
        vcd.voxelize_trees(
            tree1_filename=tree_filename_2022,
            tree2_filename=tree_filename_2023,
            voxel_size=self.voxel_size
        )
        ref_min = vcd.ref_min_point
        _ = vcd.compare_voxels()

        df1 = vcd.dataframe_1
        df2 = vcd.dataframe_2
        if df1 is None or df2 is None:
            raise RuntimeError("Voxelization did not produce df1/df2")

        # tolerance-aware missing voxels (dilated)
        only1_df = VoxelChangeDetector.missing_tree1_voxels(df1, df2, dilation_voxels=self.voxel_dilation)

        total_branch_volume = 0.0
        fallen_branch_rows = []

        if only1_df.shape[0] > 0:
            connected_df, _ = dbscan_connectivity(only1_df, self.voxel_size, min_samples=self.min_samples)

            start_pts = cylinders_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
            end_pts = cylinders_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)

            start_idx, _ = points_in_voxels(start_pts, connected_df, self.voxel_size, ref_min)
            end_idx, _ = points_in_voxels(end_pts, connected_df, self.voxel_size, ref_min)

            inside_idx = np.union1d(start_idx, end_idx)
            inside_cyl = cylinders_df.iloc[inside_idx].copy()

            # Exclude trunk from analysis
            inside_cyl = inside_cyl[inside_cyl["branch"].astype("Int64") != trunk_branch]

            cylinders_df.loc[inside_idx, "label"] = 1

            # ------------------------------------------------------------
            # CHANGE: determine "missing cylinders" first, then decompose
            # ------------------------------------------------------------

            # KD connectivity test against 2023 at cylinder level
            s_all = inside_cyl[["startX", "startY", "startZ"]].to_numpy(dtype=float)
            e_all = inside_cyl[["endX", "endY", "endZ"]].to_numpy(dtype=float)

            start_neighbors = tree_2023.query_ball_point(s_all, r=self.radius)
            end_neighbors = tree_2023.query_ball_point(e_all, r=self.radius)

            start_counts = np.fromiter((len(v) for v in start_neighbors), dtype=int)
            end_counts = np.fromiter((len(v) for v in end_neighbors), dtype=int)

            start_ok = start_counts >= self.min_neighbors
            end_ok = end_counts >= self.min_neighbors

            # missing cylinders = neither end connects to 2023
            valid_missing_cyl = inside_cyl[~(start_ok | end_ok)].copy()
            if not valid_missing_cyl.empty:

                # components in parent-child graph of missing cylinders
                comps = _components_by_parent_graph(valid_missing_cyl)

                # split each component by rTwig branch id
                units = []
                if "branch" in valid_missing_cyl.columns:
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

                # process each unit independently (preserves sub-branches)
                for unit_df in units:
                    if unit_df.empty:
                        continue

                    # branch id label
                    if "branch" in unit_df.columns and unit_df["branch"].notna().any():
                        branch_id = int(unit_df["branch"].dropna().iloc[0])
                    else:
                        branch_id = -1

                    # metrics
                    branch_length = float(pd.to_numeric(unit_df["length"], errors="coerce").sum())
                    if branch_length <= 5.0:
                        continue

                    total_volume = float(pd.to_numeric(unit_df["volume"], errors="coerce").sum())
                    total_branch_volume += total_volume

                    max_radius = float(pd.to_numeric(unit_df["radius_cyl"], errors="coerce").max())

                    if "surface_area_m2" in unit_df.columns:
                        total_surface = float(pd.to_numeric(unit_df["surface_area_m2"], errors="coerce").sum())
                    else:
                        r = pd.to_numeric(unit_df["radius_cyl"], errors="coerce").to_numpy(dtype=float)
                        L = pd.to_numeric(unit_df["length"], errors="coerce").to_numpy(dtype=float)
                        total_surface = float(np.nansum(2.0 * np.pi * r * L))

                    # mean angle from vertical (length-weighted)
                    if "angle_from_vertical_deg" in unit_df.columns:
                        w = pd.to_numeric(unit_df["length"], errors="coerce").to_numpy(dtype=float)
                        a = pd.to_numeric(unit_df["angle_from_vertical_deg"], errors="coerce").to_numpy(dtype=float)
                        avg_angle_deg = float(np.nansum(a * w) / np.nansum(w)) if np.nansum(w) > 0 else np.nan
                    else:
                        avg_angle_deg = np.nan

                    # direction (proximal->distal) from your existing helper
                    dx, dy, dz, azimuth_deg, inclination_deg = self._branch_direction(unit_df)

                    # origin2com (your existing logic, adapted to unit_df)
                    s2 = unit_df[["startX", "startY", "startZ"]].to_numpy(dtype=float)
                    e2 = unit_df[["endX", "endY", "endZ"]].to_numpy(dtype=float)
                    centers = 0.5 * (s2 + e2)

                    if "base_distance" in unit_df.columns and unit_df["base_distance"].notna().any():
                        ordv = pd.to_numeric(unit_df["base_distance"], errors="coerce").to_numpy()
                        order = np.argsort(ordv)
                    elif "growth_length" in unit_df.columns and unit_df["growth_length"].notna().any():
                        ordv = pd.to_numeric(unit_df["growth_length"], errors="coerce").to_numpy()
                        order = np.argsort(ordv)
                    else:
                        order = np.argsort(pd.to_numeric(unit_df["cyl_ID"], errors="coerce").to_numpy())

                    centers_o = centers[order]
                    first_center = centers_o[0]
                    last_center = centers_o[-1]

                    axis_vec = last_center - first_center
                    axis_len = float(np.linalg.norm(axis_vec))
                    if axis_len > 0:
                        axis_u = axis_vec / axis_len
                        vols = pd.to_numeric(unit_df["volume"], errors="coerce").to_numpy(dtype=float)
                        vols = np.where(np.isfinite(vols) & (vols > 0), vols, 0.0)
                        vtot = float(np.sum(vols))
                        if vtot > 0:
                            com = (centers * vols[:, None]).sum(axis=0) / vtot
                            s_first = float(first_center @ axis_u)
                            s_last = float(last_center @ axis_u)
                            s_com = float(com @ axis_u)
                            denom = abs(s_last - s_first)
                            origin2com = float(100.0 * abs(s_com - s_first) / denom) if denom > 0 else np.nan
                        else:
                            origin2com = np.nan
                    else:
                        origin2com = np.nan

                    # save per-unit point subset from 2022 (XYZ)
                    saved_xyz = ""
                    try:
                        search_radius = 1.5 * max_radius
                        out_subdir = self.out_dir / "lost_branches_xyz" / f"tree_{tree_index_str}"
                        out_path = self._save_branch_points_xyz(
                            out_subdir=out_subdir,
                            tree_index_str=tree_index_str,
                            branch_id=branch_id,
                            points_2022=points_2022,
                            tree_2022_kd=tree_2022,
                            starts=s2,
                            ends=e2,
                            search_radius=search_radius,
                        )
                        saved_xyz = str(out_path) if out_path else ""
                    except Exception:
                        saved_xyz = ""

                    fallen_branch_rows.append({
                        "tree_index": tree_index_str,
                        "tree_folder": tree_folder,
                        "branch_id": branch_id,
                        "volume": total_volume,
                        "length": branch_length,
                        "surface": total_surface,
                        "max_radius": max_radius,
                        "angle": avg_angle_deg,
                        "origin2com": origin2com,
                        "dir_dx": dx,
                        "dir_dy": dy,
                        "dir_dz": dz,
                        "azimuth_deg": azimuth_deg,
                        "inclination_deg": inclination_deg,
                        "dbh_m": dbh_m,
                        "height_m": height_m,
                        "saved_xyz": saved_xyz,
                    })

        denom = float(pd.to_numeric(cylinders_df["volume"], errors="coerce").sum())
        lost_volume_ratio = (total_branch_volume / denom) if denom > 0 else np.nan

        tree_metrics = {
            "tree_index": tree_index_str,
            "tree_folder": tree_folder,
            "lost_volume_ratio": lost_volume_ratio,
            "dbh_m": dbh_m,
            "dbh_n_cyl": dbh_n_cyl,
            "height_m": height_m,
            "n_cylinders": int(len(cylinders_df)),
            "n_fallen_branches": int(len(fallen_branch_rows)),
        }

        return lost_volume_ratio, cylinders_df, fallen_branch_rows, tree_metrics


    
    def _branch_direction(self, df: pd.DataFrame):
        """
        Returns (dx,dy,dz, azimuth_deg, inclination_deg) for a branch.
        Uses proximal->distal centers if base_distance exists, otherwise growth_length, else cyl_ID.
        """
        if df.empty:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        # order cylinders along the branch
        if "base_distance" in df.columns and df["base_distance"].notna().any():
            dd = pd.to_numeric(df["base_distance"], errors="coerce")
            dfo = df.iloc[np.argsort(dd.to_numpy())].copy()
        elif "growth_length" in df.columns and df["growth_length"].notna().any():
            dd = pd.to_numeric(df["growth_length"], errors="coerce")
            dfo = df.iloc[np.argsort(dd.to_numpy())].copy()
        else:
            dfo = df.sort_values("cyl_ID").copy()

        # cylinder centers
        s = dfo[["startX","startY","startZ"]].to_numpy(dtype=float)
        e = dfo[["endX","endY","endZ"]].to_numpy(dtype=float)
        centers = 0.5 * (s + e)

        # proximal = first center, distal = last center
        v = centers[-1] - centers[0]
        dx, dy, dz = float(v[0]), float(v[1]), float(v[2])

        # azimuth in XY
        az = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        # inclination relative to horizontal plane
        horiz = np.hypot(dx, dy)
        inc = np.degrees(np.arctan2(dz, horiz))

        return (dx, dy, dz, float(az), float(inc))


    def _save_branch_points_xyz(
        self,
        out_subdir: Path,
        tree_index_str: str,
        branch_id: int,
        points_2022: np.ndarray,
        tree_2022_kd: cKDTree,
        starts: np.ndarray,
        ends: np.ndarray,
        search_radius: float,
    ):
        out_subdir.mkdir(parents=True, exist_ok=True)
        sample = np.vstack([starts, ends])  # (2N, 3)
        neigh = tree_2022_kd.query_ball_point(sample, r=float(search_radius))

        idxs = set()
        for n in neigh:
            idxs.update(n)
        if not idxs:
            return None

        idxs = np.array(sorted(idxs), dtype=int)
        pts = points_2022[idxs, :3]

        out_path = out_subdir / f"tree_{tree_index_str}_branch_{branch_id}.xyz"
        np.savetxt(out_path, pts, fmt="%.6f")
        return out_path


    # ----------------------------
    # Batch runner (incremental writes)
    # ----------------------------
    def run(self):
        # ---- locate source 2022 folder (support both patterns) ----
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

        if not src_files:
            raise FileNotFoundError(f"No .xyz files found in 2022 source folder: {src_2022}")
        if not tgt_files:
            raise FileNotFoundError(f"No .xyz files found in 2023 target folder: {tgt_2023}")

        src_by_idx = {self._tree_index_str(p.name): p for p in src_files}
        tgt_by_idx = {self._tree_index_str(p.name): p for p in tgt_files}

        common = sorted(set(src_by_idx.keys()) & set(tgt_by_idx.keys()), key=lambda x: int(x))
        if not common:
            raise RuntimeError("No matching tree indices between 2022 source and 2023 target folders.")

        # ---- ensure outputs exist with headers ----
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.tree_out.exists():
            pd.DataFrame(columns=self.tree_cols).to_csv(self.tree_out, index=False)
        if not self.fallen_out.exists():
            pd.DataFrame(columns=self.fallen_cols).to_csv(self.fallen_out, index=False)
        if not self.fail_out.exists():
            pd.DataFrame(columns=self.fail_cols).to_csv(self.fail_out, index=False)

        # ---- optional resume: skip indices already written in tree_out ----
        done = set()
        try:
            if self.tree_out.exists():
                prev = pd.read_csv(self.tree_out, usecols=["tree_index"], dtype={"tree_index": str})
                done = set(prev["tree_index"].dropna().astype(str).tolist())
        except Exception:
            done = set()

        todo = [idx for idx in common if idx not in done]
        if not todo:
            print("[run] Nothing to do: all common indices already in tree output.")
            return pd.DataFrame(), pd.DataFrame()

        fallen_rows_all = []
        tree_rows_all = []
        fail_rows_all = []

        # in-memory buffers (flush_every controls write frequency)
        fallen_buf = []
        tree_buf = []
        fail_buf = []

        def _flush_buffers():
            nonlocal fallen_buf, tree_buf, fail_buf

            if tree_buf:
                tree_df = pd.DataFrame(tree_buf)
                for c in self.tree_cols:
                    if c not in tree_df.columns:
                        tree_df[c] = np.nan
                tree_df = tree_df[self.tree_cols]
                tree_df.to_csv(self.tree_out, mode="a", header=False, index=False)
                tree_rows_all.extend(tree_df.to_dict(orient="records"))
                tree_buf = []

            if fallen_buf:
                fallen_df = pd.DataFrame(fallen_buf)
                for c in self.fallen_cols:
                    if c not in fallen_df.columns:
                        fallen_df[c] = np.nan
                fallen_df = fallen_df[self.fallen_cols]
                fallen_df.to_csv(self.fallen_out, mode="a", header=False, index=False)
                fallen_rows_all.extend(fallen_df.to_dict(orient="records"))
                fallen_buf = []

            if fail_buf:
                fail_df = pd.DataFrame(fail_buf)
                for c in self.fail_cols:
                    if c not in fail_df.columns:
                        fail_df[c] = np.nan
                fail_df = fail_df[self.fail_cols]
                fail_df.to_csv(self.fail_out, mode="a", header=False, index=False)
                fail_rows_all.extend(fail_df.to_dict(orient="records"))
                fail_buf = []

        for loop_i, tree_index_str in enumerate(todo):
            f2022 = src_by_idx[tree_index_str]
            f2023 = tgt_by_idx[tree_index_str]

            print(f"\n[Tree {loop_i+1}/{len(todo)}] idx={tree_index_str}")
            print("  2022:", f2022)
            print("  2023:", f2023)

            try:
                lost_ratio, cylinders_df, fallen_branch_rows, tree_metrics = self.process_one_tree(
                    tree_filename_2022=str(f2022),
                    tree_filename_2023=str(f2023),
                    tree_index_str=tree_index_str,
                )

                # tree_folder consistency
                tree_folder = tree_metrics.get("tree_folder", cylinders_df.attrs.get("rtwig_folder", ""))
                tree_metrics["tree_folder"] = tree_folder

                # buffer tree row
                tree_row = {k: tree_metrics.get(k, np.nan) for k in self.tree_cols}
                tree_buf.append(tree_row)

                # buffer fallen branches
                if fallen_branch_rows:
                    for r in fallen_branch_rows:
                        r.setdefault("lost_volume_ratio", lost_ratio)
                        r.setdefault("tree_folder", tree_folder)
                        r.setdefault("tree_index", tree_index_str)
                    fallen_buf.extend(fallen_branch_rows)

                print("  rTwig matched folder:", tree_folder)
                print(f"  lost_volume_ratio: {lost_ratio:.6f}")
                print(f"  fallen branches: {len(fallen_branch_rows)}")

            except Exception as e:
                fail_row = {"tree_index": tree_index_str, "tree_folder": "", "err": str(e)}
                fail_buf.append(fail_row)
                print(f"[SKIP] idx={tree_index_str}: {e}")

            # ---- flush cadence ----
            fe = int(getattr(self, "flush_every", 1))
            if fe <= 1 or ((loop_i + 1) % fe == 0):
                _flush_buffers()
                if fe > 1:
                    print(f"[WRITE] progress checkpoint at tree {loop_i+1}")

        # final flush
        _flush_buffers()

        return pd.DataFrame(fallen_rows_all), pd.DataFrame(tree_rows_all)


# ----------------------------
# Example usage
# ----------------------------
bc = BranchesChange(
    in_dir_src=r"D:\Chris\Hydro\Karl\translation\raw\part1",
    in_dir_target=r"D:\Karl\hydro\dataset\Working\single_trees\part1\2023\matched_xyz_species",
    qsm_dir=r"D:\Chris\Hydro\Karl\translation\rTwig\part1",
    out_dir=r"D:\Chris\Hydro\Karl\translation\change",
    radius=0.25,
    min_neighbors=1,
    voxel_size=0.25,
    min_samples=10,
    voxel_dilation=0,
    flush_every=1,
)
fallen_df, tree_df = bc.run()