import os
import glob
import numpy as np
import laspy
import open3d as o3d

# =============================
# Config
# =============================
CLEAN_DIR = r"D:/Chris/Hydro/Karl/translation/raw/part6"
RAW_DIR   = r"D:/Karl/hydro/dataset/Working/single_trees/part6/2022/matched_las_species_georef"
OUT_DIR   = r"D:/Chris/Hydro/Karl/translation/translated/part6"

VOXEL = 0.03
CROP_PAD = 1.5
ICP_MAX_CORR = 0.25
ICP_ITERS = 60

MIN_FITNESS = 0.60
MAX_RMSE = 0.30

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# Helpers
# =============================
def extract_tree_id(path: str):
    """
    Return integer tree ID from the 3rd token:
    'arbre 2022 00 17.70 GLEDITSIA-T.xyz' -> 0
    """
    name = os.path.splitext(os.path.basename(path))[0].strip()
    parts = name.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None

def read_las_or_laz(path: str) -> np.ndarray:
    las = laspy.read(path)  # .laz requires lazrs installed
    return np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

def read_clean_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".las", ".laz"):
        return read_las_or_laz(path)

    # LAS saved with .xyz
    with open(path, "rb") as f:
        sig = f.read(4)
    if sig == b"LASF":
        return read_las_or_laz(path)

    # ASCII XYZ
    try:
        pts = np.loadtxt(path, usecols=(0, 1, 2), dtype=np.float64)
    except Exception:
        pts = np.genfromtxt(path, usecols=(0, 1, 2), dtype=np.float64, invalid_raise=False)
        pts = pts[~np.isnan(pts).any(axis=1)]

    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts

def np_to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz)
    return p

def write_laz_from_template(out_laz_path: str, xyz: np.ndarray, template_las_path: str) -> None:
    tmpl = laspy.read(template_las_path)
    header = tmpl.header

    out = laspy.LasData(header)
    n = xyz.shape[0]
    out.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)

    # remove non-finite rows to avoid laspy cast warnings/errors
    m = np.isfinite(xyz).all(axis=1)
    xyz = xyz[m]
    if xyz.shape[0] == 0:
        raise ValueError("All points non-finite after transform.")

    out.x = xyz[:, 0]
    out.y = xyz[:, 1]
    out.z = xyz[:, 2]

    out.write(out_laz_path)

# =============================
# Discover inputs
# =============================
clean_paths = []
for ext in ("*.xyz", "*.XYZ", "*.las", "*.LAS", "*.laz", "*.LAZ"):
    clean_paths.extend(glob.glob(os.path.join(CLEAN_DIR, ext)))

raw_candidates = []
for ext in ("*.las", "*.LAS", "*.laz", "*.LAZ"):
    raw_candidates.extend(glob.glob(os.path.join(RAW_DIR, ext)))

# Drop COPC (prefer plain LAS)
raw_candidates = [p for p in raw_candidates if "copc" not in os.path.basename(p).lower()]

print(f"Found cleaned candidates: {len(clean_paths)}")
print(f"Found raw candidates (non-COPC): {len(raw_candidates)}")

# =============================
# Index cleaned by int ID
# =============================
clean_by_id = {}
bad_clean = 0
dup_clean = 0
for p in clean_paths:
    tid = extract_tree_id(p)
    if tid is None:
        bad_clean += 1
        continue
    if tid in clean_by_id:
        dup_clean += 1
    clean_by_id[tid] = p

# =============================
# Select one raw file per int ID, prefer .las
# =============================
raw_by_id = {}
bad_raw = 0
dup_raw = 0

for p in raw_candidates:
    tid = extract_tree_id(p)
    if tid is None:
        bad_raw += 1
        continue

    ext = os.path.splitext(p)[1].lower()
    cur = raw_by_id.get(tid)
    if cur is None:
        raw_by_id[tid] = p
    else:
        dup_raw += 1
        cur_ext = os.path.splitext(cur)[1].lower()
        if cur_ext != ".las" and ext == ".las":
            raw_by_id[tid] = p

raw_paths = list(raw_by_id.values())

print(f"Clean IDs: {len(clean_by_id)} (bad={bad_clean}, dup={dup_clean})")
print(f"Raw IDs:   {len(raw_by_id)} (bad={bad_raw}, dup_seen={dup_raw})")

# =============================
# Preflight mismatch report (this should be near 0 if 1-to-1)
# =============================
raw_ids = set(raw_by_id.keys())
clean_ids = set(clean_by_id.keys())
missing_in_clean = sorted(raw_ids - clean_ids)
missing_in_raw = sorted(clean_ids - raw_ids)

print("IDs present in raw but missing in clean (first 50):", missing_in_clean[:50])
print("IDs present in clean but missing in raw (first 50):", missing_in_raw[:50])

# =============================
# Main processing loop
# =============================
processed = 0
no_match = 0
read_fail = 0
too_small = 0
bad_icp = 0
nonfinite = 0

for tid, raw_path in sorted(raw_by_id.items(), key=lambda kv: kv[0]):
    clean_path = clean_by_id.get(tid)
    if clean_path is None:
        no_match += 1
        continue

    base = os.path.splitext(os.path.basename(raw_path))[0]

    try:
        tgt = read_las_or_laz(raw_path)
        src = read_clean_points(clean_path)
    except Exception:
        read_fail += 1
        continue

    if tgt.shape[0] < 100 or src.shape[0] < 50:
        too_small += 1
        continue

    # Coarse translation
    t0 = tgt.mean(axis=0) - src.mean(axis=0)
    src0 = src + t0

    # Crop target
    bb_min = src0.min(axis=0) - CROP_PAD
    bb_max = src0.max(axis=0) + CROP_PAD
    mask = (
        (tgt[:, 0] >= bb_min[0]) & (tgt[:, 0] <= bb_max[0]) &
        (tgt[:, 1] >= bb_min[1]) & (tgt[:, 1] <= bb_max[1]) &
        (tgt[:, 2] >= bb_min[2]) & (tgt[:, 2] <= bb_max[2])
    )
    tgt_crop = tgt[mask]
    if tgt_crop.shape[0] < 200:
        tgt_crop = tgt

    # Downsample for ICP
    src_ds = np_to_pcd(src0).voxel_down_sample(VOXEL)
    tgt_ds = np_to_pcd(tgt_crop).voxel_down_sample(VOXEL)

    reg = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds,
        max_correspondence_distance=ICP_MAX_CORR,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERS)
    )

    if reg.fitness < MIN_FITNESS or reg.inlier_rmse > MAX_RMSE:
        bad_icp += 1
        continue

    R = reg.transformation[:3, :3]
    t_icp = reg.transformation[:3, 3]
    t = R @ t0 + t_icp

    aligned = (src @ R.T) + t

    m = np.isfinite(aligned).all(axis=1)
    if not m.all():
        aligned = aligned[m]
        nonfinite += 1
    if aligned.shape[0] < 50:
        too_small += 1
        continue

    out_path = os.path.join(OUT_DIR, f"{base}_georef.laz")
    try:
        write_laz_from_template(out_path, aligned, raw_path)
    except Exception:
        read_fail += 1
        continue

    processed += 1
    if processed % 50 == 0:
        print(f"Processed {processed} (last id={tid} file={base}) fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f}")

print(
    "Done.\n"
    f"  processed   = {processed}\n"
    f"  no_match    = {no_match}\n"
    f"  read_fail   = {read_fail}\n"
    f"  too_small   = {too_small}\n"
    f"  bad_icp     = {bad_icp}\n"
    f"  nonfinite   = {nonfinite}\n"
)
