from typing import Any, Optional
import numpy as np
from pandas import DataFrame
import os
import laspy as lp # type: ignore
import json
from typing import Any
import matplotlib.pyplot as plt

class Voxelization:
    def __init__(self, *, tree_dir: str|None=None, tree_filename: str|None=None) -> None:
        """Voxelization of a tree point cloud

        Args:
            tree_dir (str, optional): Tree point clouds base directory. Defaults to None.
            tree_filename (str, optional): Tree point cloud to voxelize. Defaults to None.
        """
                
        self._tree_dir: str | None = tree_dir
        self._tree_filename: str | None = tree_filename
        self._voxel_grid: dict[str, Any] | None = None
        self._metadata: dict[str, Any] | None = None
        self._voxel_df: DataFrame | None = None
        self._voxel_size: float | None = None
    
    
    def voxelize(self, points: np.ndarray|None= None, *, voxel_size: float=1, nb_voxel_points_min: int=3,
                    ref_min_point: np.ndarray|None=None) -> np.ndarray[Any, np.dtype[np.float64]] | None:
        """Voxelization of the 3D points from the numpy array
            Compute the voxel barycenter, points density, covariant matrix and its singular values and vectors

        Args:
            points (np.array): 3D points
            voxel_size (int, optional): Voxels size. Defaults to 1.
            nb_voxel_points_min (int, optional): Minimum number of point in a voxel to compute
                its covariant matrix and SVD. Defaults to 3.
            ref_min_point (np.array): Min point of the voxel.
                If None, this value is set to the min points of the provided points array.
                This value can be provided to compared multiple voxel and to be sure all voxels are aligned.
                Defautls to None
        """
        
        if points is None and self._tree_filename is None:
            raise ValueError('You must provide the points or the filename argument')
        
        if points is None:
            assert self._tree_filename is not None
            points = self.load_points(self._tree_dir, self._tree_filename)
        
        # Taille des voxels en m (unité des nuages d'arbre)
        self._voxel_size = voxel_size
        nb_voxel_points_min = 3 # Nombre de mininum de points par voxel
        self._voxel_grid = None
        self._voxel_df = None

        grid_size = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)
        nb_points = len(points)

        if ref_min_point is None:
            ref_min_point = np.min(points, axis=0)

        non_empty_voxel_keys, inverse, nb_pts_per_voxel = \
            np.unique(((points - ref_min_point) // voxel_size).astype(int),
                    axis=0, return_inverse=True, return_counts=True)

        idx_pts_vox_sorted = np.argsort(inverse) # Permet d'accélérer l'accès au points dans la boucle for

        voxel_labels = {} # Grille de voxels contenants leurs points et stats
        grid_barycenter: list[Any] = []
        grid_nb_points: list[int] = []
        grid_cov_matrix: list[Any] = []
        grid_svd_vecs: list[Any] = []
        grid_svd_vals: list[Any] = []
        start_idx=0

        for idx, vox_key in enumerate(non_empty_voxel_keys):
            end_idx = start_idx + nb_pts_per_voxel[idx]
            voxel_points = points[idx_pts_vox_sorted[start_idx:end_idx]]
            start_idx = end_idx
            
            nb_voxel_points = len(voxel_points)
            
            voxel_labels[tuple(vox_key)] = voxel_points
            
            grid_barycenter.append(np.mean(voxel_labels[tuple(vox_key)], axis=0))
            grid_nb_points.append(nb_voxel_points)
            
            # Matrice de covariance
            if nb_voxel_points >= nb_voxel_points_min:
                cov_matrix = np.cov(voxel_points, rowvar=False) # Vecteurs colonne
                
                # Calcul du SVD de la matrice de covariance
                U, S, _ = np.linalg.svd(cov_matrix)
            else:
                cov_matrix, U, S = None, None, None
                
            grid_cov_matrix.append(cov_matrix) # Matrice de covvariance
            # Les 3 valeurs propres sont rangées par ordre décroissant
            grid_svd_vals.append(S)
            # Les 3 vecteurs propres colonne sont associes aux 3 valeurs de S
            grid_svd_vecs.append(U)
            
        self._voxel_grid = {
            'voxel_label': voxel_labels,
            'barycenter': grid_barycenter,
            'nb_points': grid_nb_points,
            'density': (np.array(grid_nb_points) / nb_points).tolist(),
            'cov_matrix': grid_cov_matrix,
            'svd_vecs': grid_svd_vecs,
            'svd_vals': grid_svd_vals
            }
        
        self._metadata = {
            'grid_size': grid_size.tolist(),
            'ref_min_point': [] if ref_min_point is None else ref_min_point.tolist(),
            'voxels_size': voxel_size,
            'nb_points_total': nb_points
        }
        
        return ref_min_point


    def get_voxel_grid(self) -> dict[str, Any] | None:
        """Return the voxel grid generated by the voxelize method

        Returns:
            dict: Dictionnary containing all the voxel grid
        """
        return self._voxel_grid


    def get_metadata(self) -> dict[str, Any] | None:
        """Return the voxel metadata generated by the voxelize method

        Returns:
            dict: Dictionnary containing all the voxel metadata
        """
        return self._metadata


    def get_dataframe(self) -> DataFrame | None:
        """Return the voxel grid in a pandas DataFrame
            The matrixes are flattened in one line to allow the 2D pandas DataFrame format

        Returns:
            DataFrame: The pandas DataFrame
        """
        
        if self._voxel_grid is None:
            return None
        
        if self._voxel_df is None:
            self._create_data_frame()
        
        if self._voxel_df is None:
            return None

        return self._voxel_df


    def save_to_csv(self, file_path: str) -> bool:
        """Save the voxel grid to a CSV file

        Args:
            file_path (str): Destination file path

        Returns:
            bool: True if OK, False if Error
        """
        if self._voxel_grid is None:
            return False
        
        if self._voxel_df is None:
            self._create_data_frame()
        
        if self._voxel_df is None:
            return False
        
        self._voxel_df.to_csv(file_path, index=False)
        return True


    def __str__(self) -> str:
        """Print the voxel grid to the console
        """
        if self._voxel_grid is None or self._metadata is None:
            return ""
        
        vox_keys = [list(vox) for vox in self._voxel_grid['voxel_label'].keys()]
        ref_min_point = np.round(self._metadata['ref_min_point'], 3) # Précis au mm
        recentred_voxels = [list(np.array(vox)*self._voxel_size + ref_min_point) for vox in vox_keys]
        barycenters = [[x, y, z] for (x, y, z) in np.round(self._voxel_grid['barycenter'], 3)]
        nb_points = self._voxel_grid['nb_points']
        density = self._voxel_grid['density']
        cov_mat_lines = [mat.reshape(1, -1).tolist()[0] if mat is not None else np.zeros((1, 9)).tolist()[0] for mat in self._voxel_grid['cov_matrix']]
        svd_vals = [[val[0], val[1], val[2]] if val is not None else [0,0,0] for val in self._voxel_grid['svd_vals']]
        svd_vecs = [mat.T.reshape(1, -1).tolist()[0] if mat is not None else np.zeros((1, 9)).tolist()[0] for mat in self._voxel_grid['svd_vecs']]


        return (
            "Metadata:\n"
            f"  Voxels size: {self._metadata['voxels_size']} | Grid size: {self._metadata['grid_size']} | Nb total points: {self._metadata['nb_points_total']} | Ref min point: {ref_min_point}\n"
            "Voxel grid:\n"
            f"  vox_keys:         {vox_keys}\n"
            f"  Voxels (keys):    {vox_keys}\n"
            f"  Voxels recentrés: {recentred_voxels}\n"
            f"  Barycentres:      {barycenters} \n"
            f"  Nb points:        {nb_points}\n"
            f"  Density  :        {density}\n"
            f"  Covariant mat:    {cov_mat_lines}\n"
            f"  Valeurs propres:  {svd_vals}\n"
            f"  Vecteurs propres: {svd_vecs}\n"
        )


    def _create_data_frame(self) -> None:
        """Create the voxel pandas DataFrame
        """
        voxel_size = self._voxel_size
        
        assert self._voxel_grid is not None and self._metadata is not None
        
        vox_keys = [list(vox) for vox in self._voxel_grid['voxel_label'].keys()]
        ref_min_point = np.round(self._metadata['ref_min_point'], 3) # Précis au mm
        recentred_voxels = [list(np.array(vox)*voxel_size + ref_min_point) for vox in vox_keys]
        barycenters = [[x, y, z] for (x, y, z) in np.round(self._voxel_grid['barycenter'], 3)]
        nb_points = self._voxel_grid['nb_points']
        density = self._voxel_grid['density']
        cov_mat_lines = [mat.reshape(1, -1).tolist()[0]
                         if mat is not None else np.zeros((1, 9)).tolist()[0]
                         for mat in self._voxel_grid['cov_matrix']]
        svd_vals = [[val[0], val[1], val[2]]
                    if val is not None else [0,0,0]
                    for val in self._voxel_grid['svd_vals']]
        svd_vecs = [mat.T.reshape(1, -1).tolist()[0]
                    if mat is not None else np.zeros((1, 9)).tolist()[0]
                    for mat in self._voxel_grid['svd_vecs']]

        # Create a pandas DataFrame
        col_header: list[str] = ['X', 'Y', 'Z', # barycenters
                    'NbPoints', # Number of points
                    'Density', # Density
                    'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z', # Voxel label (indexes)
                    'VoxPos_X', 'VoxPos_Y', 'VoxPos_Z' # Voxel origin position
                    ]

        col_header += [f"CovMat_{i}{j}" for i in range(3) for j in range(3)] # Covariant matrix
        col_header += [f"EigenVal{i}" for i  in range(3)] # SVD values 
        col_header += [f"EigenVec{i}_{val}" for i in range(3) for val in ['X', 'Y', 'Z']] # SVD values 

        data: np.ndarray = np.hstack([np.array(barycenters),
                np.array(nb_points).reshape(-1,1),
                np.array(density).reshape(-1,1),
                np.array(vox_keys),
                np.array(recentred_voxels),
                np.array(cov_mat_lines),
                np.array(svd_vals),
                np.array(svd_vecs)
                ])

        df = DataFrame(data, columns=col_header)

        df[['NbPoints', 'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z']] = \
            df[['NbPoints', 'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z']].astype(int)
            
        self._voxel_df = df


    def save_voxels(self, *, results_dir: str|None=None, tree_filename: str|None=None, verbose: bool=True) -> None:
        if tree_filename is None:
            tree_filename = self._tree_filename
            if tree_filename is None:
                raise ValueError('You must provide a filename')

        if results_dir is None:
            results_dir = self._tree_dir
            
        assert results_dir is not None
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        assert self._voxel_size is not None
        root_filename: str = f"{os.path.splitext(tree_filename)[0]}_{self._voxel_size*100:.0f}cm"
        csv_file_path: str = os.path.join(results_dir, root_filename + '.csv')

        # Save CSV file
        self.save_to_csv(csv_file_path)
        if verbose:
            print(f"CSV file '{csv_file_path}' saved")
        
        # XYZ text file (only barycenters)
        voxel_grid : dict[str, Any] | None = self.get_voxel_grid()
        assert voxel_grid is not None
        barycenters: np.ndarray = voxel_grid['barycenter']
        xyz_file_path: str = os.path.join(results_dir, root_filename + '_points.xyz')

        np.savetxt(xyz_file_path, barycenters, delimiter=";", fmt="%s")
        
        if verbose:
            print(f"XYZ file '{xyz_file_path}' saved")
        
        # Saving voxel_grid metadata
        json_file_path = os.path.join(results_dir, root_filename + '.json')
        metadata = self.get_metadata()
        with open(json_file_path, 'w') as file:
            json.dump(metadata, file, indent=4)
            
        if verbose:
            print(f"Json metadata file '{json_file_path}' saved")


    @staticmethod
    def _load_las_points(file_path: str) -> np.ndarray:
        # Import dataset
        pcd_las: lp.LasData = lp.read(file_path)
        
        lp.read(file_path)
        x: np.ndarray = np.array(pcd_las.x)
        y: np.ndarray = np.array(pcd_las.y)
        z: np.ndarray = np.array(pcd_las.z)
        return np.vstack((x, y, z)).transpose()


    @staticmethod
    def _load_pcd_points(file_path) -> np.ndarray:
        # Lire le fichier PCD
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asanyarray(pcd.points)


    @classmethod
    def load_points(cls, tree_dir: str | None, tree_filename: str) -> np.ndarray:

        # --- CLEAN QUOTES / STRAY CHARACTERS ---
        tree_filename = tree_filename.strip().strip('"').strip("'")

        pcd_path: str = tree_filename if tree_dir is None else os.path.join(tree_dir, tree_filename)

        file_ext: str = os.path.splitext(pcd_path)[1].lower()

        match file_ext:
            case '.las' | '.laz':
                xyz = cls._load_las_points(pcd_path)

            case '.pcd':
                xyz = cls._load_pcd_points(pcd_path)

            case '.xyz':
                xyz = np.loadtxt(pcd_path)

            case _:
                raise ValueError(f"Unknown format: {pcd_path} (ext={file_ext})")

        return xyz
