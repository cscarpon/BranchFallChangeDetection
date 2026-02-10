import numpy as np
import pandas as pd # type: ignore
from pandas import DataFrame
import os
import glob
from typing import Any
import re


class VoxelChangeDetector():
    def __init__(self, df_1: DataFrame|None=None, df_2: DataFrame|None=None) -> None:
        """
        Initialize a VoxelChangeDetector instance with optional voxels DataFrames for comparison.
            When df_1 and df_2 are None, they must be set later with the 'voxelize_trees' method.

        Args:
            df_1 (DataFrame | None, optional): The first DataFrame representing tree voxel data.
            df_2 (DataFrame | None, optional): The second DataFrame representing tree voxel data.
        """
        self._df_1: DataFrame | None = df_1
        self._df_2: DataFrame | None = df_2
        self._df_diff: DataFrame | None = None
        self._stats: dict[str, Any] | None = None
        self._source_counts: DataFrame | None = None
        self._tree1_filename: str | None = None
        self._tree2_filename: str | None = None
        self._ref_min_point: np.ndarray | None = None
    
    @property
    def dataframe_1(self) -> DataFrame | None:
        return self._df_1

    
    @property
    def dataframe_2(self) -> DataFrame | None:
        return self._df_2
    
    @property
    def dataframe_diff(self) -> DataFrame | None:
        return self._df_diff

    
    def voxelize_trees(self, *, trees_dir: str|None=None, tree1_filename: str, tree2_filename: str,
                       voxel_size: float=0.5, nb_voxel_points_min: int=3) -> tuple[DataFrame|None, DataFrame|None]:
        """Voxelize a pair of trees to compare them

        Args:
            trees_dir (str | None, optional): Root directory of both trees.
                Trees may be directly in this directory or in a subfolder. Defaults to None.
            tree1_filename (str): First tree filename. The subdirectory or a full path (if trees_dir is None) may also added
            tree2_filename (str): second tree filename. The subdirectory or a full path (if trees_dir is None) may also added
            voxel_size (float, optional): Voxels size. Defaults to 50 cm (0.5).
            nb_voxel_points_min (int, optional): Minimum number on a voxel to be considered as non empty (to comptutes SVD).
                Defaults to 3.

        Returns:
            tuple[DataFrame|None, DataFrame|None]: The DataFrames of the two voxelized trees
        """

        from voxelization import Voxelization

        assert tree1_filename is not None and tree2_filename is not None
        
        self._tree1_filename = tree1_filename
        self._tree2_filename = tree2_filename
        
        # Creation two trees voxelization objet to generag the voxel grids
        voxelization1: Voxelization = Voxelization(tree_dir=trees_dir, tree_filename=tree1_filename)
        voxelization2: Voxelization = Voxelization(tree_dir=trees_dir, tree_filename=tree2_filename)

        # Voxelization of both trees
        ref_min_point: np.ndarray | None = voxelization1.voxelize(voxel_size=voxel_size, nb_voxel_points_min=nb_voxel_points_min)
        voxelization2.voxelize(voxel_size=voxel_size, nb_voxel_points_min=nb_voxel_points_min, ref_min_point=ref_min_point)
        
        self._df_1 = voxelization1.get_dataframe()
        self._df_2 = voxelization2.get_dataframe()

        self._ref_min_point = ref_min_point
        
        return self._df_1, self._df_2
    
    @property
    def ref_min_point(self) -> np.ndarray | None:
        return self._ref_min_point


    def compare_voxels(self) -> DataFrame:
        assert self._df_1 is not None and self._df_2 is not None
        self._df_diff = self.compare_voxel_dataframes(self._df_1, self._df_2)
        return self._df_diff
    
    
    def calculate_stats(self, n_std_threshold_min: float=float('inf')) -> dict[str, Any]:
        """
        Calculate the statistics from the difference of two voxel grids

        Args:
            n_std_threshold_min (float, optional): Minimum number of standard deviations difference required to consider a voxel an outlier.
                Defaults to float('inf').

        Returns:
            dict[str, Any]: A dictionary containing the following keys:
                - 'stats': DataFrame with mean and standard deviation of Log2_NbPoints_ratio for source 3.
                - 'source_counts': DataFrame with the number of rows for each source tree 1, 3 and both.
                - 'outliers': dict with the following keys:
                    - 'n_std_threshold_min': The minimum number of standard deviations difference required to consider a voxel an outlier.
                    - 'n_outliers': The number of outliers.
                    - 'outlier_indices': The indices of the outliers in the original DataFrame.
        """
        assert self._df_diff is not None
        return self.calculate_stats_static(self._df_diff, n_std_threshold_min)
    
    
    def plot_distribution_log2(self, nb_bins: int = 50, *,
                               is_display: bool=True,
                               is_save: bool=False, figure_filename: str|None=None) -> None:
        from . import VoxelVisualizer

        assert self._df_diff is not None
        VoxelVisualizer.plot_distribution_log2_static(self._df_diff, nb_bins, is_display=is_display,
                                                      is_save=is_save, figure_filename=figure_filename)
    
    
    def plot_distribution(self, nb_bins: int=50) -> None:
        from . import VoxelVisualizer

        assert self._df_diff is not None
        VoxelVisualizer.plot_distribution_static(self._df_diff, nb_bins)
    
    
    def plot_stats(self, *, is_display: bool=True, is_save: bool=False,
                   figure_filename: str|None=None) -> None:
        from . import VoxelVisualizer

        if self._source_counts is not None and self._stats is not None:
            VoxelVisualizer.plot_stats(self._source_counts, self._stats, is_display=is_display,
                                       is_save=is_save, figure_filename=figure_filename)
             
              
    @staticmethod # type: ignore
    def compare_voxel_dataframes(df1: DataFrame, df2: DataFrame) -> DataFrame:
        # Merge the dataframes on VoxLabel_X, VoxLabel_Y, VoxLabel_Z
        merged_df: DataFrame = pd.merge(df1, df2, on=['VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z'], how='outer', suffixes=('_1', '_2'))

        # Determine the source of each label before filling NaN values
        conditions: list = [
            merged_df['NbPoints_1'].notna() & merged_df['NbPoints_2'].notna(),
            merged_df['NbPoints_1'].notna() & merged_df['NbPoints_2'].isna(),
            merged_df['NbPoints_1'].isna() & merged_df['NbPoints_2'].notna()
        ]
        choices: list[str] = ['3', '1', '2']
        merged_df['Source'] = np.select(conditions, choices, default='0')

        # Fill NaN values with 0 for NbPoints columns
        merged_df.fillna({'NbPoints_1': 0, 'NbPoints_2': 0}, inplace=True)

        # Calculate the difference in NbPoints (probably not usefull, but who knows)
        merged_df['NbPoints_diff'] = merged_df['NbPoints_1'] - merged_df['NbPoints_2']

        # Calculate the ratio of NbPoints, set to inf where NbPoints_2 is 0
        merged_df['NbPoints_ratio'] = np.where(merged_df['NbPoints_2'] != 0,
                                            merged_df['NbPoints_1'] / merged_df['NbPoints_2'],
                                            np.inf)

        # Pre-calculate inf conditions and set log values
        merged_df['Log2_NbPoints_ratio'] = np.nan
        inf_mask = merged_df['NbPoints_ratio'] == np.inf
        zero_mask = merged_df['NbPoints_ratio'] == 0
        log_mask = ~(inf_mask | zero_mask)

        # Apply conditions
        merged_df.loc[inf_mask, 'Log2_NbPoints_ratio'] = np.inf
        merged_df.loc[zero_mask, 'Log2_NbPoints_ratio'] = -np.inf
        merged_df.loc[log_mask, 'Log2_NbPoints_ratio'] = np.log2(merged_df.loc[log_mask, 'NbPoints_ratio'])

        # Select only the columns of interest
        columns_of_interest: list[str] = [
            'VoxLabel_X', 'VoxLabel_Y', 'VoxLabel_Z',
            'NbPoints_1', 'NbPoints_2',
            'NbPoints_diff', 'NbPoints_ratio', 'Log2_NbPoints_ratio',
            'Source'
        ]
        
        return merged_df[columns_of_interest]
    
    @staticmethod
    def _unique_rows_int32(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int32)
        if a.size == 0:
            return a.reshape(0, 3)
        view = a.view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
        uniq = np.unique(view).view(np.int32).reshape(-1, 3)
        return uniq

    @staticmethod
    def _dilate_labels(labels_xyz: np.ndarray, r: int) -> np.ndarray:
        """
        labels_xyz: (M,3) int voxel indices
        r: dilation radius in voxels
        returns: (K,3) int expanded + unique
        """
        labels_xyz = np.asarray(labels_xyz, dtype=np.int32)
        if r <= 0 or labels_xyz.size == 0:
            return labels_xyz.reshape(-1, 3)

        offsets = np.array(
            [(dx, dy, dz)
            for dx in range(-r, r + 1)
            for dy in range(-r, r + 1)
            for dz in range(-r, r + 1)],
            dtype=np.int32
        )
        expanded = (labels_xyz[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        return VoxelChangeDetector._unique_rows_int32(expanded)

    @staticmethod
    def missing_tree1_voxels(df1: pd.DataFrame, df2: pd.DataFrame, dilation_voxels: int = 0) -> pd.DataFrame:
        """
        Return voxels occupied in df1 that are NOT matched by df2 within dilation.
        Output columns: VoxLabel_X/Y/Z, NbPoints_1, NbPoints_2, Source (like compare_voxels output)
        """
        L1 = df1[["VoxLabel_X","VoxLabel_Y","VoxLabel_Z"]].to_numpy(dtype=np.int32)
        L2 = df2[["VoxLabel_X","VoxLabel_Y","VoxLabel_Z"]].to_numpy(dtype=np.int32)

        if L1.size == 0:
            return pd.DataFrame(columns=["VoxLabel_X","VoxLabel_Y","VoxLabel_Z","NbPoints_1","NbPoints_2","Source"])

        if L2.size == 0:
            return pd.DataFrame({
                "VoxLabel_X": L1[:,0],
                "VoxLabel_Y": L1[:,1],
                "VoxLabel_Z": L1[:,2],
                "NbPoints_1": df1["NbPoints"].to_numpy(dtype=np.int32),
                "NbPoints_2": 0,
                "Source": "1",
            })

        # dilate L2 in voxel space
        r = int(dilation_voxels)
        if r > 0:
            offsets = np.array([(dx,dy,dz) for dx in range(-r,r+1) for dy in range(-r,r+1) for dz in range(-r,r+1)], dtype=np.int32)
            L2 = (L2[:,None,:] + offsets[None,:,:]).reshape(-1,3)

            # unique rows
            view = L2.view([("x",np.int32),("y",np.int32),("z",np.int32)])
            L2 = np.unique(view).view(np.int32).reshape(-1,3)

        # hash set membership for L2
        mins = L2.min(axis=0)
        shifted2 = L2 - mins
        ranges = shifted2.max(axis=0) + 1

        key2 = (shifted2[:,0].astype(np.int64) +
                shifted2[:,1].astype(np.int64) * ranges[0].astype(np.int64) +
                shifted2[:,2].astype(np.int64) * (ranges[0].astype(np.int64) * ranges[1].astype(np.int64)))
        set2 = set(key2.tolist())

        shifted1 = L1 - mins
        key1 = (shifted1[:,0].astype(np.int64) +
                shifted1[:,1].astype(np.int64) * ranges[0].astype(np.int64) +
                shifted1[:,2].astype(np.int64) * (ranges[0].astype(np.int64) * ranges[1].astype(np.int64)))

        missing_mask = np.fromiter((k not in set2 for k in key1), dtype=bool, count=key1.shape[0])

        miss = L1[missing_mask]
        nb1 = df1.loc[missing_mask, "NbPoints"].to_numpy(dtype=np.int32)

        return pd.DataFrame({
            "VoxLabel_X": miss[:,0],
            "VoxLabel_Y": miss[:,1],
            "VoxLabel_Z": miss[:,2],
            "NbPoints_1": nb1,
            "NbPoints_2": 0,
            "Source": "1",
        })



  
    @classmethod
    def calculate_stats_static(cls, df_diff: DataFrame, n_std_threshold_min: float=float('inf')) -> dict[str, Any]:
        """calculate the statistiques from the df_diff DataFrame, the difference of two voxel grids
            Also calulate the source_count, a DataFrame containing the number of trees from voxel 1, 2 and both

        Args:
            df_diff (DataFrame): The DataFrame containing the difference of the two voxel grids
            n_std_threshold_min (float, optional): Minimum number of standard deviations difference required to consider a voxel an outlier. Defaults to inf.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the following keys:
            - 'stats': DataFrame with mean and standard deviation of Log2_NbPoints_ratio for source 3.
            - 'source_counts': DataFrame with the number of rows for each source tree 1, 3 and both.
            - 'outliers': dict with the following keys:
                - 'n_std_threshold_min': The minimum number of standard deviations difference required to consider a voxel an outlier.
                - 'n_outliers': The number of outliers.
                - 'outlier_indices': The indices of the outliers in the original DataFrame.
        """
        
        df_source_3: DataFrame = df_diff[df_diff['Source'] == '3']

        # Calculate mean and standard deviation for Log2_NbPoints_ratio for source 3
        stats: DataFrame = df_source_3['Log2_NbPoints_ratio'].agg(['mean', 'std']).reset_index()

        # Calculate the number of rows for each source
        source_counts: DataFrame = df_diff['Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']

        # List of all possible sources
        all_sources = pd.DataFrame({'Source': ['1', '2', '3'], 'Count': [0, 0, 0]})

        # Combine source_counts with all_sources to fill missing ones with 0
        source_counts = pd.concat([all_sources, source_counts]).groupby('Source', as_index=False).sum()

        # Calculate the total number of rows
        total_count: DataFrame = pd.DataFrame({'Source': ['Total'], 'Count': [len(df_diff)]})

        # Append the total count to the source counts
        source_counts = pd.concat([source_counts, total_count], ignore_index=True)

        # Sort the source counts in the order of sources
        source_order: list = ['1', '2', '3', 'Total']
        source_counts['Source'] = pd.Categorical(source_counts['Source'], categories=source_order, ordered=True)
        source_counts = source_counts.sort_values('Source')
        
        # Find outliers
        outliers: dict[str, Any] = cls.find_outliers(stats, df_diff, n_std_threshold_min)
        
        return {
            'stats': stats,
            'source_counts': source_counts,
            'outliers': outliers
            }
            

    @staticmethod
    def _sort_filenames(filenames: list[str]) -> list[str]:
        # Function to extract the name and number from filenames
        def extraire_nom_et_numero(filename: str) -> tuple[str, int]:
            match: re.Match[str] | None = re.match(r'([a-zA-Z_/:\\\-\.#]*)(\d+)', filename)
            if match:
                nom, numero = match.groups()
                return nom, int(numero)
            return filename, 0  # Fallback if no match is found

        # Sort the list by name and number
        filenames_tries: list[str] = sorted(filenames, key=extraire_nom_et_numero)
        
        return filenames_tries


    @classmethod
    def compare_multiple_tree_pairs(cls, root_dir: str, subdir_1: str, subdir_2: str, *,
                                    pattern: str="*.las",
                                    voxel_size: float=0.5, nb_voxel_points_min: int=3) -> dict[str, list[Any]]:
        """Compare multiple trees in two subdirectory of a street at two different time.
            Both subdirs are in the root_dir.
            Each subdir must contain the same tree filenames. Only the same filnames will be compared. 

        Args:
            root_dir (str): Main directory containing the two street subdirectories
            subdir_1 (str): Subdirectory 1
            subdir_2 (str): Subdirectory 2
            pattern (str, optional): filename pattern. Defaults to "*.las".
            voxel_size (float, optional): Voxels size. Defaults to 50 cm (0.5).
            nb_voxel_points_min (int, optional): Minimum number on a voxel to be considered as non empty (to comptutes SVD).
                Defaults to 3.
                
        Returns:
            dict containing the following data:
                'tree_filenames' (list[str]): The sorted filename of the common trees use for the voxelizations List common_tree_filenames,
                'df_diffs' (list[DataFrame]]): The DataFrame containing the voxel grids differences.
                'df_trees_1' (list[DataFrame]]): The DataFrame containing voxel grids of the first trees.
                'df_trees_2' (list[DataFrame]]): The DataFrame containing voxel grids of the second trees.
        """
        root_dir_abs: str = os.path.abspath(root_dir)
        
        # Absolute path of each tree dir
        dir_path_1: str = os.path.normpath(os.path.join(root_dir_abs, subdir_1))
        dir_path_2: str = os.path.normpath(os.path.join(root_dir_abs, subdir_2))

        # Tree paths
        tree_paths_1: list[str] = glob.glob(os.path.join(dir_path_1, pattern))
        tree_paths_2: list[str] = glob.glob(os.path.join(dir_path_2, pattern))
        
        # Tree filenames
        tree_filenames_1: list[str] = cls._extract_tree_filenames(dir_path_1, tree_paths_1)
        tree_filenames_2: list[str] = cls._extract_tree_filenames(dir_path_2, tree_paths_2)
        
        # Sort filenames using alphabetic and numerique number: Tree_10 is after Tree_9

        # Common trees only
        common_tree_filenames: list[str] = cls._sort_filenames(list(set(tree_filenames_1) & set(tree_filenames_2)))

        df_diffs: list[DataFrame] = []
        df_trees_1: list[DataFrame|None] = []
        df_trees_2: list[DataFrame|None] = []
        
        for tree_filename in common_tree_filenames:
            tree_filename_1: str = os.path.join(subdir_1, tree_filename)
            tree_filename_2: str = os.path.join(subdir_2, tree_filename)

            voxel_change_detector = VoxelChangeDetector()
            
            df_1: DataFrame | None = None
            df_2: DataFrame | None = None
            df_1, df_2 = voxel_change_detector.voxelize_trees(trees_dir=root_dir,
                                                 tree1_filename=tree_filename_1, tree2_filename=tree_filename_2,
                                                 voxel_size=voxel_size, nb_voxel_points_min=nb_voxel_points_min)
            
            df_trees_1.append(df_1)
            df_trees_2.append(df_2)
            df_diffs.append(voxel_change_detector.compare_voxels())
            
        return {
            'tree_filenames': common_tree_filenames,
            'df_diffs': df_diffs,
            'df_trees_1': df_trees_1,
            'df_trees_2': df_trees_2
            }
    
    
    @classmethod
    def calculate_multiple_stats(cls, df_diffs: list[DataFrame], n_std_threshold_min: float=float('inf')) -> dict[str, list[Any]]:
        """Calculate all the stats from the list of voxel grids DataFrame differences
        Args:
            df_diffs (list[DataFrame]): List of voxel grids DataFrame differences
            n_std_threshold_min (float, optional): Minimum number of standard deviation to consider an outlier from the trees common voxels.

        Returns:
            list_stats (tuple[list[DataFrame])
            list_source_counts (list[DataFrame])
            missing_ratios (list[float])
        """

        stats_list: list[dict[str, Any]] = [cls.calculate_stats_static(df_diff, n_std_threshold_min) for df_diff in df_diffs]
        source_counts: list[DataFrame] = [stats['source_counts'] for stats in stats_list]
        missing_ratios: list[float] = cls.calculate_all_trees_missing_ratios(source_counts)
        
        return {
            'stats_list': stats_list,
            'source_counts': source_counts,
            'missing_ratios': missing_ratios
            }
    
    
    @staticmethod
    def calculate_all_trees_missing_ratios(source_counts_list: list[DataFrame], std_threshold: float=1.5) -> list[float]:
        """For each tree, calculate the missing voxels from tree 1 versus all the voxels in trees 2

        Args:
            source_counts_list (list[DataFrame]): list of voxels counts for all trees
            std_threshold (float, optional): Ratio of standard deviation (std) to compute the minimum number
                of points ratio (tree1/tree2) in the common voxels to consider it as empty and in tree 1 only.
                Defaults to 1.5 (1.5 std)

        Returns:
            list[float]: _description_
        """
        missing_ratios: list[float] = []
        
        for df in source_counts_list:
            missing_from_1: int = df.loc[df['Source'].astype(str) == '1', 'Count'].values[0]
            # new_in_2: int = df.loc[df['Source'] == '2', 'Count'].values[0]
            common: int = df.loc[df['Source'] == '3', 'Count'].values[0]
            
            # Compute missing voxels from tree 1 versus all voxels from tree 2
            missing_ratios.append(missing_from_1 / (missing_from_1 + common))
        
        return missing_ratios
    
    @staticmethod
    def _extract_tree_filenames(root_dir: str, tree_paths: list[str]) -> list[str]:
        """
        Extract the tree filenames (with subdirectory) from the tree paths with respect to the root directory.
        
        Args:
            root_dir (str): The root directory of the trees.
            tree_paths (list[str]): The tree paths to extract the filenames from.
        
        Returns:
            list[str]: The list of tree filenames (with subdirectory) relative to the root directory.
        """
        
        normalized_root_dir: str = os.path.normpath(root_dir)
        tree_filenames: list[str] = []
        for tree_path in tree_paths:
            normalized_tree_path: str = os.path.normpath(tree_path)
            if normalized_root_dir in normalized_tree_path:
                relative_path: str = normalized_tree_path.replace(normalized_root_dir, "").lstrip(os.sep)
                tree_filenames.append(relative_path)
                
        return tree_filenames
    
    @staticmethod
    def find_outliers(df_stats: DataFrame, df_diff: DataFrame, n_std_threshold_min: float) -> dict[str, Any]:
        """Find outliers from the stats and the differences DataFrame using the number of standard deviations.
            The outliers are identified based on the number of points ratio in the common voxels of the two trees.
            The threshold is computed as the mean + n_std_threshold_min * std

        Args:
            stats_dic (dict): Dictionary containing statistics of the voxels
            df_diff DataFrame: DataFrame containing differences between the two trees voxel grids
            n_std_threshold_min (float): Number of standard deviations to use for determining outliers

        Returns:
            dict: Dictionary of outliers containing:
                nb_outliers (int): Number of outliers
                points_ratio_th_log2 (float): Threshold for outliers (log2 based). log2 of points ratio over the threshold are considered as outliers.
                points_ratio_th (float): Threshold for outliers
        """

        mean_log2: float = df_stats[df_stats['index'] == 'mean']['Log2_NbPoints_ratio'].values[0]
        std_log2: float = df_stats[df_stats['index'] == 'std']['Log2_NbPoints_ratio'].values[0]

        points_th_log2: float = mean_log2 + n_std_threshold_min * std_log2
        points_ratio_th: float = 2 ** points_th_log2
        
        # Get outiliers from common voxels having points ratio over the threshold
        df_outliers: DataFrame = df_diff[(df_diff['Source'].astype(str) == '3') & (df_diff['Log2_NbPoints_ratio'] > points_th_log2)]
        
        return {
            'nb_outliers': len(df_outliers),
            'points_ratio_th_log2': points_th_log2,
            'points_ratio_th': points_ratio_th,
            'df_outlier': df_outliers,
            'n_std_threshold_min': n_std_threshold_min
        }

