library("aRchi")

# Extract the tree index from the file name
# Put the adtree .obj file
filename = ""

# import the obj file output from adTree
mesh = Morpho::obj2mesh(filename)

# plot the original mesh
rgl::open3d()
rgl::shade3d(mesh,col="black",add=T)

qsm = lidUrb::adtree2qsm(mesh, min_diameter = 0.02) # minimum 2cm 

print("building archi")

qsm$model = "yolo"# goes into the else if condition

archi_out = build_aRchi(qsm, keep_original = FALSE) # creates an archi object

archi_out_paths = Make_Path(archi_out) # extracts the branches as paths

qsm_out = archi_out_paths@QSM # extracts the qsm object from the archi object
tree_data_out = qsm[3:5] # extracts tree-level data from the qsm object
paths_out = archi_out_paths@Paths # extracts the paths data frame from the archi object

# Generate output file names based on the input file name
# output_dir <- paste0("D:/Karl/hydro/dataset/Working/single_trees/", part, "/2022/adtree_qsm3")
# qsm_output_file <- file.path(output_dir, paste0(tree_index, "_qsm_out.csv"))
# tree_data_output_file <- file.path(output_dir, paste0(tree_index, "_tree_data_out.csv"))
# paths_out_output_file <- file.path(output_dir, paste0(tree_index, "_paths_out.csv"))

# Save the output files
#write.table(qsm_out, file = qsm_output_file, row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)
#write.table(tree_data_out, file = tree_data_output_file, row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)
#write.table(paths_out, file = paths_out_output_file, row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

print("done")
