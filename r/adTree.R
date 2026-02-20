runADTreeDirectory <- function(in_dir, adtree_exe, out_dir) {

  xyz_raw_list <- list.files(in_dir, pattern="\\.xyz$", full.names=TRUE)

  for(r in 1:length(xyz_raw_list)){

    xyz_file <- normalizePath(xyz_raw_list[r])
    dir_name <- tools::file_path_sans_ext(basename(xyz_file))
    adtree_out_dir <- file.path(out_dir, dir_name)

    dir.create(adtree_out_dir, recursive = TRUE, showWarnings = FALSE)

    res <- system2(
      adtree_exe,
      args = c(shQuote(xyz_file), shQuote(adtree_out_dir), "-s"),
      stdout = TRUE,
      stderr = TRUE
    )

  }

}

### Run this with your own inputs. 

in_dir <- "D:/Chris/Hydro/Karl/translation/raw"
out_dir <- "D:/Chris/Hydro/Karl/translation/ADTree"

adtree_exe <- normalizePath(
  "D:/Chris/Hydro/AdTree/AdTree-v1.1.2_for_Windows/AdTree-v1.1.2_for_Windows/AdTree.exe"
)

runADTreeDirectory(in_dir, adtree_exe, out_dir)
