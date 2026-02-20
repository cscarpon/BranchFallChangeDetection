# rTwig batch correction with AdTree (optional) + per-file timeout skip
# ADJUSTMENT: unknown/NA species no longer skips. It uses a fallback twig radius and continues.

suppressPackageStartupMessages({
  library(rTwig)
  library(R6)
  library(R.utils)  # withTimeout
})

rTwigBatchCorrector <- R6::R6Class(
  classname = "rTwigBatchCorrector",
  public = list(
    # ---- AdTree inputs (optional stage) ----
    xyz_root_dir = NULL,
    adtree_exe = NULL,
    adtree_out_dir = NULL,
    run_adtree = FALSE,
    adtree_args = NULL,

    # ---- rTwig inputs ----
    in_dir = NULL,
    out_dir = NULL,
    save_obj = TRUE,
    start_i = 1L,
    pattern = "_branches\\.obj$",

    recursive = TRUE,
    verbose = TRUE,

    # ---- timeout controls ----
    timeout_sec = 600,
    timeout_on_import = FALSE,

    # ---- fallback when species is unknown ----
    unknown_species_radius_mm = NULL,  # if NULL, uses median(rTwig::twigs$radius_mm)

    # ---- run state ----
    obj_files = NULL,
    run_summary = NULL,
    not_processed = NULL,
    adtree_failures = NULL,
    adtree_summary = NULL,
    timeouts = NULL,

    initialize = function(
    # AdTree (optional)
      xyz_root_dir = NULL,
      adtree_exe = NULL,
      adtree_out_dir = NULL,
      run_adtree = FALSE,
      adtree_args = c("-s"),

      # rTwig
      in_dir,
      out_dir,
      save_obj = TRUE,
      start_i = 1L,
      pattern = "_branches\\.obj$",
      recursive = TRUE,
      verbose = TRUE,

      # timeout
      timeout_sec = 600,
      timeout_on_import = FALSE,

      # fallback
      unknown_species_radius_mm = NULL
    ) {
      self$xyz_root_dir <- if (!is.null(xyz_root_dir)) normalizePath(xyz_root_dir, winslash = "/", mustWork = FALSE) else NULL
      self$adtree_exe <- if (!is.null(adtree_exe)) normalizePath(adtree_exe, winslash = "/", mustWork = FALSE) else NULL
      self$adtree_out_dir <- if (!is.null(adtree_out_dir)) normalizePath(adtree_out_dir, winslash = "/", mustWork = FALSE) else NULL
      self$run_adtree <- isTRUE(run_adtree)
      self$adtree_args <- adtree_args

      self$in_dir <- normalizePath(in_dir, winslash = "/", mustWork = FALSE)
      self$out_dir <- normalizePath(out_dir, winslash = "/", mustWork = FALSE)
      self$save_obj <- isTRUE(save_obj)
      self$start_i <- as.integer(start_i)
      self$pattern <- pattern
      self$recursive <- isTRUE(recursive)
      self$verbose <- isTRUE(verbose)

      self$timeout_sec <- as.numeric(timeout_sec)
      self$timeout_on_import <- isTRUE(timeout_on_import)

      self$unknown_species_radius_mm <- if (is.null(unknown_species_radius_mm)) NULL else as.numeric(unknown_species_radius_mm)

      dir.create(self$out_dir, recursive = TRUE, showWarnings = FALSE)

      self$run_summary <- list()
      self$not_processed <- list()
      self$adtree_failures <- list()
      self$adtree_summary <- list()
      self$timeouts <- list()
    },

    # --------------------------
    # helpers
    # --------------------------
    infer_part_dir = function(path) {
      p <- normalizePath(path, winslash = "/", mustWork = FALSE)
      parent <- basename(dirname(p))
      parent2 <- basename(dirname(dirname(p)))
      if (grepl("^part", parent, ignore.case = TRUE)) parent else parent2
    },

    log_status = function(status_file, msg) {
      cat(
        paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", msg, "\n"),
        file = status_file,
        append = TRUE
      )
    },

    add_not_processed = function(file, reason) {
      self$not_processed[[length(self$not_processed) + 1]] <- data.frame(
        file = file, reason = reason, stringsAsFactors = FALSE
      )
    },

    add_timeout = function(file, stage, seconds) {
      self$timeouts[[length(self$timeouts) + 1]] <- data.frame(
        file = file,
        stage = stage,
        timeout_sec = seconds,
        when = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        stringsAsFactors = FALSE
      )
    },

    add_adtree_failure = function(file, reason) {
      self$adtree_failures[[length(self$adtree_failures) + 1]] <- data.frame(
        file = file, reason = reason, stringsAsFactors = FALSE
      )
    },

    parse_species_token = function(obj_path) {
      folder <- basename(dirname(obj_path))
      sp_token <- sub("^.*[[:space:]]([^[:space:]]+)$", "\\1", folder)

      if (is.na(sp_token) || sp_token %in% c("", "NA", "Unknown")) {
        return(list(species_token = NA_character_, genus = NA_character_,
                    scientific_name = NA_character_, genus_spp = NA_character_))
      }

      parts <- strsplit(sp_token, "-", fixed = TRUE)[[1]]
      parts <- parts[nzchar(parts)]
      genus <- parts[1]
      species <- if (length(parts) >= 2) parts[2] else NA_character_

      sci <- if (!is.na(species)) paste(tolower(genus), tolower(species)) else NA_character_
      if (!is.na(sci)) sci <- paste0(toupper(substr(sci, 1, 1)), substr(sci, 2, nchar(sci)))

      genus_spp <- if (!is.na(genus)) {
        g <- paste0(toupper(substr(tolower(genus), 1, 1)), substr(tolower(genus), 2, nchar(genus)))
        paste0(g, " spp.")
      } else NA_character_

      list(species_token = sp_token, genus = genus, scientific_name = sci, genus_spp = genus_spp)
    },

    sanitize_species_for_file = function(x) {
      x <- trimws(as.character(x))
      x <- gsub('[\\\\/:*?"<>|]', "-", x)
      x <- gsub("[[:space:]]+", " ", x)
      x
    },

    lookup_twig_radius_mm = function(scientific_name, genus_spp) {
      tw <- rTwig::twigs
      r1 <- tw$radius_mm[match(scientific_name, tw$scientific_name)]
      if (!is.na(r1)) return(list(radius_mm = r1, match_type = "species"))
      r2 <- tw$radius_mm[match(genus_spp, tw$scientific_name)]
      if (!is.na(r2)) return(list(radius_mm = r2, match_type = "genus_spp"))
      list(radius_mm = median(tw$radius_mm, na.rm = TRUE), match_type = "median_fallback")
    },

    angle_from_vertical_deg = function(axis_z) {
      az <- pmin(1, pmax(-1, abs(axis_z)))
      acos(az) * 180 / pi
    },

    # --------------------------
    # AdTree stage (optional)
    # --------------------------
    run_adtree_stage = function() {
      if (!isTRUE(self$run_adtree)) return(invisible(NULL))

      if (is.null(self$xyz_root_dir) || !dir.exists(self$xyz_root_dir)) {
        stop("run_adtree=TRUE but xyz_root_dir is missing or invalid: ", self$xyz_root_dir)
      }
      if (is.null(self$adtree_out_dir)) {
        stop("run_adtree=TRUE but adtree_out_dir is NULL")
      }
      if (is.null(self$adtree_exe) || !file.exists(self$adtree_exe)) {
        stop("run_adtree=TRUE but adtree_exe is missing/invalid: ", self$adtree_exe)
      }

      dir.create(self$adtree_out_dir, recursive = TRUE, showWarnings = FALSE)

      xyz_files <- sort(list.files(self$xyz_root_dir, pattern = "\\.xyz$", full.names = TRUE, recursive = TRUE))
      if (!length(xyz_files)) stop("No .xyz files found under xyz_root_dir: ", self$xyz_root_dir)

      n <- length(xyz_files)
      for (i in seq_len(n)) {
        xyz_file <- normalizePath(xyz_files[i], winslash = "/", mustWork = FALSE)
        part_dir <- self$infer_part_dir(xyz_file)
        tree_folder <- tools::file_path_sans_ext(basename(xyz_file))
        out_tree <- file.path(self$adtree_out_dir, part_dir, tree_folder)
        dir.create(out_tree, recursive = TRUE, showWarnings = FALSE)

        if (self$verbose) {
          cat("\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              " | ADTree", i, "/", n, " | ", xyz_file, "\n", sep = "")
          cat("  out:", out_tree, "\n")
          flush.console()
        }

        res <- tryCatch({
          system2(
            self$adtree_exe,
            args = c(shQuote(xyz_file), shQuote(out_tree), self$adtree_args),
            stdout = TRUE,
            stderr = TRUE
          )
        }, error = function(e) e)

        if (inherits(res, "error")) {
          self$add_adtree_failure(xyz_file, paste0("system2_error: ", conditionMessage(res)))
          next
        }

        out_files <- list.files(out_tree, full.names = TRUE)
        if (!length(out_files)) self$add_adtree_failure(xyz_file, "adtree_no_outputs_written")

        self$adtree_summary[[length(self$adtree_summary) + 1]] <- data.frame(
          part = part_dir,
          xyz_file = xyz_file,
          tree_folder = tree_folder,
          adtree_out_dir = out_tree,
          n_out_files = length(out_files),
          stringsAsFactors = FALSE
        )
      }

      adtree_summary_df <- if (length(self$adtree_summary)) do.call(rbind, self$adtree_summary) else data.frame()
      adtree_failures_df <- if (length(self$adtree_failures)) do.call(rbind, self$adtree_failures) else data.frame()

      write.csv(adtree_summary_df, file.path(self$adtree_out_dir, "adtree_run_summary.csv"), row.names = FALSE)
      write.csv(adtree_failures_df, file.path(self$adtree_out_dir, "adtree_failures.csv"), row.names = FALSE)

      invisible(list(run_summary = adtree_summary_df, failures = adtree_failures_df))
    },

    # --------------------------
    # rTwig per-file
    # --------------------------
    process_one = function(f, save_obj_override = NULL) {
      tree_folder <- basename(dirname(f))
      part_dir <- basename(dirname(dirname(normalizePath(f, winslash = "/", mustWork = FALSE))))
      tree_out <- file.path(self$out_dir, part_dir, tree_folder)
      dir.create(tree_out, recursive = TRUE, showWarnings = FALSE)

      tree_base <- tools::file_path_sans_ext(basename(f))

      status_file <- file.path(tree_out, "status.txt")
      cat("", file = status_file)

      self$log_status(status_file, paste0("start: ", f))
      self$log_status(status_file, paste0("out:   ", tree_out))
      self$log_status(status_file, paste0("timeout_sec: ", self$timeout_sec))

      # ---- species parse (DO NOT SKIP) ----
      sp <- self$parse_species_token(f)

      # Fallback twig radius if unknown species
      if (is.na(sp$species_token)) {
        fallback_mm <- if (is.null(self$unknown_species_radius_mm)) {
          median(rTwig::twigs$radius_mm, na.rm = TRUE)
        } else {
          self$unknown_species_radius_mm
        }
        twig <- list(radius_mm = fallback_mm, match_type = "unknown_species_fallback")
        self$log_status(status_file, paste0("WARN: unknown species token -> twig_radius_mm fallback = ", fallback_mm))
      } else {
        twig <- self$lookup_twig_radius_mm(sp$scientific_name, sp$genus_spp)
      }

      # ---- import_adqsm ----
      self$log_status(status_file, "STEP: import_adqsm()")
      qsm <- NULL
      if (isTRUE(self$timeout_on_import)) {
        qsm <- tryCatch({
          R.utils::withTimeout(
            expr = rTwig::import_adqsm(f, "adtree"),
            timeout = self$timeout_sec,
            onTimeout = "error"
          )
        }, TimeoutException = function(e) e, error = function(e) e)

        if (inherits(qsm, "TimeoutException")) {
          self$add_timeout(f, "import_adqsm", self$timeout_sec)
          self$add_not_processed(f, "timeout_import_adqsm")
          self$log_status(status_file, "SKIP: timeout_import_adqsm")
          return(NULL)
        }
      } else {
        qsm <- tryCatch(rTwig::import_adqsm(f, "adtree"), error = function(e) e)
      }

      if (inherits(qsm, "error")) {
        self$add_not_processed(f, paste0("import_error: ", conditionMessage(qsm)))
        self$log_status(status_file, paste0("FAILED: import_error: ", conditionMessage(qsm)))
        return(NULL)
      }

      # ---- correct_radii (timeout protected) ----
      self$log_status(status_file, "STEP: correct_radii() [timeout-protected]")
      cyl <- tryCatch({
        R.utils::withTimeout(
          expr = rTwig::correct_radii(qsm, twig_radius = twig$radius_mm),
          timeout = self$timeout_sec,
          onTimeout = "error"
        )
      }, TimeoutException = function(e) e, error = function(e) e)

      if (inherits(cyl, "TimeoutException")) {
        self$add_timeout(f, "correct_radii", self$timeout_sec)
        self$add_not_processed(f, "timeout_correct_radii")
        self$log_status(status_file, "SKIP: timeout_correct_radii")
        return(NULL)
      }

      if (inherits(cyl, "error") || !is.data.frame(cyl) || nrow(cyl) == 0) {
        self$add_not_processed(f, paste0("correct_radii_error: ", conditionMessage(cyl)))
        self$log_status(status_file, paste0("FAILED: correct_radii_error: ", conditionMessage(cyl)))
        return(NULL)
      }

      # update_cylinders intentionally removed

      cyl$volume_m3 <- pi * (cyl$radius^2) * cyl$length
      cyl$surface_area_m2 <- 2 * pi * cyl$radius * cyl$length
      cyl$angle_from_vertical_deg <- self$angle_from_vertical_deg(cyl$axis_z)

      branch_summary <- aggregate(
        cbind(branch_length_m = cyl$length,
              volume_m3 = cyl$volume_m3,
              surface_area_m2 = cyl$surface_area_m2) ~ branch,
        data = cyl,
        FUN = function(x) sum(x, na.rm = TRUE)
      )

      bo <- tapply(cyl$branch_order, cyl$branch, function(x) {
        x <- x[!is.na(x)]
        if (!length(x)) return(NA_integer_)
        as.integer(stats::median(x))
      })

      ang <- tapply(seq_len(nrow(cyl)), cyl$branch, function(idx) {
        w <- cyl$length[idx]
        a <- cyl$angle_from_vertical_deg[idx]
        ok <- is.finite(w) & is.finite(a)
        if (!any(ok)) return(NA_real_)
        sum(a[ok] * w[ok]) / sum(w[ok])
      })

      branch_summary$branch_order <- unname(bo[as.character(branch_summary$branch)])
      branch_summary$mean_angle_from_vertical_deg <- unname(ang[as.character(branch_summary$branch)])

      self$log_status(status_file, "STEP: summarise_qsm()")
      qsm_sum <- tryCatch(rTwig::summarise_qsm(cyl), error = function(e) e)

      cyl_csv <- file.path(tree_out, paste0(tree_base, "_cylinders_corrected.csv"))
      br_csv  <- file.path(tree_out, paste0(tree_base, "_branches_summary.csv"))
      sum_csv <- file.path(tree_out, paste0(tree_base, "_qsm_summary_by_order.csv"))
      man_csv <- file.path(tree_out, paste0(tree_base, "_manifest.csv"))

      self$log_status(status_file, "STEP: write.csv()")
      ok_write <- tryCatch({
        write.csv(cyl, cyl_csv, row.names = FALSE)
        write.csv(branch_summary, br_csv, row.names = FALSE)
        if (!inherits(qsm_sum, "error") && is.list(qsm_sum) && length(qsm_sum) >= 1) {
          write.csv(as.data.frame(qsm_sum[[1]]), sum_csv, row.names = FALSE)
        }
        TRUE
      }, error = function(e) {
        self$add_not_processed(f, paste0("write_error: ", conditionMessage(e)))
        self$log_status(status_file, paste0("FAILED: write_error: ", conditionMessage(e)))
        FALSE
      })
      if (!ok_write) return(NULL)

      save_obj_now <- if (is.null(save_obj_override)) self$save_obj else isTRUE(save_obj_override)

      ok_mesh <- NA
      corrected_obj <- NA_character_
      if (save_obj_now) {
        self$log_status(status_file, "STEP: export_mesh()")
        ok_mesh <- tryCatch({
          rTwig::export_mesh(
            cyl,
            filename = file.path(tree_out, paste0(tree_base, "_corrected")),
            format = "obj"
          )
          TRUE
        }, error = function(e) {
          self$add_not_processed(f, paste0("export_mesh_error: ", conditionMessage(e)))
          self$log_status(status_file, paste0("FAILED: export_mesh_error: ", conditionMessage(e)))
          FALSE
        })
        corrected_obj <- file.path(tree_out, paste0(tree_base, "_corrected.obj"))
      }

      manifest <- data.frame(
        part = part_dir,
        tree_folder = tree_folder,
        input_obj = f,
        output_dir = tree_out,
        corrected_obj = if (save_obj_now) corrected_obj else NA_character_,
        cylinders_csv = cyl_csv,
        branches_csv = br_csv,
        summary_csv = if (file.exists(sum_csv)) sum_csv else NA_character_,
        species_token = if (is.na(sp$species_token)) "UNKNOWN" else self$sanitize_species_for_file(sp$species_token),
        scientific_name = sp$scientific_name,
        twig_radius_mm = twig$radius_mm,
        twig_match = twig$match_type,
        exported_mesh = if (save_obj_now) isTRUE(ok_mesh) else NA,
        stringsAsFactors = FALSE
      )
      write.csv(manifest, man_csv, row.names = FALSE)

      self$log_status(status_file, "DONE")
      manifest
    },

    # --------------------------
    # runner
    # --------------------------
    run = function(save_obj = NULL, timeout_sec = NULL) {
      if (!is.null(save_obj)) self$save_obj <- isTRUE(save_obj)
      if (!is.null(timeout_sec)) self$timeout_sec <- as.numeric(timeout_sec)

      adtree_res <- NULL
      if (isTRUE(self$run_adtree)) {
        adtree_res <- self$run_adtree_stage()
        self$in_dir <- self$adtree_out_dir
      }

      self$obj_files <- sort(list.files(
        self$in_dir,
        pattern = self$pattern,
        full.names = TRUE,
        recursive = self$recursive
      ))
      if (!length(self$obj_files)) {
        stop("No matching OBJ files found using pattern: ", self$pattern, "\nIn dir: ", self$in_dir)
      }

      n <- length(self$obj_files)
      if (self$start_i < 1L) self$start_i <- 1L
      if (self$start_i > n) stop("start_i is > number of files (", n, ")")

      for (i in self$start_i:n) {
        f <- self$obj_files[i]
        if (self$verbose) {
          cat("\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              " | i =", i, "/", n,
              " | timeout_sec =", self$timeout_sec,
              " | save_obj =", self$save_obj,
              " | ", f, "\n", sep = "")
          flush.console()
        }

        man <- self$process_one(f, save_obj_override = self$save_obj)
        if (!is.null(man)) self$run_summary[[length(self$run_summary) + 1]] <- man
        flush.console()
      }

      run_summary_df <- if (length(self$run_summary)) do.call(rbind, self$run_summary) else data.frame()
      not_processed_df <- if (length(self$not_processed)) do.call(rbind, self$not_processed) else data.frame()
      timeouts_df <- if (length(self$timeouts)) do.call(rbind, self$timeouts) else data.frame()

      write.csv(run_summary_df, file.path(self$out_dir, "rtwig_run_summary.csv"), row.names = FALSE)
      write.csv(not_processed_df, file.path(self$out_dir, "rtwig_not_processed.csv"), row.names = FALSE)
      write.csv(timeouts_df, file.path(self$out_dir, "rtwig_timeouts.csv"), row.names = FALSE)

      if (self$verbose) {
        cat("\nProcessed    :", nrow(run_summary_df), "\n")
        cat("Not processed:", nrow(not_processed_df), "\n")
        cat("Timeouts     :", nrow(timeouts_df), "\n")
      }

      invisible(list(
        adtree = adtree_res,
        run_summary = run_summary_df,
        not_processed = not_processed_df,
        timeouts = timeouts_df
      ))
    }
  )
)

# -------------------------
# Example usage
# -------------------------
bc <- rTwigBatchCorrector$new(
  in_dir  = "D:/Chris/Hydro/Karl/translation/ADTree", ### this is the folder with the .obj files to process with rTwig.
  out_dir = "E:/Hydro/rTwig", ## this is where you want to save the new rTwig outputs (csv summaries + corrected .obj if save_obj=TRUE)
  save_obj = FALSE,
  start_i = 1,
  timeout_sec = 600,
  timeout_on_import = FALSE,
  unknown_species_radius_mm = NULL  # NULL = median fallback
)

res <- bc$run(save_obj = FALSE, timeout_sec = 600)