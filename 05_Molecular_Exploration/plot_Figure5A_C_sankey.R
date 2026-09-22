invisible(Sys.setlocale("LC_CTYPE", ".UTF-8"))
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggalluvial)
})

PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")
topic <- file.path(PROJECT_ROOT, "05_Molecular_Exploration")
input_dir <- file.path(DATA_ROOT, "05_Molecular_Exploration")
PREDICTION_CUTOFF <- 0.395

GROUP_COLORS <- c(HRD = "#98C4E9", HRP = "#F7C5C8")

master <- as.data.frame(read_excel(file.path(input_dir, "Clinical_Molecular_Pathology_Master_Source.xlsx"),
                                  sheet = "CPGEA-TCGA"))

master <- master[master$Cohort %in% c("CPGEA", "TCGA-PRAD"), ]
master$ID <- trimws(as.character(master$UNIQUEID))
master$Score <- suppressWarnings(as.numeric(master$Path_lightGBM))
master$HRR_ANY <- as.numeric(master$HRR_ANY)
stopifnot(!anyDuplicated(master$ID), all(master$HRR_ANY %in% 0:1))
patients <- master[!is.na(master$Score), ]
stopifnot(all(is.finite(patients$Score)),
          all(patients$Score >= 0 & patients$Score <= 1))
patients$Prediction <- ifelse(patients$Score >= PREDICTION_CUTOFF, "HRD", "HRP")
patients$Truth <- ifelse(patients$HRR_ANY == 1, "HRD", "HRP")
patients$Subtype <- ifelse(patients$Epithelial %in% c("Basal", "LumA", "LumB"), patients$Epithelial, NA_character_)

saved <- bind_rows(lapply(c("train", "test"), function(split) {
  x <- read.csv(file.path(DATA_ROOT, "07_Model_Scores", "Pathology", paste0("Path_LightGBM_", split, ".csv")), check.names = FALSE)
  data.frame(ID = as.character(x$ID), score = x[["HRR_ANY-1"]])
}))
stopifnot(!anyDuplicated(saved$ID), setequal(saved$ID, patients$ID),
          all(abs(patients$Score - saved$score[match(patients$ID, saved$ID)]) < 1e-8))

cohort_rows <- function(cohort) if (cohort == "Overall") patients else patients[patients$Cohort == cohort, ]
base_theme <- theme_classic(base_size = 9, base_family = "Arial") +
  theme(plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8), plot.tag = element_text(size = 15, face = "bold"),
        axis.text = element_text(color = "black"), legend.title = element_blank(),
        legend.key.size = grid::unit(3, "mm"), plot.margin = margin(7, 9, 7, 7),
        panel.border = element_rect(fill = NA, color = "black", linewidth = .4))
theme_set(base_theme)

sankey_tables <- list()
make_sankey <- function(cohort, tag) {
  d <- cohort_rows(cohort)
  d <- d[!is.na(d$Subtype), ]
  paths <- d %>% count(Prediction, Subtype, Truth, name = "N")
  paths$Cohort <- cohort
  sankey_tables[[cohort]] <<- paths
  paths <- paths %>% transmute(Predicted = paste0("pre", Prediction),
                              Subtype = factor(Subtype, levels = c("Basal", "LumA", "LumB")),
                              Actual = paste0("tru", Truth), N)
  lodes <- to_lodes_form(paths, axes = 1:3, key = "Stage", value = "Node", id = "Flow")
  node_colors <- c(preHRD = GROUP_COLORS[["HRD"]], preHRP = GROUP_COLORS[["HRP"]],
                   Basal = "#8FC9AD", LumA = "#F89335", LumB = "#F39463",
                   truHRD = "#58C1E9", truHRP = "#E98BB6")
  ggplot(lodes, aes(x = Stage, y = N, stratum = Node, alluvium = Flow)) +
    geom_flow(aes(fill = Node), width = 0.42, alpha = 0.43, color = "white", linewidth = 0.12,
              decreasing = NA, reverse = TRUE) +
    geom_stratum(aes(fill = Node), width = 0.42, color = "white", linewidth = 0.35,
                 decreasing = NA, reverse = TRUE) +
    geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 2.65,
              decreasing = NA, reverse = TRUE) +
    scale_fill_manual(values = node_colors) + scale_x_discrete(expand = expansion(add = .22)) +
    scale_y_continuous(expand = expansion(mult = 0)) + labs(tag = tag) +
    theme_void(base_size = 9, base_family = "Arial") +
    theme(legend.position = "none", plot.title = element_text(face = "bold", size = 10),
          plot.subtitle = element_text(size = 8), plot.tag = element_text(face = "bold", size = 15),
          plot.margin = margin(7, 9, 7, 7))
}

plots <- list(A = make_sankey("Overall", "A"),
              B = make_sankey("TCGA-PRAD", "B"),
              C = make_sankey("CPGEA", "C"))
output_files <- c(A = "Figure5A_Sankey_Overall.pdf",
                  B = "Figure5B_Sankey_TCGA_PRAD.pdf",
                  C = "Figure5C_Sankey_CPGEA.pdf")
for (panel in names(plots)) {
  ggsave(file.path(topic, output_files[[panel]]), plots[[panel]],
         width = 6, height = 4, device = cairo_pdf, bg = "white")
}
sankey_paths <- bind_rows(sankey_tables)
