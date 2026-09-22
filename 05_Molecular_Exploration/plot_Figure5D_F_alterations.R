invisible(Sys.setlocale("LC_CTYPE", ".UTF-8"))
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
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

SIGNIFICANCE_COLUMN <- "p_value"
ALTERATIONS_ALL <- c("ETS", "FOXA1", "SPOP", "RB1_del", "HDAC2_del", "ROS1_del",
                    "CHD1_del", "MYC_amp", "CDKN1B_del", "MYCBP2_del",
                    "MMR_PMS2", "MMR_MLH3", "MMR_MSH2", "MMR_MSH6")
ALTERATIONS_COHORT <- c("FOXA1", "SPOP", "CDKN1B_del", "CHD1_del", "HDAC2_del",
                       "MYC_amp", "MYCBP2_del", "RB1_del", "ROS1_del", "ETS",
                       "MMR_ANY", "MMR_MLH3", "MMR_MSH2", "MMR_MSH6", "MMR_PMS2",
                       "DDR_ANY", "DDR_CDK12", "DDR_CHEK1", "DDR_CHEK2", "DDR_ERCC3",
                       "DDR_PTEN", "DDR_ARID1A", "DDR_HDAC2")
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

alteration_counts <- list()
alteration_tests <- list()
make_alterations <- function(cohort, genes, tag) {
  d <- cohort_rows(cohort)
  counts <- list()
  tests <- list()
  for (definition in c("Prediction", "Truth")) for (gene in genes) {
    value <- suppressWarnings(as.numeric(d[[gene]]))
    if (any(!is.na(value) & !value %in% 0:1)) stop("Non-binary alteration column: ", gene)
    group <- factor(d[[definition]], levels = c("HRD", "HRP"))
    for (g in levels(group)) {
      use <- !is.na(value) & group == g
      counts[[length(counts) + 1L]] <- data.frame(Cohort = cohort, Definition = definition, Gene = gene,
        Group = g, Altered = sum(value[use]), Available = sum(use),
        Percent = if (sum(use)) 100 * mean(value[use]) else NA_real_)
    }
    tab <- table(factor(value, levels = 0:1), group)
    p <- if (all(colSums(tab) > 0)) fisher.test(tab)$p.value else NA_real_
    tests[[length(tests) + 1L]] <- data.frame(Cohort = cohort, Definition = definition, Gene = gene,
                                            Test = "Two-sided Fisher exact", p_value = p)
  }
  counts <- bind_rows(counts)
  tests <- bind_rows(tests) %>% group_by(Definition) %>% mutate(q_value = p.adjust(p_value, "BH")) %>% ungroup()
  alteration_counts[[cohort]] <<- counts
  alteration_tests[[cohort]] <<- tests
  counts <- counts %>% mutate(Gene = factor(Gene, levels = genes), Group = factor(Group, levels = c("HRD", "HRP")),
                              SignedPercent = ifelse(Definition == "Prediction", Percent, -Percent))
  marks <- counts %>% group_by(Definition, Gene) %>% summarise(Top = max(Percent), .groups = "drop") %>%
    mutate(Gene = as.character(Gene)) %>% left_join(tests, by = c("Definition", "Gene")) %>%
    mutate(Gene = factor(Gene, levels = genes), Label = ifelse(.data[[SIGNIFICANCE_COLUMN]] < .05, "*", ""),
           Y = ifelse(Definition == "Prediction", Top + 6, -Top - 6))
  limit <- if (cohort == "Overall") 60 else 110
  ggplot(counts, aes(Gene, SignedPercent, fill = Group)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.72) +
    geom_hline(yintercept = 0, linewidth = 0.35) +
    geom_text(data = marks, aes(x = Gene, y = Y, label = Label), inherit.aes = FALSE, size = 3) +
    scale_fill_manual(values = GROUP_COLORS, drop = FALSE) +
    scale_y_continuous(limits = c(-limit, limit), breaks = seq(-limit + 10, limit - 10, by = if (limit > 60) 50 else 25),
                       labels = function(x) abs(x), expand = expansion(mult = c(0, 0))) +
    scale_x_discrete(labels = function(x) if (cohort == "Overall") sub("^MMR_", "", x) else x,
                     expand = expansion(add = c(0.5, 1.5))) +
    annotate("text", x = length(genes) + 1.15, y = limit * .55, label = "Prediction", angle = 90, size = 2.5) +
    annotate("text", x = length(genes) + 1.15, y = -limit * .55, label = "Truth", angle = 90, size = 2.5) +
    labs(x = NULL, y = "Alteration frequency (%)", tag = tag) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1, size = if (length(genes) > 14) 5.8 else 7),
          legend.position = "inside", legend.position.inside = c(.75, .94),
          legend.direction = "vertical", legend.background = element_blank()) +
    {if (cohort == "Overall") geom_vline(xintercept = 10.5, linetype = "dashed", linewidth = .35)}
}

plots <- list(D = make_alterations("Overall", ALTERATIONS_ALL, "D"),
              E = make_alterations("TCGA-PRAD", ALTERATIONS_COHORT, "E"),
              F = make_alterations("CPGEA", ALTERATIONS_COHORT, "F"))
output_files <- c(D = "Figure5D_Alterations_Overall.pdf",
                  E = "Figure5E_Alterations_TCGA_PRAD.pdf",
                  F = "Figure5F_Alterations_CPGEA.pdf")
for (panel in names(plots)) {
  ggsave(file.path(topic, output_files[[panel]]), plots[[panel]],
         width = 8, height = 5, device = cairo_pdf, bg = "white")
}
alteration_count_table <- bind_rows(alteration_counts)
alteration_test_table <- bind_rows(alteration_tests)
