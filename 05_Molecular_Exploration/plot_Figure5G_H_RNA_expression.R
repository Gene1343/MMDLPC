invisible(Sys.setlocale("LC_CTYPE", ".UTF-8"))
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
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

RNA_INPUT_TYPE <- "counts"
COUNT_FILES <- c(CPGEA = "CPGEA_counts.xlsx", `TCGA-PRAD` = "TCGA_counts.xlsx")
COUNT_PATHS <- file.path(input_dir, COUNT_FILES)

missing_inputs <- COUNT_PATHS[!file.exists(COUNT_PATHS)]
if (length(missing_inputs)) stop("Raw count inputs are missing:\n", paste(missing_inputs, collapse = "\n"),
                                 "\nSupply both count matrices before running Figure 5G-H.")
if (!requireNamespace("DESeq2", quietly = TRUE)) stop("Install DESeq2 for count normalization")
SIGNIFICANCE_COLUMN <- "p_value"
RNA_GENES <- c("ATM", "BARD1", "BRCA1", "BRCA2", "BRIP1", "CDK12", "CHEK1",
               "CHEK2", "FANCL", "PALB2", "RAD51B", "RAD51C", "RAD51D", "RAD54L")
RNA_COLORS <- c(HRP = "#58B2D5", HRD = "#EE8988")

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

read_count_cohort <- function(path, cohort, mapping) {
  if (!file.exists(path)) stop("Raw count input is missing: ", path)
  x <- as.data.frame(read_excel(path, .name_repair = "minimal"))
  if (anyDuplicated(names(x))) stop(cohort, ": duplicated input column names")
  gene_col <- if ("gene_name" %in% names(x)) "gene_name" else names(x)[1]
  genes <- trimws(as.character(x[[gene_col]]))
  annotation_cols <- unique(c(gene_col, intersect(c("gene_id", "Geneid", "Length"), names(x))))
  sample_cols <- setdiff(names(x), annotation_cols)
  if (cohort == "CPGEA") {
    ids <- mapping$UNIQUEID[match(sub("_WTS$", "", sample_cols), mapping$Sample_ID)]
    ids[sample_cols %in% patients$ID] <- sample_cols[sample_cols %in% patients$ID]
  } else {
    ids <- substr(sample_cols, 1, 12)

    ids[!sample_cols %in% patients$ID & !grepl("^TCGA-[^-]+-[^-]+-01", sample_cols)] <- NA_character_
  }
  order_samples <- order(sample_cols)
  eligible <- patients$ID[patients$Cohort == cohort]
  keep <- order_samples[!is.na(ids[order_samples]) & ids[order_samples] %in% eligible]
  keep <- keep[!duplicated(ids[keep])]
  if (!length(keep)) stop(cohort, ": count matrix has no eligible patient IDs")
  selection <- data.frame(Cohort = cohort, Sample_ID = sample_cols, ID = ids,
    Selected = seq_along(sample_cols) %in% keep,
    Reason = ifelse(is.na(ids), "Not a mapped tumor sample",
             ifelse(!ids %in% eligible, "No eligible pathology score",
             ifelse(seq_along(sample_cols) %in% keep, "Retained", "Duplicate patient; first sample name retained"))))

  mat <- as.matrix(data.frame(lapply(x[sample_cols], as.numeric), check.names = FALSE))
  if (any(!is.finite(mat)) || any(mat < 0) || any(abs(mat - round(mat)) > 1e-8)) {
    stop(cohort, ": counts must be finite nonnegative integers; FPKM/TPM and rounded expression are not valid inputs.")
  }
  if (any(mat > .Machine$integer.max)) stop(cohort, ": count exceeds the supported integer range")
  mat <- mat[, keep, drop = FALSE]
  if (nrow(mat) <= length(RNA_GENES)) stop("Supply the full gene count matrix for size-factor estimation, not only the 14 genes.")
  if (any(colSums(mat) == 0)) stop("A count sample has zero library size")
  selected <- match(RNA_GENES, genes)
  if (anyNA(selected) || anyDuplicated(genes[genes %in% RNA_GENES])) stop("Missing or duplicated target gene symbols in count matrix")
  colnames(mat) <- ids[keep]
  rownames(mat) <- paste0(seq_along(genes), ":", genes)
  storage.mode(mat) <- "integer"
  dds <- DESeq2::DESeqDataSetFromMatrix(mat, data.frame(row.names = colnames(mat)), design = ~1)
  dds <- DESeq2::estimateSizeFactors(dds)
  normalized <- DESeq2::counts(dds, normalized = TRUE)[selected, , drop = FALSE]
  result <- as.data.frame(as.table(normalized), stringsAsFactors = FALSE)
  names(result) <- c("Feature", "ID", "Expression")
  result$Gene <- genes[match(result$Feature, rownames(mat))]
  result$Cohort <- cohort
  result$RawCount <- as.numeric(mat[selected, , drop = FALSE])
  result$InputUnit <- "raw count"
  result$ExpressionUnit <- "DESeq2 normalized count"
  result$SizeFactor <- DESeq2::sizeFactors(dds)[result$ID]
  selection$SizeFactor <- result$SizeFactor[match(selection$ID, result$ID)]
  selection$SizeFactor[!selection$Selected] <- NA_real_
  list(expression = result[, c("Cohort", "ID", "Gene", "RawCount", "Expression", "InputUnit", "ExpressionUnit", "SizeFactor")],
       selection = selection)
}

mapping <- as.data.frame(read_excel(file.path(input_dir, "CPGEA_TCGA_sample_mapping_source.xlsx"), sheet = "CPGEA-TCGA"))
mapping <- mapping[mapping$Cohort == "CPGEA" & !is.na(mapping$Sample_ID), ]
mapping$Sample_ID <- trimws(as.character(mapping$Sample_ID))
mapping$UNIQUEID <- trimws(as.character(mapping$UNIQUEID))
stopifnot(!anyDuplicated(mapping$Sample_ID), !anyDuplicated(mapping$UNIQUEID))
rna_inputs <- lapply(names(COUNT_FILES), function(cohort) read_count_cohort(file.path(input_dir, COUNT_FILES[[cohort]]), cohort, mapping))
expression <- bind_rows(lapply(rna_inputs, `[[`, "expression"))
sample_selection <- bind_rows(lapply(rna_inputs, `[[`, "selection"))
input_sources <- data.frame(Cohort = names(COUNT_FILES), File = unname(COUNT_FILES),
                            MD5 = unname(tools::md5sum(COUNT_PATHS)), InputUnit = "raw counts")
expression_axis <- "log2(normalized\ncount + 1)"
stopifnot(!anyDuplicated(expression[c("Cohort", "ID", "Gene")]), all(is.finite(expression$Expression)),
          all(expression$Expression >= 0))
expression <- expression %>% inner_join(patients[c("ID", "Cohort", "Prediction", "Truth")], by = c("ID", "Cohort"))
stopifnot(setequal(unique(expression$Gene), RNA_GENES))
expression$LogExpression <- log2(expression$Expression + 1)

rna_summaries <- list()
rna_tests <- list()
make_expression <- function(cohort, tag) {
  d <- expression[expression$Cohort == cohort, ] %>%
    pivot_longer(c(Prediction, Truth), names_to = "Definition", values_to = "Group")
  summary <- d %>% group_by(Cohort, Definition, Gene, Group) %>%
    summarise(N = n(), Mean = mean(LogExpression), SEM = if (n() > 1) sd(LogExpression)/sqrt(n()) else NA_real_, .groups = "drop")
  tests <- d %>% group_by(Cohort, Definition, Gene) %>% group_modify(~{
    a <- .x$LogExpression[.x$Group == "HRD"]
    b <- .x$LogExpression[.x$Group == "HRP"]
    p <- if (min(length(a), length(b)) >= 2) tryCatch(t.test(a, b, var.equal = FALSE)$p.value, error = function(e) NA_real_) else NA_real_
    data.frame(N_HRD = length(a), N_HRP = length(b), Test = "Two-sided Welch t-test on log2(expression + 1)", p_value = p)
  }) %>% ungroup() %>% group_by(Definition) %>% mutate(q_value = p.adjust(p_value, "BH")) %>% ungroup()
  rna_summaries[[cohort]] <<- summary
  rna_tests[[cohort]] <<- tests
  summary <- summary %>% mutate(Gene = factor(Gene, levels = RNA_GENES), Group = factor(Group, levels = c("HRP", "HRD")),
    Sign = ifelse(Definition == "Prediction", 1, -1), Y = Sign * Mean,
    Lower = pmin(Sign * (Mean - SEM), Sign * (Mean + SEM)), Upper = pmax(Sign * (Mean - SEM), Sign * (Mean + SEM)))
  limit <- ceiling((max(summary$Mean + summary$SEM, na.rm = TRUE) + .6) * 2) / 2
  marks <- summary %>% group_by(Definition, Gene) %>% summarise(Top = max(Mean + SEM), .groups = "drop") %>%
    mutate(Gene = as.character(Gene)) %>% left_join(tests, by = c("Definition", "Gene")) %>%
    mutate(Gene = factor(Gene, levels = RNA_GENES),
           Label = ifelse(is.na(.data[[SIGNIFICANCE_COLUMN]]), "NA", ifelse(.data[[SIGNIFICANCE_COLUMN]] < .05, "*", "ns")),
           Y = ifelse(Definition == "Prediction", Top + .45, -Top - .45))

  panes <- lapply(c("Prediction", "Truth"), function(definition) {
    predicted <- definition == "Prediction"
    limits <- if (predicted) c(0, limit) else c(-limit, 0)
    ggplot(summary[summary$Definition == definition, ], aes(Gene, Y, fill = Group)) +
      geom_col(position = position_dodge(.8), width = .72) +
      geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(.8), width = .15, linewidth = .25) +
      geom_text(data = marks[marks$Definition == definition, ], aes(Gene, Y, label = Label),
                inherit.aes = FALSE, size = 2.1) +
      scale_fill_manual(values = RNA_COLORS, drop = FALSE) +
      scale_y_continuous(limits = limits, breaks = pretty(limits, n = 3), labels = abs,
                         expand = expansion(mult = 0)) +
      scale_x_discrete(expand = expansion(add = c(.5, 1.8))) +
      annotate("text", x = 15.35, y = if (predicted) limit * .45 else -limit * .45,
               label = definition, angle = 90, size = 2.3) +
      labs(x = NULL, y = expression_axis, tag = if (predicted) tag else NULL) +
      theme(axis.text.x = if (predicted) element_text(angle = 50, hjust = 1, size = 6.5) else element_blank(),
            axis.ticks.x = if (predicted) element_line(linewidth = .3) else element_blank(),
            axis.title.y = element_text(size = 7.5), axis.text.y = element_text(size = 7),
            legend.position = "inside", legend.position.inside = c(.96, if (predicted) .96 else .04),
            legend.justification = c(1, if (predicted) 1 else 0), legend.direction = "vertical",
            legend.background = element_blank(), legend.text = element_text(size = 6.5),
            legend.key.size = grid::unit(2, "mm"), legend.margin = margin(0, 0, 0, 0),
            plot.margin = margin(if (predicted) 7 else 0, 9, if (predicted) 2 else 7, 7))
  })
  wrap_plots(panes, ncol = 1, heights = c(1, 1))
}

plots <- list(G = make_expression("CPGEA", "G"),
              H = make_expression("TCGA-PRAD", "H"))
output_files <- c(G = "Figure5G_RNA_CPGEA_Counts.pdf",
                  H = "Figure5H_RNA_TCGA_PRAD_Counts.pdf")
for (panel in names(plots)) {
  ggsave(file.path(topic, output_files[[panel]]), plots[[panel]],
         width = 8, height = 6, device = cairo_pdf, bg = "white")
}
rna_summary_table <- bind_rows(rna_summaries)
rna_test_table <- bind_rows(rna_tests)
