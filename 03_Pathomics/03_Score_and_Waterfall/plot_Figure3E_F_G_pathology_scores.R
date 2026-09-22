
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(pROC))
topic_dir <- file.path(PROJECT_ROOT, "03_Pathomics/03_Score_and_Waterfall")
normalize_id <- function(x) toupper(sub("\\.nii(\\.gz)?$", "", trimws(as.character(x)), ignore.case = TRUE))
labels <- read.csv(file.path(DATA_ROOT, "03_Pathomics", "labels_locked.csv"), check.names = FALSE)
labels$ID <- normalize_id(labels$ID)
stopifnot(!anyDuplicated(labels$ID), all(labels$HRR_ANY %in% c(0, 1)))

read_scores <- function(prefix, split) {
  filename <- paste0(prefix, "_", split, ".csv")
  raw <- read.csv(file.path(file.path(DATA_ROOT, "07_Model_Scores/Pathology"), filename), check.names = FALSE)
  positive <- grep("-1$", names(raw), value = TRUE)
  stopifnot(length(positive) == 1, "ID" %in% names(raw))
  result <- data.frame(ID = normalize_id(raw$ID), score = as.numeric(raw[[positive]]))
  stopifnot(!anyDuplicated(result$ID), all(is.finite(result$score)), all(result$score >= 0 & result$score <= 1))
  result$HRR_ANY <- labels$HRR_ANY[match(result$ID, labels$ID)]
  stopifnot(!anyNA(result$HRR_ANY))
  result$split <- split
  result$source_file <- filename
  result
}

save_plot <- function(plot, name, width, height) {
  ggsave(file.path(topic_dir, paste0(name, "_R1_recalculated.pdf")), plot, width = width, height = height)
  ggsave(file.path(topic_dir, paste0(name, "_R1_recalculated.png")), plot, width = width, height = height, dpi = 300)
}

MODEL_PREFIX <- "Path_LightGBM"
MODEL_LABEL <- "Pathology Signature"
FIGURE_WATERFALL <- "Figure3F_pathology_waterfall"

train_scores <- read_scores(MODEL_PREFIX, "train")
test_scores <- read_scores(MODEL_PREFIX, "test")
stopifnot(length(intersect(train_scores$ID, test_scores$ID)) == 0)

cutoff <- 0.395
waterfall <- rbind(train_scores, test_scores)
waterfall$cutoff <- cutoff
waterfall$centered_score <- waterfall$score - cutoff
waterfall$status <- factor(waterfall$HRR_ANY, levels = c(0, 1), labels = c("HRP", "HRD"))
waterfall <- waterfall[order(waterfall$split, -waterfall$score, waterfall$ID), ]
waterfall$patient_order <- ave(waterfall$score, waterfall$split, FUN = seq_along)
waterfall$split <- factor(waterfall$split, levels = c("train", "test"), labels = c("Training", "Test"))
write.csv(waterfall, file.path(topic_dir, paste0(FIGURE_WATERFALL, "_scores_R1_recalculated.csv")), row.names = FALSE)
plot <- ggplot(waterfall, aes(patient_order, centered_score, fill = status)) +
  geom_col(width = 0.94) + geom_hline(yintercept = 0, linewidth = 0.35) +
  facet_wrap(~split, ncol = 1, scales = "free_x") +
  scale_fill_manual(values = c(HRP = "#6B8491", HRD = "#BC5A42")) +
  labs(x = "Patients sorted by predicted score", y = paste0(MODEL_LABEL, " score - fixed cutoff"),
       subtitle = sprintf("R1 fixed cutoff = %.6f", cutoff), fill = NULL) +
  theme_classic(base_size = 11) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
save_plot(plot, FIGURE_WATERFALL, 8, 6)
cat(MODEL_PREFIX, "training n =", nrow(train_scores), "test n =", nrow(test_scores), "locked cutoff =", cutoff, "\n")

model_names <- c("LR", "NaiveBayes", "SVM", "KNN", "RandomForest", "ExtraTrees",
                 "XGBoost", "LightGBM", "GradientBoosting", "MLP")
test_files <- paste0("Path_", model_names, "_test.csv")
metrics <- do.call(rbind, lapply(test_files, function(filename) {
  prefix <- sub("_test\\.csv$", "", filename)
  data <- read_scores(prefix, "test")
  curve <- roc(data$HRR_ANY, data$score, levels = c(0, 1), direction = "<", quiet = TRUE)
  interval <- as.numeric(ci.auc(curve, method = "delong"))
  data.frame(model = sub("^Path_", "", prefix), n = nrow(data), HRD_n = sum(data$HRR_ANY),
             AUC = as.numeric(auc(curve)), lower95 = interval[1], upper95 = interval[3])
}))
write.csv(metrics, file.path(topic_dir, "Figure3E_test_AUC_DeLong_R1_recalculated.csv"), row.names = FALSE)
auc_plot <- ggplot(metrics, aes(reorder(model, AUC), AUC)) +
  geom_col(fill = "#7F608B", width = 0.65) +
  geom_errorbar(aes(ymin = lower95, ymax = upper95), width = 0.2) +
  coord_flip(ylim = c(0, 1)) + labs(x = NULL, y = "Test AUC (DeLong 95% CI)") + theme_classic(base_size = 11)
save_plot(auc_plot, "Figure3E_pathology_test_AUC", 7, 5)
distribution <- ggplot(waterfall, aes(status, score, fill = status)) +
  geom_violin(trim = TRUE, alpha = 0.6) + geom_boxplot(width = 0.16, outlier.shape = NA) +
  facet_wrap(~split, nrow = 1) +
  scale_fill_manual(values = c(HRP = "#6B8491", HRD = "#BC5A42")) +
  scale_y_continuous(limits = c(0, 1)) + labs(x = NULL, y = "Pathology Signature score", fill = NULL) + theme_classic(base_size = 11)
save_plot(distribution, "Figure3G_pathology_score_distribution", 7, 4)
print(metrics)
