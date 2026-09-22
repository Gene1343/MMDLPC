
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

library(pROC)
library(ggplot2)

panel_dir <- file.path(PROJECT_ROOT, "03_Pathomics/01_Patch_Evaluation")
files <- c(Training = "BST_TRAIN_RESULTS.txt", Test = "BST_VAL_RESULTS.txt")
roc_results <- list()
metric_results <- list()
cutoff <- 0.5

for (cohort in names(files)) {
  data <- read.delim(file.path(DATA_ROOT, "03_Pathomics", files[[cohort]]), header = FALSE, quote = "", comment.char = "",
                     col.names = c("ID", "pred_score", "pred_label", "gt"),
                     colClasses = c("NULL", "numeric", "integer", "integer"))
  stopifnot(all(data$pred_label %in% 0:1), all(data$gt %in% 0:1),
            all(is.finite(data$pred_score)), all(data$pred_score >= 0 & data$pred_score <= 1))
  tumor_probability <- ifelse(data$pred_label == 1L, data$pred_score, 1 - data$pred_score)
  predicted <- factor(as.integer(tumor_probability >= cutoff), levels = 0:1, labels = c("Normal", "Tumor"))
  actual <- factor(data$gt, levels = 0:1, labels = c("Normal", "Tumor"))
  confusion <- table(Predicted = predicted, Actual = actual)
  tp <- confusion["Tumor", "Tumor"]
  tn <- confusion["Normal", "Normal"]
  fp <- confusion["Tumor", "Normal"]
  fn <- confusion["Normal", "Tumor"]
  roc_obj <- roc(data$gt, tumor_probability, levels = c(0, 1), direction = "<", quiet = TRUE)
  ci <- ci.auc(roc_obj, method = "delong")
  roc_results[[cohort]] <- roc_obj
  metric_results[[cohort]] <- data.frame(
    Cohort = cohort, N_patches = nrow(data), TP = tp, TN = tn, FP = fp, FN = fn,
    ACC = (tp + tn) / sum(confusion), AUC = as.numeric(auc(roc_obj)),
    CI_lower = as.numeric(ci[1]), CI_upper = as.numeric(ci[3]),
    Sensitivity = tp / (tp + fn), Specificity = tn / (tn + fp),
    PPV = tp / (tp + fp), NPV = tn / (tn + fn), Precision = tp / (tp + fp),
    Recall = tp / (tp + fn), F1 = 2 * tp / (2 * tp + fp + fn),
    Cutoff = cutoff, Positive_class = "Tumor (1)", CI_unit = "Patch", CI_method = "DeLong")
  write.csv(as.data.frame(confusion), file.path(panel_dir, paste0("Figure_3D_", cohort, "_Confusion_R1_recalculated.csv")), row.names = FALSE)
  pdf(file.path(panel_dir, paste0("Figure_3D_", cohort, "_Confusion_R1_recalculated.pdf")), width = 5, height = 5)
  fourfoldplot(confusion, color = c("#C1536A", "#BCD5E3"), margin = 1, conf.level = 0,
               main = paste(cohort, "patches"))
  dev.off()
}
results <- do.call(rbind, metric_results)
write.csv(results, file.path(panel_dir, "Table_S5_Patch_Metrics_R1_recalculated.csv"), row.names = FALSE)
print(results)
ROCplot <- ggroc(roc_results, legacy.axes = TRUE, linewidth = 0.8) +
  geom_abline(intercept = 0, slope = 1, linewidth = 0.2) +
  scale_color_manual(values = c(Training = "black", Test = "#FF0050")) +
  theme_bw() + theme(legend.title = element_blank(), legend.position = "bottom") +
  coord_equal() + labs(x = "1 - Specificity", y = "Sensitivity")
ggsave(file.path(panel_dir, "Figure_3C_Patch_ROC_R1_recalculated.pdf"), ROCplot, width = 5, height = 5)
