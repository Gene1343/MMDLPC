
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(ggpubr))
suppressPackageStartupMessages(library(ggforce))
pdf(NULL)
topic_dir <- file.path(PROJECT_ROOT, "04_Multimodal/02_Features_Scores_and_Complementarity")
package_dir <- DATA_ROOT
score_dir <- file.path(package_dir, "07_Model_Scores")
normalize_id <- function(x) sub("\\.nii(\\.gz)?$", "", trimws(x), ignore.case = TRUE)

read_signature <- function(folder, prefix, code) {
  result <- do.call(rbind, lapply(c("train", "test"), function(split) {
    raw <- read.csv(file.path(score_dir, folder, paste0(prefix, "_", split, ".csv")), check.names = FALSE)
    data.frame(ID = normalize_id(raw$ID), score = raw[["HRR_ANY-1"]], Dataset_Type = split)
  }))
  stopifnot(!anyDuplicated(result$ID), all(is.finite(result$score)),
            all(result$score >= 0 & result$score <= 1))
  names(result)[2:3] <- c(code, paste0("Dataset_Type_", code))
  result
}
P <- read_signature("Pathology", "Path_LightGBM", "P")
R <- read_signature("Radiology", "Rad_SVM", "R")
PR <- read_signature("Pathology_Radiology", "Path-Rad_NaiveBayes", "PR")
merge_HRD_data <- Reduce(function(x, y) merge(x, y, by = "ID"), list(P, R, PR))
labels <- read.csv(file.path(package_dir, "00_Shared_Data_and_Code", "Data",
                             "CPGEA-TCGA 20230106 OK.csv"), check.names = FALSE)
stopifnot(!anyDuplicated(labels$ID))
merge_HRD_data$HRR_ANY <- labels$HRR_ANY[match(merge_HRD_data$ID, labels$ID)]
stopifnot(!anyNA(merge_HRD_data$HRR_ANY))

merge_HRD_data$R_class <- with(merge_HRD_data,
  as.integer(R > ifelse(Dataset_Type_R == "train", 0.160, 0.201)))
merge_HRD_data$P_class <- with(merge_HRD_data,
  as.integer(P > ifelse(Dataset_Type_P == "train", 0.499, 0.363)))
merge_HRD_data$PR_class <- as.integer(merge_HRD_data$PR > 0.99)
merge_HRD_data$P_R <- with(merge_HRD_data, ifelse(P_class == 0 & R_class == 0, "both HRP",
  ifelse(P_class == 1 & R_class == 1, "both HRD",
         ifelse(P_class == 1, "Histopathological HRD", "Radiomics HRD"))))
merge_HRD_data$PR_correct <- ifelse(merge_HRD_data$PR_class == merge_HRD_data$HRR_ANY,
                                   "correct", "incorrect")
write.csv(merge_HRD_data, file.path(topic_dir, "Figure4E_F_patient_scores_R1_recalculated.csv"), row.names = FALSE)

plots <- ggscatterhist(merge_HRD_data, x = "P", y = "R", color = "P_R",
  size = 2, alpha = 0.7, palette = c("#D95A6B", "#000000", "#2D078F", "#FBBA33"),
  margin.params = list(fill = "P_R", color = "black", size = 0.5),
  main.plot.size = 2, margin.plot.size = 0.2)
plots$sp <- plots$sp + geom_mark_rect(aes(fill = P_R)) +
  scale_fill_manual(values = c("#D95A6B", "#000000", "#2D078F", "#FBBA33")) +
  coord_fixed(ratio = 1/2)
figure <- print(plots)
ggsave(file.path(topic_dir, "Figure4E_complementarity_R1_recalculated.pdf"), figure, width = 8, height = 6)

merge_HRD_data$PR_class <- factor(merge_HRD_data$PR_class, levels = c(0, 1), labels = c("HRP", "HRD"))
plots <- ggscatterhist(merge_HRD_data, x = "P", y = "R", color = "PR_class",
  size = 2, alpha = 0.5, palette = c("#D95A6B", "#2D078F"),
  margin.params = list(fill = "PR_class", color = "black", size = 0.5),
  main.plot.size = 2, margin.plot.size = 0.2)
plots$sp <- plots$sp +
  geom_point(data = subset(merge_HRD_data, PR_correct == "incorrect"), aes(x = P, y = R),
             shape = 4, size = 2, alpha = 1) +
  geom_hline(yintercept = 0.201, linetype = "dashed", color = "black") +
  geom_vline(xintercept = 0.363, linetype = "dashed", color = "black") +
  coord_fixed(ratio = 1/2)
figure <- print(plots)
ggsave(file.path(topic_dir, "Figure4F_PR_classification_R1_recalculated.pdf"), figure, width = 8, height = 6)
dev.off()
