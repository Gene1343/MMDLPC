
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

library(readxl)
library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

panel_dir <- file.path(PROJECT_ROOT, "06_Human_AI_Study")
input_file <- file.path(panel_dir, "Reader_Scores.xlsx")
reader_columns <- paste("Pathologist", 1:9)
group_names <- c("Pathologists 1-3", "Pathologists 4-6", "Pathologists 7-9")

data <- read_xlsx(input_file, sheet = "ACC_inter")
data1 <- read_xlsx(input_file, sheet = "Acc_with_pathscore")
data <- data[!is.na(data$Cases), c("Cases", "HRD.status", "Pathology.Signature", reader_columns)]
data1 <- data1[!is.na(data1$Cases), names(data)]
stopifnot(!anyDuplicated(data$Cases), !anyDuplicated(data1$Cases),
          setequal(data$Cases, data1$Cases))
data1 <- data1[match(data$Cases, data1$Cases), ]
stopifnot(identical(data$HRD.status, data1$HRD.status),
          all(as.matrix(data[, reader_columns]) %in% 0:1),
          all(as.matrix(data1[, reader_columns]) %in% 0:1))

calculate_reader_metrics <- function(df, condition) {
  result <- lapply(seq_along(reader_columns), function(i) {
    confusion <- table(factor(df$HRD.status, levels = 0:1),
                       factor(df[[reader_columns[i]]], levels = 0:1))
    data.frame(Pathologist = reader_columns[i], Group = group_names[ceiling(i / 3)],
               Condition = condition, N_cases = nrow(df),
               Sensitivity = confusion[2, 2] / sum(confusion[2, ]),
               Specificity = confusion[1, 1] / sum(confusion[1, ]),
               Accuracy = sum(diag(confusion)) / sum(confusion))
  })
  bind_rows(result)
}
reader_metrics <- bind_rows(calculate_reader_metrics(data, "Before"),
                            calculate_reader_metrics(data1, "After"))
reader_metrics$Condition <- factor(reader_metrics$Condition, levels = c("Before", "After"))
group_metrics <- reader_metrics %>% group_by(Condition, Group) %>%
  summarise(N_readers = n(), SD_sensitivity = sd(Sensitivity),
            SD_specificity = sd(Specificity), Sensitivity = mean(Sensitivity),
            Specificity = mean(Specificity), .groups = "drop")
stopifnot(all(group_metrics$N_readers == 3),
          all(is.finite(group_metrics$SD_sensitivity)),
          all(is.finite(group_metrics$SD_specificity)))
write.csv(reader_metrics, file.path(panel_dir, "Figure_6_Reader_Metrics_R1_recalculated.csv"), row.names = FALSE)
write.csv(group_metrics, file.path(panel_dir, "Figure_6BC_Group_Metrics_R1_recalculated.csv"), row.names = FALSE)

for (condition in c("Before", "After")) {
  df <- if (condition == "Before") data else data1
  roc_obj <- roc(df$HRD.status, df$Pathology.Signature,
                 levels = c(0, 1), direction = "<", quiet = TRUE)
  panel <- if (condition == "Before") "6B" else "6C"
  pdf(file.path(panel_dir, paste0("Figure_", panel, "_Reader_Performance_R1_recalculated.pdf")), width = 5, height = 5)
  plot(roc_obj, print.auc = TRUE, col = "black", lwd = 2, lty = 2,
       xlab = "Specificity", ylab = "Sensitivity", main = condition)
  colors <- viridis::inferno(3, end = 0.8)
  for (i in seq_along(group_names)) {
    v <- group_metrics[group_metrics$Condition == condition & group_metrics$Group == group_names[i], ]
    points(v$Specificity, v$Sensitivity, col = colors[i], pch = 19, cex = 1.4)
    arrows(max(0, v$Specificity - v$SD_specificity), v$Sensitivity,
           min(1, v$Specificity + v$SD_specificity), v$Sensitivity,
           code = 3, angle = 90, length = 0.08, col = colors[i], lwd = 2)
    arrows(v$Specificity, max(0, v$Sensitivity - v$SD_sensitivity),
           v$Specificity, min(1, v$Sensitivity + v$SD_sensitivity),
           code = 3, angle = 90, length = 0.08, col = colors[i], lwd = 2)
  }
  legend("bottomright", legend = group_names, col = colors, pch = 19, bty = "n", cex = 0.8)
  dev.off()
}

mean_accuracy <- reader_metrics %>% group_by(Condition) %>% summarise(Accuracy = mean(Accuracy), .groups = "drop")
p <- ggplot(reader_metrics, aes(x = Accuracy, y = Pathologist, color = Condition, shape = Condition)) +
  geom_point(size = 3) +
  geom_vline(data = mean_accuracy, aes(xintercept = Accuracy, color = Condition), linetype = "dashed") +
  scale_color_manual(values = c(Before = "#410669", After = "#FFAC45")) +
  scale_shape_manual(values = c(Before = 16, After = 2)) +
  labs(x = "Accuracy", y = "Pathologist") + theme_bw() + theme(legend.position = "bottom")
ggsave(file.path(panel_dir, "Figure_6D_Reader_Accuracy_R1_recalculated.pdf"), p, width = 6, height = 4.5)

certainty_before <- read_xlsx(input_file, sheet = "certainty_score")
certainty_after <- read_xlsx(input_file, sheet = "certainty_score_with_pathscore")
stopifnot(nrow(certainty_before) == nrow(data), nrow(certainty_after) == nrow(data1),
          isTRUE(all.equal(certainty_before$`Pathology Signature`, data$Pathology.Signature)),
          isTRUE(all.equal(certainty_after$`Pathology Signature`, data1$Pathology.Signature)),
          all(as.matrix(certainty_before[, reader_columns]) %in% 1:3),
          all(as.matrix(certainty_after[, reader_columns]) %in% 1:3))
certainty_long <- bind_rows(
  certainty_before[, reader_columns] %>% mutate(Case = data$Cases, Condition = "Before"),
  certainty_after[, reader_columns] %>% mutate(Case = data1$Cases, Condition = "After")
) %>% pivot_longer(all_of(reader_columns), names_to = "Pathologist", values_to = "Certainty") %>%
  mutate(Group = group_names[ceiling(match(Pathologist, reader_columns) / 3)],
         Condition = factor(Condition, levels = c("Before", "After")))
certainty_summary <- certainty_long %>% group_by(Condition, Group) %>%
  summarise(N_ratings = n(), Mean = mean(Certainty), SD = sd(Certainty), .groups = "drop")
write.csv(certainty_summary, file.path(panel_dir, "Figure_6E_Certainty_Summary_R1_recalculated.csv"), row.names = FALSE)
p <- ggplot(certainty_summary, aes(Group, Mean, fill = Condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD), width = 0.15, position = position_dodge(0.8)) +
  scale_fill_manual(values = c(Before = "#410669", After = "#FFAC45")) +
  labs(x = NULL, y = "Diagnostic certainty (mean and SD)") + theme_bw() +
  theme(legend.position = "bottom")
ggsave(file.path(panel_dir, "Figure_6E_Diagnostic_Certainty_R1_recalculated.pdf"), p, width = 6, height = 4.5)

certainty_by_reader <- certainty_long %>% group_by(Pathologist, Group, Condition) %>%
  summarise(Certainty = mean(Certainty), .groups = "drop")
paired_results <- list()
for (outcome in c("Accuracy", "Certainty")) {
  long_data <- if (outcome == "Accuracy") reader_metrics else certainty_by_reader
  for (group in c("All pathologists", group_names)) {
    d <- if (group == "All pathologists") long_data else long_data[long_data$Group == group, ]
    wide <- d[, c("Pathologist", "Condition", outcome)] %>%
      pivot_wider(names_from = Condition, values_from = all_of(outcome))
    test <- t.test(wide$After, wide$Before, paired = TRUE)
    d$Doctor <- factor(d$Pathologist)
    d$Condition <- factor(d$Condition, levels = c("Before", "After"))
    fit <- aov(as.formula(paste(outcome, "~ Condition + Error(Doctor / Condition)")), data = d)
    anova_p <- summary(fit)[["Error: Doctor:Condition"]][[1]][["Pr(>F)"]][1]
    paired_results[[length(paired_results) + 1L]] <- data.frame(
      Outcome = outcome, Group = group, N_readers = nrow(wide),
      Mean_before = mean(wide$Before), Mean_after = mean(wide$After),
      Mean_difference = mean(wide$After - wide$Before), Paired_t_p = test$p.value,
      Repeated_measures_ANOVA_p = anova_p)
  }
}
write.csv(bind_rows(paired_results), file.path(panel_dir, "Figure_6_Paired_Reader_Statistics_R1_recalculated.csv"), row.names = FALSE)
print(certainty_summary)
print(bind_rows(paired_results))
