
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(survminer))
suppressPackageStartupMessages(library(ggplot2))

analysis_dir <- file.path(DATA_ROOT, "01_Clinical_and_Cohort/03_BCR_Survival")
pdf_dir <- png_dir <- results_dir <- file.path(PROJECT_ROOT, "01_Clinical_and_Cohort/03_BCR_Survival")
membership <- read.csv(file.path(analysis_dir, "model_membership_long.csv"), check.names = FALSE)
bcr <- read.csv(file.path(analysis_dir, "BCR_analysis_complete_cases.csv"), check.names = FALSE)

normalize_id <- function(x) toupper(gsub("\\s+", "", trimws(as.character(x))))
membership$id <- normalize_id(membership$id)
bcr$ID <- normalize_id(bcr$ID)

stopifnot(!anyDuplicated(bcr$ID))
stopifnot(all(na.omit(unique(bcr$BCR_event)) %in% c(0, 1)))
stopifnot(all(is.finite(bcr$BCR_TIME_months) & bcr$BCR_TIME_months >= 0))

model_spec <- data.frame(
  model = c("Clinical", "Pathology", "Radiology"),
  algorithm = c("KNN", "LightGBM", "SVM"),
  short_label = c("C", "P", "R"),
  display_label = c(
    "Clinical score (C)",
    "Pathology score (P)",
    "Radiology score (R)"
  ),
  stringsAsFactors = FALSE
)

fmt_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

safe_stats <- function(data, group_variable, continuous_variable = NULL) {
  group <- data[[group_variable]]
  if (length(unique(group)) < 2) {
    return(list(
      logrank_p = NA_real_, c_index = NA_real_,
      HR = NA_real_, lower_95 = NA_real_, upper_95 = NA_real_, Cox_p = NA_real_
    ))
  }

  surv_formula <- as.formula(
    paste0("Surv(BCR_TIME_years, BCR_event) ~ ", group_variable)
  )
  lr <- survdiff(surv_formula, data = data)
  logrank_p <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)

  data$high_indicator <- as.integer(group == levels(group)[1])
  cox <- coxph(
    Surv(BCR_TIME_years, BCR_event) ~ high_indicator,
    data = data
  )
  cox_summary <- summary(cox)

  if (is.null(continuous_variable)) {
    concordance_predictor <- data$high_indicator
  } else {
    concordance_predictor <- data[[continuous_variable]]
  }
  c_index <- concordance(
    Surv(data$BCR_TIME_years, data$BCR_event) ~ concordance_predictor,
    reverse = TRUE
  )$concordance

  list(
    logrank_p = logrank_p,
    c_index = c_index,
    HR = unname(cox_summary$conf.int["high_indicator", "exp(coef)"]),
    lower_95 = unname(cox_summary$conf.int["high_indicator", "lower .95"]),
    upper_95 = unname(cox_summary$conf.int["high_indicator", "upper .95"]),
    Cox_p = unname(cox_summary$coefficients["high_indicator", "Pr(>|z|)"])
  )
}

get_model_data <- function(spec, split_name) {
  scores <- membership[
    membership$model == spec$model &
      membership$algorithm == spec$algorithm &
      membership$split == split_name &
      is.finite(membership$score),
    c("id", "score"),
    drop = FALSE
  ]
  if (anyDuplicated(scores$id)) {
    stop("Duplicate IDs for ", spec$model, " / ", split_name)
  }
  data <- merge(scores, bcr, by.x = "id", by.y = "ID", all = FALSE, sort = FALSE)
  data$BCR_TIME_years <- data$BCR_TIME_months / 12
  data
}

make_km_plot <- function(
  data,
  group_variable,
  title_text,
  annotation_text,
  legend_labels,
  palette,
  line_types,
  show_legend = FALSE,
  show_x_label = FALSE,
  show_y_label = FALSE
) {
  data$plot_group <- data[[group_variable]]
  x_limit <- max(2, ceiling(max(data$BCR_TIME_years, na.rm = TRUE)))
  fit <- survfit(
    Surv(BCR_TIME_years, BCR_event) ~ plot_group,
    data = data,
    conf.type = "log"
  )

  km <- ggsurvplot(
    fit,
    data = data,
    conf.int = TRUE,
    conf.int.style = "ribbon",
    conf.int.alpha = 0.13,
    censor = TRUE,
    censor.shape = 3,
    censor.size = 1.7,
    risk.table = TRUE,
    risk.table.height = 0.20,
    risk.table.fontsize = 3.1,
    risk.table.y.text = TRUE,
    risk.table.y.text.col = TRUE,
    risk.table.title = "Number at risk",
    palette = palette,
    linetype = "strata",
    legend.title = NULL,
    legend.labs = legend_labels,
    xlim = c(0, x_limit),
    break.time.by = 2,
    ylim = c(0, 1.05),
    xlab = if (show_x_label) "BCR-free follow-up (years)" else NULL,
    ylab = if (show_y_label) "BCR-free survival probability" else NULL,
    ggtheme = theme_bw(base_family = "Helvetica", base_size = 9),
    tables.theme = theme_cleantable(base_size = 5.5)
  )

  km$plot <- km$plot +
    scale_linetype_manual(values = line_types) +
    labs(title = title_text, subtitle = NULL) +
    annotate(
      "text",
      x = 0.97 * x_limit,
      y = 0.08,
      label = annotation_text,
      hjust = 1,
      vjust = 0,
      size = 2.7,
      family = "Helvetica"
    ) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.6),
      plot.title = element_text(face = "bold", size = 9),
      legend.position = if (show_legend) "inside" else "none",
      legend.position.inside = c(0.03, 0.03),
      legend.justification = c(0, 0),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "white", colour = NA),
      legend.title = element_blank(),
      legend.margin = margin(0, 0, 0, 0),
      legend.text = element_text(size = 7),
      legend.key.height = grid::unit(3.5, "mm"),
      legend.key.width = grid::unit(7, "mm"),
      axis.title = element_text(face = "bold", size = 8),
      axis.text = element_text(size = 7, colour = "black"),
      plot.margin = margin(4, 8, 2, 4)
    )
  km$table <- km$table +
    theme(
      plot.title = element_text(face = "bold", size = 6),
      axis.text.x = element_text(size = 5),
      plot.margin = margin(0, 8, 2, 4)
    )
  km
}

model_data <- list()
cutoff_table <- list()
for (i in seq_len(nrow(model_spec))) {
  spec <- model_spec[i, ]
  train_data <- get_model_data(spec, "train")
  test_data <- get_model_data(spec, "test")
  model_data[[spec$short_label]] <- list(train = train_data, test = test_data)

  all_training_scores <- membership[
    membership$model == spec$model &
      membership$algorithm == spec$algorithm &
      membership$split == "train" &
      is.finite(membership$score),
    "score"
  ]
  median_cutoff <- median(all_training_scores)

  cutoff_table[[spec$short_label]] <- list(
    median = list(
      cutoff = median_cutoff,
      rule = "Median of all training scores; locked for test"
    )
  )
}

all_audit <- list()

plot_list <- list()

for (i in seq_len(nrow(model_spec))) {
  spec <- model_spec[i, ]
  cutoff_info <- cutoff_table[[spec$short_label]][["median"]]
  cutoff <- cutoff_info$cutoff

  for (split_name in c("train", "test")) {
    data <- model_data[[spec$short_label]][[split_name]]
    data$score_group <- factor(
      ifelse(data$score >= cutoff, "High score", "Low score"),
      levels = c("High score", "Low score")
    )
    stats <- safe_stats(data, "score_group", continuous_variable = "score")
    annotation <- sprintf(
      "HR = %.2f    P = %s",
      stats$HR, fmt_p(stats$logrank_p)
    )
    plot_list[[length(plot_list) + 1]] <- make_km_plot(
      data = data,
      group_variable = "score_group",
      title_text = paste0(
        spec$model,
        " | ",
        if (split_name == "train") "Development" else "Validation"
      ),
      annotation_text = annotation,
      legend_labels = c("High score", "Low score"),
      palette = c("#00468B", "#ED0000"),
      line_types = c("solid", "dashed"),
      show_legend = i == 1 && split_name == "train",
      show_x_label = i == nrow(model_spec),
      show_y_label = split_name == "train"
    )

    group_n <- table(data$score_group)
    group_events <- tapply(data$BCR_event, data$score_group, sum)
    all_audit[[length(all_audit) + 1]] <- data.frame(
      version = "median",
      analysis = spec$display_label,
      model = spec$model,
      algorithm = spec$algorithm,
      split = split_name,
      cutoff = cutoff,
      cutoff_rule = cutoff_info$rule,
      BCR_complete_n = nrow(data),
      BCR_events = sum(data$BCR_event),
      group_1 = "High score",
      group_1_n = unname(group_n["High score"]),
      group_1_events = unname(group_events["High score"]),
      group_2 = "Low score",
      group_2_n = unname(group_n["Low score"]),
      group_2_events = unname(group_events["Low score"]),
      max_to_min_group_ratio = max(group_n) / min(group_n),
      c_index = stats$c_index,
      logrank_p = stats$logrank_p,
      HR_group1_vs_group2 = stats$HR,
      HR_lower_95 = stats$lower_95,
      HR_upper_95 = stats$upper_95,
      Cox_p = stats$Cox_p,
      stringsAsFactors = FALSE
    )

  }
}

plot_list <- plot_list[c(1, 3, 5, 2, 4, 6)]

arranged <- arrange_ggsurvplots(
  plot_list,
  print = FALSE,
  ncol = 2,
  nrow = 3,
  risk.table.height = 0.20
)

file_stub <- "FigureS6_BCR_KM_final"
pdf_file <- file.path(pdf_dir, paste0(file_stub, ".pdf"))
png_file <- file.path(png_dir, paste0(file_stub, ".png"))

pdf(pdf_file, width = 9.33, height = 15.0, family = "Helvetica", onefile = TRUE)
grid::grid.newpage()
grid::grid.draw(arranged[[1]])
dev.off()

png(png_file, width = 2800, height = 4500, res = 300, type = "windows")
grid::grid.newpage()
grid::grid.draw(arranged[[1]])
dev.off()

median_files <- c(pdf = pdf_file, png = png_file)

audit <- do.call(rbind, all_audit)

write.csv(
  audit,
  file.path(results_dir, "BCR_survival_results_final.csv"),
  row.names = FALSE,
  na = ""
)

cat("Created median files:\n", paste(median_files, collapse = "\n"), "\n", sep = "")
print(audit)
