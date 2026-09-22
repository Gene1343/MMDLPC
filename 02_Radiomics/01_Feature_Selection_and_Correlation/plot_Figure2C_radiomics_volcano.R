
PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

suppressPackageStartupMessages(library(EnhancedVolcano))
suppressPackageStartupMessages(library(readxl))
topic_dir <- file.path(PROJECT_ROOT, "02_Radiomics/01_Feature_Selection_and_Correlation")
volcano_data <- read_xlsx(file.path(DATA_ROOT, "02_Radiomics/01_Feature_Selection_and_Correlation/volcano_data_processed.xlsx"))
stopifnot(all(c("Log2FoldChange", "PValue") %in% names(volcano_data)))
stopifnot(all(is.finite(volcano_data$Log2FoldChange)),
          all(volcano_data$PValue > 0 & volcano_data$PValue <= 1))
pdf(file.path(topic_dir, "Figure2C_volcano_from_saved_statistics.pdf"), width = 8, height = 7)
print(EnhancedVolcano(volcano_data,
                lab = volcano_data$Feature,
                x = 'Log2FoldChange',
                y = 'PValue',
                title = 'Volcano Plot',
                xlab = bquote(~Log[2]~'fold change'),
                pCutoff = 0.05,
                FCcutoff = 1,
                pointSize = 2,
                labSize = 4.0,
                colAlpha = 0.6,
                legendPosition = 'top',
                legendLabSize = 16,
                legendIconSize = 5.0,
                labCol = 'black',
                labFace = 'bold',
                boxedLabels = TRUE,
                drawConnectors = TRUE,
                widthConnectors = 1,
                colConnectors = 'black'))
dev.off()
