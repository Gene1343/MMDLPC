PROJECT_ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
while (!file.exists(file.path(PROJECT_ROOT, "modeling_pipeline.py"))) {
  parent <- dirname(PROJECT_ROOT)
  if (identical(parent, PROJECT_ROOT)) stop("Run this code from the MMDLPC_Code_PDF folder.")
  PROJECT_ROOT <- parent
}
DATA_ROOT <- file.path(dirname(PROJECT_ROOT), "MMDLPC_Code_PDF_Data")

library(readxl)
library(pheatmap)

data <- read_excel(file.path(DATA_ROOT, "02_Radiomics/01_Feature_Selection_and_Correlation/Supplementary_Table_3_original.xlsx"), sheet = "Sheet1")
data <- data[,-11]

cor_matrix <- cor(data[, -1])

mycol1 <- colorRampPalette(c("navy", "white", "orange"))(100)

pheatmap(cor_matrix,
         filename = file.path(PROJECT_ROOT, "02_Radiomics/01_Feature_Selection_and_Correlation/Figure2D_Pearson_heatmap.pdf"),
         scale="none",
         fontsize = 8,
         fontsize_row = 1e-10,
         fontsize_col = 1e-10,
         cellwidth = 25,
         cellheight = 25,
         color = mycol1,
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         width = 8,
         height = 8
)
