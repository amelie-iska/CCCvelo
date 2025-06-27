library(Seurat)
library(readr)
library(dplyr)
library(ggsci)
library(scales)
library(ggplot2)
library(SeuratWrappers)

rm(list = ls())
gc()

setwd("/home/yll/velocity_methods/01_analysis/apply_in_stereo_cortex")
source('/home/yll/velocity_methods/01_analysis/apply_in_stereo_cortex/R/preprocess_code.R')

# load data
data_path <- paste0(getwd(),'/data/')
files <- list.files(data_path)
files <- files[grep(".csv",files)]
cell_file <- files[grep("bin60",files)]

bin60_cnt <- read.csv(paste0(data_path,"bin60_clustered_with_count.csv")) %>% .[,-1]
bin60_gene <- read_csv(paste0(data_path,"bin60_clustered_with_gene.csv")) %>% .[,-1] %>% as.data.frame(.)
bin60_meta <- read_csv(paste0(data_path,"bin60_clustered_with_meta.csv")) %>% .[,-1] %>% as.data.frame(.)
bin60_loc <- read_csv(paste0(data_path,"bin60_clustered_with_loc.csv")) %>% .[,-1] %>% as.data.frame(.)

bin60_cnt <- t(bin60_cnt) %>% as.matrix(.)
gene_num <- dim(bin60_cnt)[1]
cell_num <- dim(bin60_cnt)[2]

rownames(bin60_cnt) <- toupper(bin60_gene$`0`)
colnames(bin60_cnt) <- paste0("bin60_",seq(cell_num))
str(bin60_cnt)

bin60_meta$scc_anno <- gsub("/","",bin60_meta$scc_anno)
rownames(bin60_meta) <- colnames(bin60_cnt)
rownames(bin60_loc) <- colnames(bin60_cnt)
colnames(bin60_loc) <- c("x","y")
str(bin60_meta)
bin60_loc <- bin60_loc[rownames(bin60_meta),]

# creat seurat object 
ser_obj <- CreateSeuratObject(bin60_cnt,
                                 meta.data = bin60_meta, 
                                 assay="Spatial",
                                 min.cells = 20)
ser_obj@images$spatial <- bin60_loc

ser_obj <- SCTransform(ser_obj, assay = 'Spatial')
ser_obj <- FindVariableFeatures(ser_obj, nfeatures = 3000)

## imputation
seed <- 4321
norm.matrix <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
exprMat.Impute <- run_Imputation(exprMat = norm.matrix,use.seed = T,seed = seed)

ser_obj$Cluster <- ser_obj$scc_anno
sub_anno <- data.frame(Barcode=colnames(ser_obj),Cluster=ser_obj$Cluster)
sub_loca <- data.frame(x=ser_obj@images$spatial$x,y=ser_obj@images$spatial$y)
Idents(ser_obj) <- ser_obj@meta.data$Cluster

# load prior databse
Databases <- readRDS('/home/yll/velocity_methods/01_analysis/prior_knowledge/Databases.rds')
names(Databases) <- gsub('\\.','',names(Databases))
quan.cutoff <- 0.98
Databases <- Databases
Databases$RecTFDB <- Databases$RecTFDB %>%
  .[.$score > quantile(.$score, quan.cutoff),] %>%
  dplyr::distinct(source, target)
Databases$LigRecDB <- Databases$LigRecDB %>%
  dplyr::distinct(source, target) %>%
  dplyr::filter(target %in% Databases$RecTFDB$source)
Databases$TFTGDB <- Databases$TFTGDB %>%
  dplyr::distinct(source, target) %>%
  dplyr::filter(source %in% Databases$RecTFDB$target)

LRTG_list <- select_LRTG(ser_obj, Databases, log.gc = 0.25, p_val_adj=0.05, pct.ct=0.01, expr.ct = 0.1)

TGs_list <- LRTG_list[["TGs_list"]]
Ligs_expr_list <- LRTG_list[["Ligs_expr_list"]]
Recs_expr_list <- LRTG_list[["Recs_expr_list"]]

## save results
output_fpath <- paste0(getwd(), '/data/processed/')

if(!dir.exists(output_fpath)){
  dir.create(output_fpath)
}

write_json(TGs_list, path=paste0(output_fpath,"TGs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Ligs_expr_list, path=paste0(output_fpath,"Ligs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Recs_expr_list, path=paste0(output_fpath,"Recs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Databases, path=paste0(output_fpath,"Databases.json"), pretty = TRUE, auto_unbox = TRUE)

df_count <- as.matrix(GetAssayData(ser_obj, "data", "Spatial"))
rownames(exprMat.Impute) = rownames(df_count)
df_count = df_count[rownames(exprMat.Impute),]
df_count = t(df_count)
exprMat.Impute <- as.matrix(exprMat.Impute)
exprMat.Impute = t(exprMat.Impute)

write.table(df_count,file=paste0(output_fpath, 'raw_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(exprMat.Impute,file=paste0(output_fpath, 'imputation_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(sub_anno,file=paste0(output_fpath, 'cell_meta.csv'),sep = ",",row.names = FALSE,col.names = TRUE)
write.table(sub_loca,file=paste0(output_fpath, 'cell_location.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
