CCCvelo Project Structure

# step1: data preparing

Before running CCCvelo, using 1_select_LRTG.R function to select candidate ligands, receptors, and feature genes from the expression data, and then save the result into the input files under the path Input/your_project_name/. The input files include:
```
 raw_expression_mtx.csv # Raw expression matrix (cells × genes)
 cell_meta.csv # Cell meta information (Cluster annotations)
 cell_location.csv # Cell spatial coordinates
 Databases.json # Ligand-Receptor-TF database
 Ligs_list.json # Candidate Ligands
 Recs_list.json # Candidate Receptors
 TGs_list.json # Candidate Target Genes
```

# step2: Load Input Data

Setting the global path
```
DATA_DIR = "./data/processed/"
MLNET_DIR = "./results2/mlnet/"
MODEL_DIR = "./results2/trained_model/"
TG_PRED_DIR = "./results2/tg_prediction/"
LOSS_DIR = "./results2/loss_curves/"
VISUALIZE_DIR = './results2/visualize/'
```
# This loads the expression matrix, metadata, and spatial coordinates into an AnnData object.
```
input_dir = os.path.join(DATA_DIR, "your_project_name")
print("Loading data...")
data_files = {
    'count_file': 'raw_expression_mtx.csv',
    'imput_file': 'imputation_expression_mtx.csv',
    'meta_file': 'cell_meta.csv',
    'loca_file': 'cell_location.csv'
}
paths = {key: os.path.join(input_dir, fname) for key, fname in data_files.items()}
adata = ReadData(**paths)
```
# Step 2: Constructing Multilayer Network

(1) load candidate ligands, receptors, and feature genes
```
print("Loading database...")
TGs_list = load_json(os.path.join(input_dir, "TGs_list.json")) # feature genes
Ligs_list = load_json(os.path.join(input_dir, "Ligs_list.json")) # condinate ligands 
Recs_list = load_json(os.path.join(input_dir, "Recs_list.json")) # condinate receptors 
```
(2) construct multilayer network
```
print("Building multilayer network...")
resMLnet = runMLnet(
    adata=adata,
    LigClus=None,
    RecClus=rec_clusters,
    OutputDir=MLNET_DIR,
    Databases=None,
    RecTF_method="Search",
    TFTG_method="Search",
    TGList=TGs_list,
    LigList=Ligs_list,
    RecList=Recs_list
)
ex_mulnetlist = {
    name: mlnet
    for receiver, sender_dict in resMLnet["mlnets"].items()
    for name, mlnet in sender_dict.items()
    if not mlnet["LigRec"].empty
}

print("Multilayer network nodes summary:")
print(summarize_multilayer_network(ex_mulnetlist))
```
# step3: Computes signaling scores for each LR–TF path using predefined ligand–receptor databases, where ligand–receptor databases contain diffusion-based LR database and contact-based LR database.
```
loop_calculate_LRTF_allscore(
    adata=adata,
    ex_mulnetlist=ex_mulnetlist,
    receiver_celltype=rec_clusters,
    diff_LigRecDB_path='E:/CCCvelo/data/Database/diff_LigRecDB.csv', 
    cont_LigRecDB_path='E:/CCCvelo/data/Database/cont_LigRecDB.csv', 
    OutputDir=MLNET_DIR
)
```
Processing LR–TF signaling scores 
```
TFLR_all_score = get_TFLR_allactivity(
    mulNetList=ex_mulnetlist,
    OutputDir=MLNET_DIR
)
save_LRscore_and_MLnet(
    adata,
    mulNetList=ex_mulnetlist,
    TFLR_all_score=TFLR_all_score,
    save_path=MLNET_DIR
)
```
# step4: prepare CCCvelo input, including loading linkage files (LR pairs, TF-TG linkages, score matrix) and filtering cells belonging to the recipient cluster.

(1) filtering cells belonging to the recipient cluster
```
print("Selecting receiver cells...")
celltype_ls = adata.obs['Cluster'].to_list()
ct_index_ls = []
for name in rec_clusters:
    ct_index_ls.extend(get_index1(celltype_ls, name))

adata = adata[ct_index_ls, :].copy()
```
loading linkage files (LR pairs, TF-TG linkages, score matrix)
```
link_files = {
    'LR_link_file': 'LR_links.csv',
    'TFTG_link_file': 'TFTG_links.csv',
    'LRTF_score_file': 'TFLR_score/'
}
paths = {key: os.path.join(MLNET_DIR, fname) for key, fname in link_files.items()}
print('Loading link files from:', paths)
```
preparing CCCvelo input
```
adata = PrepareInputData(adata, **paths)
adata.uns['Cluster_colors'] = ["#DAA0B0", "#908899", "#9D5A38"]
torch.save(adata, os.path.join(MLNET_DIR, "pp_adata.pt"))
```
# step 5: select root cell and train CCCvelo model

identify root cell
```
adata = root_cell(adata, select_root='UMAP')
```
trainging CCCvelo model, when the number of cells is greater than 10,000, choose to use batch training.
```
print("Training spatial velocity model...")
n_cells = adata.n_obs
print(f"Number of receiver cells: {n_cells}")
if n_cells <= 10000:
    print("Training with standard SpatialVelocity (full batch)...")
    
    from models2.train_CCCvelo import SpatialVelocity
    from models2.plot_CCCvelo import plot_gene_dynamic

    data = PrepareData(adata, hidden_dims=hidden_dims)
    model = SpatialVelocity(*data, lr=learning_rate, Lambda=lambda_reg)
    iteration_adam, loss_adam = model.train(200)

    # plt_path = os.path.join(results_path, "figure/")
    # create_directory(plt_path)
    plot_gene_dynamic(adata_velo, model, VISUALIZE_DIR)
else:
    print("Training with batch SpatialVelocity (mini-batch mode)...")

    from models2.train_CCCvelo_batchs import SpatialVelocity
    from models2.plot_CCCvelo_batch import plot_gene_dynamic

    data = PrepareData(adata, hidden_dims=hidden_dims)
    model = SpatialVelocity(*data, lr=learning_rate, Lambda=lambda_reg, batch_size=batch_size)
    iteration_adam, loss_adam = model.train(n_epochs)
adata.write_h5ad(os.path.join(MODEL_DIR, 'adata_pyinput.h5ad'))
```
# step5: plots predicted TG dynamics, spatial velocity streamline and saves trained model and velocity-augmented AnnData
```
adata_copy = adata[:, adata.var['TGs'].astype(bool)]
adata_velo = get_raw_velo(adata_copy, model)
plot_gene_dynamic(adata_velo, model, VISUALIZE_DIR)

print('===============plot spatial stream line and calculate velocity_psd without setting root cell==============')
adata_spa = adata_velo.copy()
plot_velocity_streamline(adata_spa, basis='spatial', vkey='velocity', xkey='Imputate', root_key=False, density=2,
                            smooth=0.5, cutoff_perc=0, plt_path=VISUALIZE_DIR, save_name='v0')


save_model_and_data(model, adata_velo, MODEL_DIR)

print("Pipeline finished successfully!")
```
