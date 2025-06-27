# CCCvelo: Decoding dynamic cell-cell communication-driven cell state transitions in spatial transcriptomics

<p align="center">
  <img src="https://github.com/SunXQlab/CCCvelo/blob/main/fig1.framework.png">
</p>

CCCvelo is a computational framework designed to reconstruct CCC-driven CST dynamics by jointly optimizing a dynamic CCC signaling network and a latent cell-state transitions (CST) clock through a multiscale, nonlinear network kinetics model. CCCvelo can estimated RNA velocity, cell pseudotime, pseudo-temporal dynamics of TGs’ expressions or TFs’ activities, and the cell state-specific multilayer signaling network of CCC. These functionalities enable the reconstruction of spatiotemporal trajectories of cells while simultaneously capturing dynamic cellular communication driving CST. CCCvelo employs several visualization strategies to facilitate the analysis of CCC-driven CST dynamics. These visualizations mainly include velocity streamlines illustrating CST trajectories, heatmap visualizations of gene expression and TF activity along pseudotime, and multilayer network plots of CCC displaying the signaling paths from upstream LR pairs to TFs and then to the downstream TGs.

The main features of CCCvelo are：

* (1) the reconstruction of spatiotemporal dynamics of CCC-regulated CSTs within a spatial context <br>
* (2) quantitative ordering of cellular progression states through velocity vector field embedding <br>
* (3) the identification of dynamic rewiring of CCC signaling <br>

# Environment
h5py                3.11.0 <br>
matplotlib          3.7.5 <br>
mpmath              1.3.0 <br>
networkx            3.1 <br>
numba               0.58.1 <br>
numpy               1.24.4 <br>
pandas              2.0.3 <br>
pip                 23.2.1 <br>
python-dateutil     2.9.0.post0 <br>
python-utils        3.8.2 <br>
scanpy              1.9.8 <br>
scipy               1.10.1 <br>
scvelo              0.3.2 <br>
seaborn             0.13.2 <br>
setuptools          68.2.0 <br>
threadpoolctl       3.5.0 <br>
torch               2.0.1 <br>
anndata             0.9.2  <br>     

# Usage

The use of CCCvelo involves the following steps. 

## Installation

 Please install the corresponding version of the package according to the environment provided above, then the package CCCvelo can be directly downloaded for usage.

## Data Preparation

Before running CCCvelo, you need using `select_LRTG.R` function to select candidate ligands, receptors, and feature genes from the expression data, and then save the result into the input files under the path `Input/your_project_name/`. The input files include:

```
Input/your_project_name/
├── raw_expression_mtx.csv # Raw expression matrix (cells × genes)
├── cell_meta.csv # Cell meta information (Cluster annotations)
├── cell_location.csv # Cell spatial coordinates
├── Databases.json # Ligand-Receptor-TF database
├── Ligs_list.json # Candidate Ligands
├── Recs_list.json # Candidate Receptors
├── TGs_list.json # Candidate Target Genes
```

## Running CCCvelo 

Edit run_CCCvelo.py if needed to set:

* base_path (your project root path) <br>
* project_name (your input folder name) <br>
* hyperparameters, includes `batch_size`, `hidden_dims`, `n_epochs`, `learning_rate`, etc. <br>

Then run

    python run_CCCvelo.py

The output files (trained model, velocity vectors, processed AnnData) will be saved under `Output/your_project_name/`.


# Demo
We here illustrate how to run CCCvelo using a demo dataset step-by-step. 


# Application of CCCvelo
To apply CCCvelo to any other ST datasets:

A detailed tutorial is provided in the `CCCvelo_tutorial.md` (https://github.com/SunXQlab/CCCvelo/blob/main/CCCvelo_tutorial.md), and you can follow it step-by-step to analyze your dataset.



