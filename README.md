# CCCvelo: Decoding dynamic cell-cell communication-driven cell state transitions in spatial transcriptomics

<p align="center">
  <img src="https://github.com/SunXQlab/CCCvelo/blob/main/fig1.framework.png">
</p>

CCCvelo is a computational framework designed to reconstruct CCC-driven CST dynamics by jointly optimizing a dynamic CCC signaling network and a latent cell-state transitions (CST) clock through a multiscale, nonlinear network kinetics model. CCCvelo can estimated RNA velocity, cell pseudotime, pseudo-temporal dynamics of TGs’ expressions or TFs’ activities, and the cell state-specific multilayer signaling network of CCC. These functionalities enable the reconstruction of spatiotemporal trajectories of cells while simultaneously capturing dynamic cellular communication driving CST. CCCvelo employs several visualization strategies to facilitate the analysis of CCC-driven CST dynamics. These visualizations mainly include velocity streamlines illustrating CST trajectories, heatmap visualizations of gene expression and TF activity along pseudotime, and multilayer network plots of CCC displaying the signaling paths from upstream LR pairs to TFs and then to the downstream TGs.

The main features of CCCvelo are：

* (1) the reconstruction of spatiotemporal dynamics of CCC-regulated CSTs within a spatial context <br>
* (2) quantitative ordering of cellular progression states through velocity vector field embedding <br>
* (3) the identification of dynamic rewiring of CCC signaling <br>

## Installation

Create a separate conda environment for version control and to avoid potential conflicts. Please install the corresponding version of the package according to the 'requirements.txt', then the package CCCvelo can be directly used.
```
conda create -n cccvelo python=3.8.10
conda activate cccvelo
pip install -r requirements.txt
```

## Data Preparation

Before running CCCvelo, you need using `select_LRTG.R` function to select candidate ligands, receptors, and feature genes from the expression data, and then save the result into the input files under the path `data/precessed/`. The input files include:

```
data/precessed/
├── raw_expression_mtx.csv # Raw expression matrix (cells × genes)
├── cell_meta.csv # Cell meta information (Cluster annotations)
├── cell_location.csv # Cell spatial coordinates
├── Databases.json # Ligand-Receptor-TF database
├── Ligs_list.json # Candidate Ligands
├── Recs_list.json # Candidate Receptors
├── TGs_list.json # Candidate Target Genes
```

# Demo

We provide a step-by-step demonstration of how to run CCCvelo using a sample dataset, available in the `Demo` directory. The demo dataset can be download from https://www.dropbox.com/s/c5tu4drxda01m0u/mousebrain_bin60.h5ad?dl=0

The demonstration includes the following files:

* '0_preprocess_MouseBrain.ipynb' contains the code for preprocessing the raw dataset and exporting formatted inputs for use in R <br>
* '1_select_LRTG.R' prepares the inputs for CCCvelo model <br>
* '2_runCCCvelo_on_MouseBrainCortex.ipynb' provides the full pipeline to run CCCvelo on the mouse cortex data <br>

# Application of CCCvelo

To apply CCCvelo to any other ST datasets:

A detailed tutorial is provided in the `CCCvelo_tutorial.md` (https://github.com/SunXQlab/CCCvelo/blob/main/CCCvelo_tutorial.md), and you can follow it step-by-step to analyze your dataset.



