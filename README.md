# Genetic Algorithm Feature Selection for Scaffold-Aware Prediction of Dual JNK3/GSK3β Candidates

Can simple 2D molecular descriptors predict whether a molecule is a promising candidate for **two** Alzheimer's-relevant kinase targets at once, and still work on molecules with **completely new chemical scaffolds**?

This project builds a machine learning pipeline to answer that question. It compares descriptor sets and feature selection strategies across roughly 97,000 molecules, and uses a Genetic Algorithm to search a 626-descriptor space for the most informative subset.

## Key results

The final model is an **XGBoost classifier using 331 descriptors selected by a Genetic Algorithm**. It was evaluated on a scaffold-separated test set, meaning none of the test molecules share a core structure with the molecules used for training.

| Metric | Scaffold test set |
|---|---|
| MCC | 0.7086 |
| Balanced accuracy | 0.8680 |
| ROC-AUC | 0.9536 |
| Average precision | 0.9864 |
| Precision | 0.9487 |
| Recall | 0.9198 |
| F1 | 0.9340 |

A multi-output neural network trained on 100 of the selected descriptors predicted the continuous docking scores directly, reaching a scaffold-test **R² of 0.7968 for JNK3 and 0.8159 for GSK3β**.

## Why the scaffold split matters

In a random train/test split, very similar molecules often end up on both sides, which makes a model look better than it really is. Here, molecules were grouped by their Bemis-Murcko scaffold, and each scaffold was placed in only one partition. The results above therefore reflect performance on **unseen chemotypes**, which is much closer to how a model would be used in a real screening campaign.

## What this project covers

- Data cleaning, SMILES canonicalisation and duplicate consolidation
- Scaffold-separated 70:15:15 train/validation/test split
- RDKit and Mordred descriptor generation with training-only preprocessing
- Baseline regression and classification models
- Direct versus regression-derived dual-candidate classification
- XGBoost, Extra Trees and soft-voting ensembles
- mRMR feature selection
- Genetic Algorithm wrapper feature selection, with stability analysis across six random seeds
- XGBoost boosting-iteration, model-capacity and sensitivity analyses
- Multi-output neural-network regression in PyTorch
- SHAP-based model interpretation

## Dataset

The dataset was derived from the EVOSYNTH project and contained 100,000 molecules with precomputed docking scores for JNK3 and GSK3β. A molecule was labelled a **dual candidate** when both docking scores were ≤ −8.0.

After cleaning, the dataset contained 96,771 unique molecules (75,917 dual candidates and 20,854 non-dual candidates), split by scaffold into:

- Training: 67,502 molecules
- Validation: 14,569 molecules
- Test: 14,700 molecules

There is no scaffold overlap between the three partitions. The validation set was used for all model, feature, threshold and checkpoint selection. The test set was used for final evaluation, and later test-set analyses are exploratory follow-up work.

**The molecular dataset is not included in this repository**, because it was provided through the EVOSYNTH project and is not mine to redistribute.

Note that labels are derived from **docking scores**, which are computational estimates of binding. They are not experimental measurements of inhibition.

## Repository contents

| Folder or file | What it contains |
|---|---|
| Notebooks (`.ipynb`) | The full analysis, in the order listed below |
| `scripts/` | Genetic Algorithm scripts, including the repeated-seed runs |
| `results/` | Metrics, comparison tables, GA histories, selected descriptors and feature rankings |
| `requirements.txt` | Python package versions used |

Large files such as trained models, full prediction tables and processed descriptor matrices are not included because of their size.

## Notebook order

1. `data_audit_and_cleaning.ipynb`: cleans the raw docking data, canonicalises SMILES, consolidates duplicates and assigns labels
2. `scaffold_split.ipynb`: creates the scaffold-separated partitions
3. `molecular_descriptors.ipynb`: calculates 210 RDKit descriptors
4. `descriptor_preprocessing.ipynb`: imputation, zero-variance and correlation filtering, leaving 177 RDKit descriptors
5. `regression_models.ipynb`: Ridge, Random Forest and Histogram Gradient Boosting for each docking score
6. `classification_models.ipynb`: baseline dual-candidate classifiers
7. `direct_vs_indirect_classification.ipynb`: direct classification versus labels derived from regression predictions
8. `08_tree_based_classification_and_ensemble.ipynb`: XGBoost, Extra Trees, soft voting and chemical-space analysis
9. `09_mordred_mrmr_feature_selection.ipynb`: Mordred descriptors and mRMR feature subsets
10. `10_genetic_algorithm_feature_selection.ipynb`: GA feature selection compared with mRMR and the full Mordred set
11. `XGBoost_Best_Iteration_and_Model_Capacity.ipynb`: boosting iterations, model capacity, multi-seed GA stability and final evaluation
12. `Neural_Network_Regression_10000_Epochs.ipynb`: the final neural-network regression experiment
13. `Neural_Network_Regression_Double_Descent.ipynb`: an earlier exploratory capacity study, superseded by notebook 12

## Methods in more detail

### Molecular descriptors

**RDKit:** 210 descriptors were generated. Using the training set only, infinite values were treated as missing, missing values were median-imputed, zero-variance descriptors were removed, and one descriptor was removed from each pair with |r| > 0.95. This left 177 descriptors.

**Mordred:** 1,613 2D descriptors were generated with `ignore_3D=True`. Using the training set only, descriptors with more than 5% missing values were removed, remaining missing values were median-imputed, constant descriptors were removed, and highly correlated descriptors were filtered at |r| > 0.95. This left 626 descriptors.

### mRMR feature selection

Minimum Redundancy Maximum Relevance was applied to the training data at 20, 40, 60, 80, 100, 150, 200, 250 and 300 descriptors. The 300-descriptor subset was kept as the compact mRMR comparison.

### Genetic Algorithm feature selection

The GA searched the full 626-descriptor Mordred space, representing each candidate subset as a 626-bit chromosome.

| Setting | Value |
|---|---|
| Population | 40 |
| Generations | 30 |
| Tournament size | 3 |
| Two-point crossover probability | 0.80 |
| Individual mutation probability | 0.20 |
| Per-bit mutation probability | 1/626 |
| Fitness | Validation MCC |
| Threshold during search | 0.50 |

Full runs were completed with seeds 7, 21, 42, 84, 99 and 123. Stability was assessed using validation MCC, selected-feature counts and pairwise Jaccard similarity. The run with the highest validation MCC selected 331 descriptors and was used for the final model.

### Final XGBoost model

Validation log-loss was monitored for up to 3,000 boosting rounds with early-stopping patience of 200. The minimum occurred at zero-indexed iteration 1,412, so the final model was refitted on combined training and validation data with 1,413 trees, learning rate 0.05, maximum depth 6, minimum child weight 3, subsample 0.80, column subsample 0.80, L1 regularisation 0.10, L2 regularisation 1.00, balanced sample weights and a 0.50 classification threshold.

### Neural-network regression

The 331 GA-selected descriptors were reduced to 100 using training-only mutual-information ranking. Features and targets were standardised using the training data. Seven one-hidden-layer PyTorch networks (10, 50, 100, 250, 500, 1,000 and 2,000 neurons) predicted both docking scores together, using ReLU, mean squared error loss, Adam (learning rate 0.001), batch size 128 and 10,000 full epochs, with no dropout, weight decay or early stopping. The best checkpoint for each architecture was chosen by validation loss, architecture selection used validation data only, and the chosen model was then evaluated on the scaffold test set.

## Software

Python 3.11.14 with RDKit 2024.9.2, scikit-learn 1.8.0, Mordred Community 2.0.7, mRMR Selection 0.2.8, XGBoost 3.2.0, DEAP 1.4.4, NumPy 1.26.4, pandas 2.3.3, SHAP 0.51.0 and PyTorch 2.9.1.

Classical machine-learning analyses were run on a JupyterHub server. Neural-network training used an NVIDIA A100 GPU.

## Reproducibility

The notebooks contain all modelling parameters used. Some stages are computationally expensive, especially Mordred descriptor generation, the repeated GA runs and the 10,000-epoch neural-network training, so the key result tables are included in `results/` for inspection without rerunning those steps.

## About

**Author:** Tanya Sood

This work was carried out as my MSc research project in Artificial Intelligence for Drug Discovery at Queen Mary University of London (2026), supervised by Dr Amin Alibakhshi.
