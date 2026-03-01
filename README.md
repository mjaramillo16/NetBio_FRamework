# NetBio Generic Framework: Hybrid Modeling of Regulatory Networks (Neural ODEs + Graphs)

**Thesis Title:** Dynamic Modeling of Biological Systems: An Automated Framework based on Graph-Informed
Neural ODEs.

**Author:** Maria Leandra Guateque Jaramillo  
**Institution:** Pontifical Catholic University of Rio de Janeiro (PUC-Rio)

---

## Overview

This repository contains the official implementation of the **NetBio Framework**, a Hybrid Computational Architecture that integrates Biological Knowledge Graphs (KEGG) with Continuous-Time Deep Learning (Neural Ordinary Differential Equations).

The primary objective of this framework is to reconstruct the true continuous temporal dynamics of Gene Regulatory Networks (GRNs) from static or time-series transcriptomic data. By imposing strict, real-world topological constraints (Hadamard Masking) onto the differential vector field, the model mathematically enforces biological plausibility, stabilizes gradient convergence, and pioneers dynamic causal inference via Virtual Twins (Explainable AI - XAI).

---

## Key Contributions

1. **Agnostic Scalability (O(N)):** A single codebase seamlessly models Oncology (p53), Metabolism (Insulin Resistance), and Evolutionary Conservation (Cell Cycle), adapting dynamically to the input data.

2. **Grey-Box Paradigm:** Transforms "Black-Box" deep learning into interpretable models by directly comparing topologies derived from prior biological knowledge (KEGG) against purely statistical graph inference algorithms (Graphical Lasso).

3. **Dynamic Causal Inference (Virtual Twins):** Operationalizes true XAI by simulating targeted *in silico* biological perturbations (Gene Knockdowns) over continuous time to predict the downstream collapse of pathological networks.

4. **Unsupervised Biological Validation (GSEA):** Automatically applies Louvain community detection to the network topology and annotates the resulting modules via the Enrichr API (Gene Ontology Biological Process).

5. **Bulletproof Engineering:** Fully automated ETL pipeline, real-time KEGG API graph extraction, and a strict **Network Auditor** that aborts compilation if biological parsimony (density > 30%) is violated.

---

## Repository Structure

```text
NetBio_Framework/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- networks/
|
|-- experiments/
|   |-- run_case_a_cancer.py
|   |-- run_case_b_diabetes.py
|   |-- run_case_c_dual.py
|   |-- Test_Gemeos_real_data.py
|   |-- robustness_test.py
|
|-- results/
|   |-- models/
|   |-- thesis_results_*/
|
|-- src/
|   |-- etl/
|   |-- topology/
|   |-- modeling/
|   |-- analysis/
|
|-- reset_and_run.py
|-- requirements.txt
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/mjaramillo16/NetBio_FRamework.git
cd NetBio_Framework
```

### Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> Ensure `gseapy` is installed for automated Gene Ontology annotation.

---

## Reproduction Pipeline (Thesis Results)

To ensure full reproducibility of the thesis results, use the global orchestrator.

### Option 1: One-Click Global Orchestrator (Recommended)

```bash
python3 reset_and_run.py
```

This will automatically:

- Clean environment  
- Download raw datasets  
- Build KEGG biological graphs  
- Train all Neural ODE models  
- Execute benchmarking suite  
- Simulate Virtual Twins  
- Generate publication-ready plots  

Outputs are saved in:

```
results/thesis_results_[TIMESTAMP]/
```

---

### Option 2: Step-by-Step Execution

#### Stage 1 — Model Fabrication (ETL & Training)

```bash
python3 experiments/run_case_a_cancer.py
python3 experiments/run_case_b_diabetes.py
python3 experiments/run_case_c_dual.py
```

#### Stage 2 — Global Evaluation & XAI Knockdown

```bash
python3 experiments/run_benchmark_all.py
```

This stage:

- Evaluates trained models  
- Generates Rolling CV metrics  
- Performs in silico causal knockdowns  
- Renders annotated heatmaps (Louvain + GSEA)

---

## Comparative Benchmark (Ablation Study)

The framework evaluates six architectural paradigms:

1. **Linear Regression** — Baseline statistical mapping  
2. **Bayesian Networks** — Static probabilistic graphical causal inference baseline  
3. **Standard MLP** — Unconstrained deep learning  
4. **Pure Neural ODE** — Continuous dynamics, fully connected  
5. **Glasso Neural ODE** — Continuous dynamics + Graphical Lasso topology  
6. **Hybrid Neural ODE (Ours)** — Continuous dynamics + KEGG-constrained topology  

---

## Methodology Workflow

### 1. Data Engineering

- Automated extraction from NCBI GEO  
- Affymetrix probe summarization to Gene Symbols  
- Log2 normalization  
- Min-Max scaling  

---

### 2. Topological Gatekeeping

- Directed pathway extraction via bioservices KEGG API  
- Network Auditor aborts compilation if:

```
Density(A) > 30%
```

---

### 3. Hybrid Dynamics

The system dynamics are defined as:

$$
\frac{d\mathbf{x}}{dt} = \text{DNN}(\mathbf{x}(t), \theta) \odot \mathbf{A}
$$

---

### 4. Causal Perturbation (Virtual Twins)

Critical regulators (e.g., TP53, AKT3) are silenced at $t=0$ to simulate downstream divergence and validate true causal structure.

---

### 5. Unsupervised XAI

- Louvain community detection  
- Enrichr API (GO Biological Process 2023) annotation  

---

## Citation

If you use this framework, please cite:

Guateque Jaramillo, Maria Leandra. "Dynamic Modeling of Biological Systems: An Automated Framework based on Graph-Informed Neural ODEs". 2026. Thesis (PhD in Informatics). Pontifical Catholic University of Rio de Janeiro (PUC-Rio).

---

## License

MIT License
````
