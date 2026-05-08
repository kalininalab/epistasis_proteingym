# Beyond Additivity: Zero-shot Methods Cannot Predict Impact of Epistasis on Protein Properties and Function 

**Authors:**  
Anastasia Kolchina¹˒², Igors Dubanevics³, Fyodor A. Kondrashov³, Olga V. Kalinina¹˒²˒⁴  

**Affiliations:**  
¹ Research Group Drug Bioinformatics, Helmholtz Institute for Pharmaceutical Research Saarland (HIPS), Helmholtz Centre for Infection Research (HZI), 66123 Saarbrücken, Germany  
² Center for Bioinformatics, Saarland University, Saarbrücken, Germany  
³ Evolutionary and Synthetic Biology Unit, Okinawa Institute of Science and Technology Graduate University, 1919-1 Tancha, Onna-son, Okinawa 904-0495, Japan  
⁴ Medical Faculty, Saarland University, Homburg, Germany  

---

## 📖 Overview

This repository contains the code accompanying our preprint:

> **[Beyond Additivity: Zero-shot Methods Cannot Predict Impact of Epistasis on Protein Properties and Function](https://www.biorxiv.org/content/10.64898/2026.02.17.706292v1)**

We benchmark **95 zero-shot variant effect prediction (VEP) models** from [ProteinGym](https://proteingym.org/) for their ability to predict epistatic effects — cases where the combined impact of multiple mutations deviates from the sum of individual effects.

While zero-shot models perform reasonably well on non-epistatic variant combinations, their predictive power drops substantially for strongly epistatic variants, highlighting current limitations of unsupervised protein language models in multi-mutation regimes.

---

## ❗️ ProteinGym epistatic sequences: fast access

If you want to evaluate your model on the epistatic sequences we detected in ProteinGym, please use the tables from [```results/epistatic```](https://github.com/kalininalab/epistasis_proteingym/tree/main/results/epistatic). The column "epistatic" indicates whether the sequence is considered epistatic or not (multimutants only).

---

## 📂 Repository structure

```text
.
├── external/
│   └── tsuboyama/
│       ├── additive_model.py
│       ├── protease-pipeline.yml
│       └── README.md
│
├── notebooks/
│   ├── 01_datasets_exploration.ipynb
│   ├── 02_epistasis_detection.ipynb
│   └── 03_model_performance.ipynb
│
├── results/
│   ├── figures/
│   │   ├── main/
│   │   └── supplementary/
│   └── tables/
│       ├── final/
│       │   ├── main/
│       │   └── supplementary/
│       └── intermediate/
│
├── scripts/
│   ├── download_data.sh
│   └── prepare_data.py
│
├── src/
│   ├── models/
│   │   ├── linear_regression.py
│   │   └── mlp.py
│   ├── analysis.py
│   ├── constants.py
│   ├── data_processing.py
│   ├── paths.py
│   ├── plotting.py
│   └── utils.py
│
├── environment.yml
├── LICENSE
└── README.md
```

---

## 🚀 Installation

Clone the repository:


```text
git clone https://github.com/kalininalab/epistasis_proteingym.git
cd epistasis_proteingym
```

Create the conda environments:

```text
conda env create -f environment.yml
conda env create -f external/tsuboyama/protease-pipeline.yml
conda activate epi_env
python -m ipykernel install --user --name=epi_env --display-name "Python (epi_env)"
pip install -e .
```

---

## 📊 Data

Download datasets:

```text
bash scripts/download_data.sh
```

Prepare data:

```text
python scripts/prepare_data.py
```

---

## ⚙️ Reproducing the analysis

Run notebooks in order using kernel ```epi_env```:

	1.	01_datasets_exploration.ipynb

	2.	02_epistasis_detection.ipynb

	3.	03_model_performance.ipynb

---

```text
results/tables/final
```

Tables used directly in the manuscript and supplementary material.

```text
results/tables/intermediate
```

Intermediate results generated during the pipeline (fully reproducible).

---

## 🔬 Tsuboyama module (external)

This module reproduces ΔG reconstruction and epistasis thresholds introduced in Tsuboyama et al.

Code is adapted from the original publication repository
(see ```external/tsuboyama/README.md``` for attribution and license).

It is used automatically within the analysis notebooks.

---

## 📄 License

MIT License — see LICENSE.

---

## ✏️ Citation

If you use this work, please cite:

Kolchina, A., Dubanevics, I., Kondrashov, F. A. & Kalinina, O. V. Beyond additivity: zero-shot methods cannot predict impact of epistasis on protein properties and function. 2026.02.17.706292 Preprint at https://doi.org/10.64898/2026.02.17.706292 (2026).


---
