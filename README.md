# Beyond Additivity: The Challenge of Predicting Epistatic Effects in Proteins  

Authors:  
Anastasia Kolchina1,2, Igors Dubanevics3, Fyodor A. Kondrashov3, Olga V. Kalinina1,2,4  

Affiliations:  
1 Research Group Drug Bioinformatics, Helmholtz Institute for Pharmaceutical Research Saarland (HIPS), Helmholtz Centre for Infection Research (HZI), 66123 Saarbrücken, Germany  
2 Center for Bioinformatics, Saarland University, Saarbrücken, Germany  
3 Evolutionary and Synthetic Biology Unit, Okinawa Institute of Science and Technology Graduate University, 1919-1 Tancha, Onna-son, Okinawa 904-0495, Japan  
4 Medical Faculty, Saarland University, Homburg, Germany  

---

## Overview

This repository contains the code accompanying our Nature Methods – Analysis submission:

Beyond Additivity: The Challenge of Predicting Epistatic Effects in Proteins

We benchmark 95 zero-shot variant effect prediction (VEP) models from ProteinGym for their ability to predict epistatic effects — cases where the combined impact of multiple mutations deviates from the sum of individual effects.

While zero-shot models perform reasonably on non-epistatic variant combinations, their predictive power drops substantially for strongly epistatic variants.

---

## Repository structure

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
│   ├── data_processing.py
│   ├── epistasis_detection.py
│   └── model_evaluation.py
│
├── environment.yml
├── LICENSE
└── README.md

---

## Installation

git clone https://github.com/kalininalab/epistasis_proteingym.git  
cd epistasis_proteingym  

conda env create -f environment.yml  
conda env create -f external/tsuboyama/protease-pipeline.yml  
conda activate epi_env  
python -m ipykernel install --user --name=epi_env --display-name "Python (epi_env)"  
pip install -e .

---

## Data

bash scripts/download_data.sh  
python scripts/prepare_data.py  

---

## Reproducing the analysis

Run notebooks in order using kernel epi_env:

01_datasets_exploration.ipynb  
02_epistasis_detection.ipynb  
03_model_performance.ipynb  

---

## Tsuboyama module

Reproduces ΔG reconstruction and thresholds from Tsuboyama et al.  
Used automatically within the analysis notebooks.  
See external/tsuboyama for attribution and license.

---

## License

MIT License.

---

## Citation

Kolchina A., Dubanevics I., Kondrashov F.A., Kalinina O.V.  
Beyond Additivity: The Challenge of Predicting Epistatic Effects in Proteins, Nature Methods – Analysis, 2025