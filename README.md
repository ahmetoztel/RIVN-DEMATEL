# RIVN-DEMATEL: Rough Interval-Valued Neutrosophic DEMATEL Method

This repository contains a complete Python implementation of the **Rough Interval-Valued Neutrosophic DEMATEL (RIVN-DEMATEL)** method. The method integrates rough set theory and interval-valued neutrosophic logic to evaluate complex interrelationships among criteria under uncertainty and indeterminacy.

## 🧠 Method Overview

RIVN-DEMATEL is designed to:

- Handle linguistic expert judgments using **Single-Valued Neutrosophic Numbers (SVNN)**
- Convert SVNNs into **Interval-Valued Neutrosophic Numbers (IVNN)** using **rough approximations**
- Aggregate multiple expert opinions with the **IVNNWA operator**
- Convert IVNNs into crisp scores using a **deneutrosophication operator**
- Construct the **crisp direct-relation matrix**, normalize it, and calculate the **total relation matrix**
- Determine **cause and effect groups** via DEMATEL’s **D + R** and **D – R** metrics
- Generate a **cause–effect diagram**

## 📁 Files

- `RIVN_DEMATEL.py`: Main Python script for all steps of the methodology
- `input.xlsx`: Sample input file (expert opinions in linguistic scale)
- `RIVN_DEMATEL_Results.xlsx`: Output Excel file with all matrices and analysis results
- `DEMATEL_Cause_Effect.png`: Cause–effect diagram showing factor classifications

## 🔧 Requirements

- Python 3.8+
- Packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `openpyxl`
  - `tkinter`

Install required packages via pip:

```bash
pip install pandas numpy matplotlib openpyxl
🚀 How to Run
Run the script RIVN_DEMATEL.py.

A file dialog will prompt you to select an Excel file with expert opinions.

The script will process the input and automatically generate:

All intermediate and final matrices

Summary statistics

Excel output

A visual cause–effect diagram

📊 Input Format
The input Excel file should contain expert evaluations in a linguistic scale (0–4 or labels like "VU", "MI", etc.), arranged as square matrices stacked vertically.

Each matrix represents one expert.

📄 Output
All intermediate results:

SVNN matrices (T, I, F)

Transformed IVNN matrices

Aggregated IVNN matrix (in matrix form)

Crisp, normalized, and total relation matrices

D, R, D+R, D–R analysis

A 2D DEMATEL cause–effect plot

📚 Citation
If you use this code in your research, please cite the corresponding article or contact the author.

Developed by: Ahmet Öztel
Affiliation: Bartın University
Contact: aoztel@bartin.edu.tr
Year: 2025
