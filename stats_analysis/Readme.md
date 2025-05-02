# Stress Score Analysis Notebook

This Jupyter Notebook (`lmm_oud.ipynb`) is used for analyzing extracted features using a Linear mixed effect model (LMM) for pair-wise comparisons between different datasets and find significant features which show stress and calm task specific differences.

## Setup Instructions

Before running the notebook, please follow these steps:

1. **Place your data file:**
   - Move the compiled data file named `all_data.csv` into the same directory as the `lmm_oud.ipynb` notebook.

2. **Create a results directory:**
   - Create a folder named `results` in the same directory. This is where the summary files and significant feature outputs will be saved.
   - You can create it manually or use the following terminal command:
     ```bash
     mkdir results
     ```

## Output

- Summary CSV files and other result artifacts will be saved inside the `results/` folder upon running the notebook.

## Requirements

Ensure the following Python packages are installed:
- `pandas`
- `matplotlib`
- `numpy`
- `statsmodels` 

You can install all required packages with:
```bash
pip install pandas matplotlib numpy statsmodels
