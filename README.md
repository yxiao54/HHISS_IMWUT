# HHISS_IMWUT
Code for Human Heterogeneity Invariant Stress Sensing 
## Source code
- `losses.py` defines the loss functions for the HHISS model, as well as for IRM, Vrex, DRO, and others.  
- `main.py` contains the code to train the HHISS model.  
- `models.py` defines the PyTorch model used for training.  
- `myutils.py` contains utility functions for change score normalization.  
- `overparameterized.py` includes the code to train the overparameterized IRM model.  
- `preprocess.py` includes the code to extract features from raw signals.  
- `prune_utils.py` provides utility functions for subject-wise pruning.  
- The `stats_analysis` folder contains the Jupyter Notebook `lmm_oud.ipynb`, which analyzes extracted features using a Linear Mixed Effects Model (LMM) to perform pairwise comparisons between datasets and identify significant features that differentiate between stress and calm tasks.  
- The `ckpt` folder holds the checkpoints.
  
## How to Run
To extract the features from raw signals
```commandline
python preprocess.py
```

To train the overparameterized IRM model
```commandline
python overparameterized.py
```

To train the HHISS model
```commandline
python main.py
```
