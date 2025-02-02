# HHISS_IMWUT
Code for Human Heterogeneity Invariant Stress Sensing 
## Source code
main.py contains the code to train the HHISS model.
losses.py defines the loss functions for the HHISS model, as well as for IRM, Vrex, DRO, and others.
prune_utils.py provides utility functions for subject-wise pruning.
myutils.py contains utility functions for change score normalization.
model.py defines the PyTorch model used for training.
overparameterized.py includes the code to train the overparameterized IRM model.

The ckpt folder holds the checkpoints for the pre-trained HHISS model and the overparameterized IRM model, as outlined in Algorithm 1 of the main paper.

## How to Run
To extract the features from raw signals
```commandline
python preprocess.py
```

To train the HHISS model
```commandline
python main.py
```
