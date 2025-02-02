# HHISS_IMWUT
Code for Human Heterogeneity Invariant Stress Sensing 
## Source code
main.py contains the code to train the HHISS model.<br/>
losses.py defines the loss functions for the HHISS model, as well as for IRM, Vrex, DRO, and others.<br/>
prune_utils.py provides utility functions for subject-wise pruning.<br/>
myutils.py contains utility functions for change score normalization.<br/>
model.py defines the PyTorch model used for training.<br/>
preprocess.py includes the code to extract feature from raw signals.<br/>
overparameterized.py includes the code to train the overparameterized IRM model.<br/>

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
