# Stacked Based Ensemble Technique
The deep learning ensemble technique combined with CNN for human breast cancer prognosis prediction.

# References

Our manuscipt titled with "Multi-modal classification for human breast cancer prognosis prediction: Proposal of deep-learning based stacked ensemble model" has been accepted at IEEE/ACM Transactions on Computational Biology and Bioinformatics.

# Requirements
[python 3.6](https://www.python.org/downloads/)


[TensorFilow 1.12](https://www.tensorflow.org/install/)

[keras 2.2.4](https://pypi.org/project/Keras/)


[scikit-learn 0.20.0](http://scikit-learn.org/stable/)


[matplotlib 3.0.1](https://matplotlib.org/users/installing.html)



# Usage
cnn_clinical.py

cnn_cnv.py

cnn_exp.py

STACKED_RF_HIDDEN.model

ttest.py

# Process to execute the Stacked-based ensemble model.

=>  Run cnn_clinical.py, cnn_cnv.py, cnn_exp.py for training individual CNNs for clinical, CNA and gene-expression data.

=>  After successfull run you will get the hidden features in three different csv files : clinical_metadata.csv, cnv_metadata.csv and exp_metadata.csv

=> Combine all the hidden features of different modalities to form stacked features : stacked_metadata.csv

=>  run RF.py and pass the stacked feature(stacked_metadata.csv) as input to get the final prediction output.

=>  Once final prediction has been made use ttest.py to perform statistical significance test.




