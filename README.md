# Stacked Based Ensemble Technique
The deep learning ensemble technique combined with CNN for human breast cancer prognosis prediction.

# References

Our manuscipt titled with "A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data" has been accepted by IEEE/ACM Transactions on Computational Biology and Bioinformatics. If you find MDNNMD useful in your research, please consider citing:

Sun, D., Wang, M., & Li, A. (2018). A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data. IEEE/ACM Transactions on Computational Biology and Bioinformatics.
# Requirements
[python 3.6](https://www.python.org/downloads/)


[TensorFilow 1.12](https://www.tensorflow.org/install/)


[scikit-learn 0.20.0](http://scikit-learn.org/stable/)


[matplotlib 3.0.1](https://matplotlib.org/users/installing.html)


[Weka 3.8.3](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)

# Usage
CNNDNN.py

STACKED_RF_HIDDEN.model

ttest.py

# Process to execute the Stacked-based ensemble model.

=>  Run the CNNDNN.py for training CNN-Clinical, CNN-CNA, CNN-Expr, DNN-Clinical, DNN-CNA and DNN-Expr.

=>  After successfull run you will get the stacked features saved in the file stacked_metadata.csv.

=>  Convert the stacked_metadata.csv file to stacked_metadata.arrf file.

=>  Load the STACKED_RF_HIDDEN.model in weka 3.8.3 and pass the stacked feature to get the final prediction output.

=>  Once final prediction has been made use ttest.py to perform statistical significance test.




