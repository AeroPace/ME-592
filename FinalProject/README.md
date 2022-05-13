# ME-592 Final Project

The results documented in the "Neural Network Architecture Effects on Deep Learning Topology Optimization" paper was done to further the existing studies conducted by Sosnovik et al. in the Neural Network for Topology Optimization.

The results were developed with Google Colaboratory environments with the FinalProject.ipynb and FinalProject_Autoencoder.ipynb python notebooks.

The autoencoder notebook utilized an additional variational auto-encoder in the neural network architecture incorporated in the source code updates in the nn4topopt-master-autoencoder.zip.

In addition, various additional models were developed and are saved in ./NNModels. In total, an additional DNN comparable model with and without the pooling and the same CNN architecture without the pooling. Utilization of these models requires overwritting the conv_model.py in the source code in the Google Colaboratory session note-book being executed from the nn4topopt-master.zip file.


REFERENCES:

TOP4040 dataset reference: https://yadi.sk/d/1EE7UdYJChIkQQ 

Original NN4TOPOPT Work referenced: https://github.com/ISosnovik/nn4topopt

top dataset generator: https://github.com/ISosnovik/top
