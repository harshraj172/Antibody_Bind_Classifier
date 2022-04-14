# Bind_Classifier

## AGENDA 01:
AIM : To classify the Binding Energy of an Antibody with an Antigen

### Overview
1. Use A pre trained Transformer to get feature representation of the protein VH sequence for Antibody.

2. This feature representation is given as input along with the contact map of the 3D-Structure to a GNN model for Antibody.

3. Similarly, initialise a GNN for Antigen.

4. Concatenate the outputs from both the model(of Antibody and Antigen).

5. Train the model for classifying the classes of Binding Energy which are made based on pre assumed thresholds(thresholding ref.:https://www.biorxiv.org/content/10.1101/2021.07.08.451480v1).

### USAGE:
Clone this repo using 

`git clone https://github.com/harshraj172/Pretrain_ProtGNN.git`

Download requirements.txt using

`pip install -r requirements.txt`

Clone Pretrain_ProtGNN using

`git clone https://github.com/harshraj172/Pretrain_ProtGNN.git`


The train script offers many options; here are the most important ones

`python train.py` 
