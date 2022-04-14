# Bind_Classifier

### USAGE:
Clone this repo using 

```git clone https://github.com/harshraj172/Bind_Classifier.git```

##### Download requirements.txt either manually or using

```pip install -r requirements.txt```

##### Data Creation:
The Bind Classifier will be trained on SabDab dataset(http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/)

1. Download the Summary File from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/

2. Run the below code which saves two json, of Antibody and Antigen separately.

  ```python Bind_classifier.create_data.py --summary_file_path 'path to summary file' --train_data_folder 'folder to save training data' --val_data_folder 'folder to save validation data' --test_data_folder 'folder to save testing data'```

  Antibody json(X_Ab_train.json) 

  Stores the coordinates of the N, CA, C, O atoms for every residue of the Antibody chain. It also stores the VH sequence under the key “seq”.

  Antigen json(X_Ag_train.json) 

  Stores the coordinates of the N, CA, C, O atoms for every residue of the Antigen chain. It also stores the protein sequence under the key “seq”.

3. Training:
    The Bind Classifier is written to be trained on 3 types of inputs - Protein structure, Protein sequence, Protein structure+Protein sequence.
    Protein structure

    To train the model you need to run 

    Protein Structure:

    ```python Bind_Classifier/train.py --use_seq False --use_struct True --train_data_dir 'path to train directory' --val_data_dir 'path to validation directory' --test_data_dir 'path to test directory'```

    Protein Sequence: 

    ```python Bind_Classifier/train.py --use_seq True --use_struct False --train_data_dir 'path to train directory' --val_data_dir 'path to validation directory' --test_data_dir 'path to test directory'```

    Protein Sequence + Protein Structure"

    ```python Bind_Classifier/train.py --use_seq True --use_struct True --train_data_dir 'path to train directory' --val_data_dir 'path to validation directory' --test_data_dir 'path to test directory'```
