# Bind_Classifier

Experiments to run:

1. Run Bind Classifier with structure representations of Antibody & Antigen only:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq False --use_struct True --nlayers_range "[2, 6]" --ndegrees_range "[2, 6]" --nchannels_range "[8, 16]" --nnlayers_range "[1, 4]" --div_range "[2, 4]" --head_range "[1, 4]"

2. Run Bind Classifier with structure+sequence representations of Antibody & Antigen:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq True --use_struct True --nlayers_range "[2, 6]" --ndegrees_range "[2, 6]" --nchannels_range "[8, 16]" --nnlayers_range "[1, 4]" --div_range "[2, 4]" --head_range "[1, 4]"

3. Run Bind Classifier with sequence representations of Antibody & Antigen only:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq True --use_struct False
