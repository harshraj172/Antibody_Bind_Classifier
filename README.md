# Bind_Classifier

Experiments to run:

1. Run Bind Classifier with structure representations of Antibody & Antigen only:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq False --use_struct True --nlayers_range "[3, 3]" --ndegrees_range "[3, 3]" --nchannels_range "[8, 12]" --nnlayers_range "[1, 2]" --div_range "[4, 4]" --batch_size 16

2. Run Bind Classifier with structure+sequence representations of Antibody & Antigen:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq True --use_struct True --nlayers_range "[3, 3]" --ndegrees_range "[3, 3]" --nchannels_range "[8, 12]" --nnlayers_range "[1, 2]" --div_range "[4, 4]" --batch_size 16

3. Run Bind Classifier with sequence representations of Antibody & Antigen only:

> python tuning_exp_se3.py --entity "maximentropy" --ntrials 20 --device_ids "[0, 1]" --use_seq True --use_struct False --batch_size 16
