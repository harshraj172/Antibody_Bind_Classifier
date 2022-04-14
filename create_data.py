import json
import pandas as pd
from prody import *

def get_data(pdb_id, chain):
    structure = parsePDB(pdb_id, chain=chain)
    
    dict_ = {
                'coords': {'N': [], 'CA': [], 'C': [], 'O': []},
                'seq': ''
            }
    
    if structure is not None:
        for res in structure.iterResidues():
            if res['N']!=None and res['CA']!=None\
               and res['C']!=None and res['O']!=None:
                dict_['coords']['N'].append(list(res['N'].getCoords()))
                dict_['coords']['CA'].append(list(res['CA'].getCoords()))
                dict_['coords']['C'].append(list(res['C'].getCoords()))
                dict_['coords']['O'].append(list(res['O'].getCoords()))
                
                dict_['seq'] += res.getSequence()[0]
    return dict_

def save_json(X, path):
    with open(path, "w") as outfile:
        json.dump(X, outfile)
        
def main():
    summary_df = pd.read_csv(FLAGS.summary_file_path, sep='\t')
    summary_df = sabdab_df.dropna(subset=['Hchain', 'antigen_chain'])
    summary_df = sabdab_df.reset_index(drop=True)

    XAg, XAb = [], []
    for i in range(len(summary_df)):
        pdb_id = summary_df['pdb'][i]
        Ag_chain = ''.join(summary_df['Hchain'][i].split('|'))
        Ab_chain = ''.join(summary_df['antigen_chain'][i].split('|'))

        dict_Ab = get_data(pdb_id, Ab_chain)
        dict_Ag = get_data(pdb_id, Ag_chain)

        if dict_Ab['coords']['CA']!=[] and dict_Ag['coords']['CA']!=[]: 
            XAb.append(get_data(pdb_id, Ab_chain))
            XAg.append(get_data(pdb_id, Ag_chain))

    XAb_train, XAb_val, XAb_test = XAb[:int(0.8*len(XAb))], XAb[int(0.8*len(XAb)):int(0.9*len(XAb))], XAb[int(0.9*len(XAb)):]
    XAg_train, XAg_val, XAg_test = XAg[:int(0.8*len(XAg))], XAg[int(0.8*len(XAg)):int(0.9*len(XAg))], XAg[int(0.9*len(XAg)):]
            
    # Training Data 
    save_json(XAb_train, FLAGS.train_data_Ab)
    save_json(XAg_train, FLAGS.train_data_Ag)
    

    # Trainig Data 
    save_json(XAb_val, FLAGS.val_data_Ab)
    save_json(XAg_val, FLAGS.val_data_Ag)
    

    # Trainig Data 
    save_json(XAb_test, FLAGS.test_data_Ab)
    save_json(XAg_test, FLAGS.test_data_Ag)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()\
    
    parser.add_argument('--summary_file_path', type=str, default=False)
    
    parser.add_argument('--train_data_Ab', type=str, default=True)
    parser.add_argument('--train_data_Ag', type=str, default=True)
    
    parser.add_argument('--val_data_Ab', type=str, default=True)
    parser.add_argument('--val_data_Ag', type=str, default=True)
    
    parser.add_argument('--test_data_Ab', type=str, default=True)
    parser.add_argument('--test_data_Ag', type=str, default=True)
    
    FLAGS, UNPARSED_ARGV = parser.parse_known_args()
    
    main()
