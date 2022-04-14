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
    summary_df = pd.read_csv('data/SabDab/sabdab_summary_all.tsv', sep='\t')
    summary_df = sabdab_df.dropna(subset=['Hchain', 'antigen_chain'])
    summary_df = sabdab_df.reset_index(drop=True)

    X_AG, X_AB = [], []
    for i in range(len(summary_df)):
        pdb_id = summary_df['pdb'][i]
        AG_chain = ''.join(summary_df['Hchain'][i].split('|'))
        AB_chain = ''.join(summary_df['antigen_chain'][i].split('|'))

        dict_AB = get_data(pdb_id, AB_chain)
        dict_AG = get_data(pdb_id, AG_chain)

        if dict_AB['coords']['CA']!=[] and dict_AG['coords']['CA']!=[]: 
            X_AB.append(get_data(pdb_id, AB_chain))
            X_AG.append(get_data(pdb_id, AG_chain))

    save_json(X_AB, 'X_AB.json')
    save_json(X_AG, 'X_AG.json')
    

if __name__ == '__main__':
    main()
