"""
Prerequisite:
- The first column of the dataset should be SMILES representation.
"""

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

class SelfiesGenerator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.smiles_lists = list(self.df.iloc[:, 0])
        self.selfies_lists = [sf.encoder(smiles) for smiles in self.smiles_lists]
        self.properties = self.df.iloc[:, 1:] if self.df.shape[1] > 1 else None
        self.alphabet = list(sorted(sf.get_alphabet_from_selfies(self.selfies_lists).union({'[nop]'})))
        self.pad_to_len = max(sf.len_selfies(s) for s in self.selfies_lists)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
        self.number_of_types_of_alphabet = len(self.symbol_to_idx)

    def generate_SELFIES_fingerprint_df(self):
        selfies_fp_temp = np.zeros([self.df.shape[0], self.pad_to_len * len(self.symbol_to_idx)])
        
        for index, selfies in enumerate(self.selfies_lists):
            one_hot = sf.selfies_to_encoding(
            selfies=selfies,
            vocab_stoi=self.symbol_to_idx,
            pad_to_len=self.pad_to_len,
            enc_type='one_hot'
            )    
            selfies_fp_temp[index, :] = np.array(one_hot).ravel()
        
        selfies_fingerprint = pd.concat([pd.DataFrame(selfies_fp_temp), self.properties], axis=1) if self.df.shape[1] > 1 else pd.DataFrame(selfies_fp_temp)
        return selfies_fingerprint
        
    def generate_unique_molecules(self, number_of_generation=10000):    
        generated_smiles_lists = []
        
        for _ in tqdm(range(number_of_generation)):
            number_of_alphabet = self.pad_to_len
            generated_one_hot = []
            for i in range(number_of_alphabet):
                one_hot_tmp = np.zeros(self.number_of_types_of_alphabet, dtype='int64')
                one_hot_tmp[np.random.randint(0, self.number_of_types_of_alphabet)] = 1
                generated_one_hot.append(list(one_hot_tmp))
            
            generated_selfies = sf.encoding_to_selfies(
            encoding=generated_one_hot,
            vocab_itos=self.idx_to_symbol,
            enc_type='one_hot'
            )

            one_hot = sf.selfies_to_encoding(
            selfies=generated_selfies,
            vocab_stoi=self.symbol_to_idx,
            pad_to_len=self.pad_to_len,
            enc_type='one_hot'
            )
            
            one_hot_arr = np.array(one_hot)
            generated_one_hot_arr = np.array(generated_one_hot)
            if one_hot_arr.shape[0] == generated_one_hot_arr.shape[0]:
                if sum(sum(abs(one_hot_arr - generated_one_hot_arr))) == 0:
                    generated_smiles = sf.decoder(generated_selfies)
                    generated_mol = Chem.MolFromSmiles(generated_smiles)
                    if generated_mol is not None and len(generated_smiles) != 0:
                        generated_smiles_modified = Chem.MolToSmiles(generated_mol)
                        generated_smiles_lists.append(generated_smiles_modified)

        # delete duplications of structures
        generated_smiles_lists = list(set(generated_smiles_lists))
        print('The number of generated unique structures :', len(generated_smiles_lists))
        return generated_smiles_lists