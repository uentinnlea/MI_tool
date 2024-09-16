"""
Prerequisite:
- The first column of the dataset should be SMILES representation.
"""

import pandas as pd
import numpy as np
import selfies as sf

class SELFIES:
    def __init__(self, dataframe):
        self.df = dataframe
        self.smileses = self.df.iloc[:, 0]
        self.properties = None
        self.selfies_lists = None

        if self.df.shape[1] > 1:
            self.properties = self.df.iloc[:, 1:]

        self.selfies_lists = []
        for smiles in self.smileses:
            selfies = sf.encoder(smiles)
            self.selfies_lists.append(selfies)


    def SELFIES_fingerprint_generation(self):
        
        alphabet = sf.get_alphabet_from_selfies(self.selfies_lists)
        alphabet.add('[nop]')
        alphabet = list(sorted(alphabet))

        pad_to_len = max(sf.len_selfies(s) for s in self.selfies_lists)
        symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
        idx_to_symbol = {i: s for i, s in enumerate(alphabet)}

        selfies_fingerprint = np.zeros([self.df.shape[0], pad_to_len * len(symbol_to_idx)])
        for index, selfies in enumerate(self.selfies_lists):
            one_hot = sf.selfies_to_encoding(
            selfies=selfies,
            vocab_stoi=symbol_to_idx,
            pad_to_len=pad_to_len,
            enc_type='one_hot'
            )    
            selfies_fingerprint[index, :] = np.array(one_hot).ravel()
        
        selfies_fingerprint = pd.DataFrame(selfies_fingerprint)
        if self.df.shape[1] > 1:
            selfies_fingerprint = pd.concat([selfies_fingerprint, self.properties], axis=1)
        return selfies_fingerprint