import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect

# Mol2vec vectors
from mol2vec.features import mol2alt_sentence, MolSentence
from gensim.models import Word2Vec #Loading pre-trained model via word2vec

# from collections import defaultdict
from tqdm import tqdm

class FingerprintDescriptorGenerator:
    def __init__(self, mols):
        self.mols = mols
    
    # MACCS Keys Fingerprints (key-based fingerprint)
    def maccskeys_fingerprint(self):
        maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in self.mols]
        df_maccs = pd.DataFrame(np.array(maccs_fps, int))
        return df_maccs

    # RDKit fingerprints (Daylight Fingerprint, topological/path-based fingerprint)
    def rdkit_fingerprit(self):
        fp = [AllChem.RDKFingerprint(mol) for mol in self.mols]
        df_fp = pd.DataFrame(np.array(fp, int))
        return df_fp

    # Morgan Fingerprint (Circular Fingerprint)
    def morgan_fingerprit(self):
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in self.mols]
        df_morgan = pd.DataFrame(np.array(morgan_fps, int))
        return df_morgan

    # Atom Pair Fingerprint
    def atompair_fingerprit(self):
        atompair_fps = [GetAtomPairFingerprintAsBitVect(mol) for mol in self.mols]
        df_atompair = pd.DataFrame(np.array(atompair_fps, int))
        return df_atompair

    # TPSA
    def tpsa(self):
        from rdkit.Chem import rdMolDescriptors
        tpsa = [rdMolDescriptors._CalcTPSAContribs(mol) for mol in self.mols]
        df_tpsa = pd.DataFrame(np.array(tpsa, dtype=object).reshape(-1, 1)) # Since the length of each element (a tuple) in tpsa is different, we should treat it as an object.
        return df_tpsa

    # Mordred descriptors
    def mordred_descriptor(self, ignore_3D=True):
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        df_mordred = calc.pandas(self.mols)
        df_mordred = df_mordred.astype(float) #Mordred descriptor contains missing values, so need to change type to float.
        return df_mordred

    # RDKit descriptors
    def rdkit_descriptor(self):
        from rdkit.ML.Descriptors import MoleculeDescriptors
        from rdkit.Chem import  Descriptors
        descriptor_names = [name[0] for name in Descriptors.descList]
        descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        desc = [descriptor_calculator.CalcDescriptors(mol) for mol in self.mols]
        df_RDKit = pd.DataFrame(desc, columns=descriptor_names)
        return df_RDKit

    def mol2vec(self, radius=2, unseen='UNK'):
        model_path = 'model_300dim.pkl'
        model = Word2Vec.load(model_path).wv
        sentences = [MolSentence(mol2alt_sentence(mol, radius)) for mol in tqdm(self.mols)]

        vectors = []
        for sentence in sentences:
            vector = []
            for word in sentence:
                if word in model.key_to_index:
                    vector.append(model[word])
                elif unseen is not None:
                    vector.append(model[unseen])
                else:
                    vector.append([0]*model.vector_size)
            vectors.append(sum(vector)/len(vector))
        
        # Return as DataFrame
        return pd.DataFrame(vectors)

if __name__=="__main__":
    pass