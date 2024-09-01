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
    def rdkit_fingerprint(self):
        fp = [AllChem.RDKFingerprint(mol) for mol in self.mols]
        df_fp = pd.DataFrame(np.array(fp, int))
        return df_fp

    # Morgan Fingerprint (Circular Fingerprint)
    def morgan_fingerprint(self):
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in self.mols]
        df_morgan = pd.DataFrame(np.array(morgan_fps, int))
        return df_morgan

    # Atom Pair Fingerprint
    def atompair_fingerprint(self):
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

if __name__ == "__main__":
    # Get SMILES input from the user
    smiles_input = input("Enter SMILES strings (separate multiple SMILES with a space): ")
    smiles_list = smiles_input.split()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    generator = FingerprintDescriptorGenerator(mols)

    # Ask user for the type of fingerprint or descriptor
    print("Choose a type of fingerprint or descriptor to generate:")
    print("1. MACCS Keys Fingerprint")
    print("2. RDKit Fingerprint")
    print("3. Morgan Fingerprint")
    print("4. Atom Pair Fingerprint")
    print("5. TPSA")
    print("6. Mordred Descriptor")
    print("7. RDKit Descriptor")
    print("8. Mol2Vec")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        print("MACCS Keys Fingerprint:")
        print(generator.maccskeys_fingerprint())
    elif choice == '2':
        print("RDKit Fingerprint:")
        print(generator.rdkit_fingerprint())
    elif choice == '3':
        print("Morgan Fingerprint:")
        print(generator.morgan_fingerprint())
    elif choice == '4':
        print("Atom Pair Fingerprint:")
        print(generator.atompair_fingerprint())
    elif choice == '5':
        print("TPSA:")
        print(generator.tpsa())
    elif choice == '6':
        print("Mordred Descriptor:")
        print(generator.mordred_descriptor())
    elif choice == '7':
        print("RDKit Descriptor:")
        print(generator.rdkit_descriptor())
    elif choice == '8':
        print("Mol2Vec:")
        print(generator.mol2vec())
    else:
        print("Invalid choice. Please run the script again and select a valid option.")