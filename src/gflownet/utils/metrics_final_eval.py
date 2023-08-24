import math
from copy import deepcopy
from itertools import product

from rdkit import Chem, DataStructs

import sqlite3
import pandas as pd

import numpy as np
import os
import glob


def candidates_eval(path, k=100, thresh=0.7):
    df = read_db_data_in_folder(path)

    df = df.drop_duplicates(subset=["smi"])

    if len(df) < k:
        raise ValueError(f"Number of unique SMILES ({len(df)}) is less than k ({k})")

    df["mol"] = df["smi"].apply(lambda x: Chem.MolFromSmiles(x))

    smiles = df["smi"].tolist()
    rewards = df["r"].tolist()
    mols = df["mol"].tolist()

    candidates = list(zip(rewards, smiles, mols))
    candidates = sorted(candidates, key=lambda m: m[0], reverse=True)

    topk = get_topk(rewards, k=k)
    diverse_topk = compute_diverse_topk(candidates, k=k, thresh=thresh)
    return {"topk": topk, "diverse_topk": diverse_topk}



def compute_diverse_topk(candidates, k, thresh=0.7):
    modes = [candidates[0]]
    mode_fps = [Chem.RDKFingerprint(candidates[0][2])]
    for i in range(1, len(candidates)):
        fp = Chem.RDKFingerprint(candidates[i][2])
        sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps) 
        # if sim to any of the modes is less than thresh, add to modes
        if max(sim) < thresh:
            modes.append(candidates[i])
            mode_fps.append(fp)
        if len(modes) >= k:
            # last_idx = i
            break
    return np.mean([i[0] for i in modes])  # return sim


def get_topk(rewards,k):
    # Sort the rewards
    rewards = sorted(rewards, reverse=True)
    # Get the top k rewards
    topk_rewards = rewards[:k]
    # Return the mean of the top k rewards
    return np.mean(topk_rewards)


def read_db_data_in_folder(folder_path: str) -> pd.DataFrame:
    """
    Reads all data from the `results` table of all SQLite databases in a specified folder.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing sqlite3 database files.

    Returns
    -------
    pd.DataFrame
        A combined dataframe containing rows from the `results` table 
        from all found databases in the folder. Returns an empty dataframe if none of the 
        tables exist or other errors.
    """
    # Step 1: Identify all .db files in the directory
    db_files = glob.glob(os.path.join(folder_path, "*.db"))

    combined_data = []
    columns = None

    for db_path in db_files:
        conn = sqlite3.connect(db_path)
        
        try:
            df = pd.read_sql("SELECT * FROM results", conn)
            
            if columns is None:
                columns = df.columns.tolist()
            
            combined_data.append(df)
        except sqlite3.OperationalError as e:
            if "no such table: results" in str(e):
                print(f"The table 'results' does not exist in the database at path: {db_path}")
            else:
                raise e
        finally:
            conn.close()

    if not combined_data:
        return pd.DataFrame(columns=columns or [])
    
    return pd.concat(combined_data, ignore_index=True)



