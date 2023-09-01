import math
from copy import deepcopy
from itertools import product

from rdkit import Chem, DataStructs

import sqlite3
import pandas as pd

import numpy as np
import os
import glob
import torch
import networkx as nx


def mols_and_reward_from_path(path):  
    df = read_db_data_in_folder(path)
    df = df.drop_duplicates(subset=["smi"])

    if len(df) < k:
        raise ValueError(f"Number of unique SMILES ({len(df)}) is less than k ({k})")

    df["mol"] = df["smi"].apply(lambda x: Chem.MolFromSmiles(x))

    smiles = df["smi"].tolist()
    rewards = df["r"].tolist()
    mols = df["mol"].tolist()
    return mols,rewards

def candidates_eval(gen_candidates_info_list, k=100, reward_thresh=8, tanimoto_thresh=0.7):
    #unpack gen_candidates_info_list
    smiles = []
    flat_rewards = []
    for batch in gen_candidates_info_list:
        smiles.extend(batch[0])
        flat_rewards.extend(batch[1])
    assert len(smiles) == len(flat_rewards)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    final_info = calculate_eval_metrics(mols,flat_rewards,k=k,reward_thresh=reward_thresh,tanimoto_thresh=tanimoto_thresh)
    return final_info


def calculate_eval_metrics(mols,rewards,k=100,reward_thresh=8,tanimoto_thresh=0.7):
    avg_topk = compute_avg_topk(rewards, k=k)

    candidates = list(zip(rewards,mols))
    candidates = sorted(candidates, key=lambda m: m[0], reverse=True)

    avg_reward_in_topk_modes = compute_diverse_topk(candidates, k=k, tanimoto_thresh=tanimoto_thresh)
    num_of_modes,num_candidates_above_reward_thresh = compute_num_of_modes(candidates, reward_thresh = reward_thresh, tanimoto_thresh=tanimoto_thresh)
    return {"avg_topk": avg_topk, "avg_reward_in_topk_modes": avg_reward_in_topk_modes,"num_of_modes":num_of_modes, "max_reward":max(rewards).item(),"num_candidates":len(candidates),"num_candidates_above_reward_thresh":num_candidates_above_reward_thresh}



def compute_diverse_topk(candidates, k=100, tanimoto_thresh=0.7):
    modes = [candidates[0]]
    mode_fps = [Chem.RDKFingerprint(candidates[0][1])]
    for i in range(1, len(candidates)):
        fp = Chem.RDKFingerprint(candidates[i][1])
        sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps) 
        # if sim to any of the modes is less than thresh, add to modes
        if max(sim) < tanimoto_thresh:
            modes.append(candidates[i])
            mode_fps.append(fp)
        if len(modes) >= k:
            # last_idx = i
            break
    avg_reward_in_topk_modes = np.mean([i[0] for i in modes])  
    return avg_reward_in_topk_modes 


def compute_avg_topk(rewards,k):
    # Sort the rewards
    rewards = sorted(rewards, reverse=True)
    # Get the top k rewards
    topk_rewards = rewards[:k]
    # Return the mean of the top k rewards
    return np.mean(topk_rewards)

def compute_num_of_modes(candidates, reward_thresh=8, tanimoto_thresh=0.7):
    candidates = sorted(candidates, key=lambda m: m[0], reverse=True)
    # cut of candidates with reward less than reward_thresh
    candidates = [c for c in candidates if c[0] >= reward_thresh]

    num_candidates_above_reward_thresh = len(candidates)
    if num_candidates_above_reward_thresh == 0:
        return 0, 0
    
    modes = [candidates[0]]
    mode_fps = [Chem.RDKFingerprint(candidates[0][1])]
    for i in range(1, len(candidates)):
        fp = Chem.RDKFingerprint(candidates[i][1])
        sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps) 
        # if sim to any of the modes is less than thresh, add to modes
        if max(sim) < tanimoto_thresh:
            modes.append(candidates[i])
            mode_fps.append(fp)

    num_of_modes = len(modes)
    return num_of_modes,num_candidates_above_reward_thresh


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



