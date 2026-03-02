#!/usr/bin/env python
# coding: utf-8

# ## Description
# This file is for simple data for transformer training. It is used for testing the training loop and the transformer structure. This project chooses the dataset Multi30k and IWSLT14 De–En. These both are small datasets for machine translation. The Multi30k dataset contains 30,000 parallel sentences in English and German, while the IWSLT14 De–En dataset contains around 160,000 parallel sentences in German and English
# 

# In[1]:


import os
import json
import csv
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    OUT_ROOT = Path(__file__).resolve().parent
except NameError:
    OUT_ROOT = Path.cwd()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OUT_ROOT   # Root directory where datasets are saved


# In[2]:


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if r["text"] and r["target"]:  # Skip empty records
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(records, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for r in records:
            if r["text"] and r["target"]:  # Skip empty records
                writer.writerow([r["text"], r["target"]])

def process_multi30k():
    try:
        ds = load_dataset("bentrevett/multi30k")  # 若失败，请尝试 "bentrevett/multi30k"
    except Exception as e:
        print("Error loading Multi30k dataset:", e)
        return

    def to_pair(example):
        src = example.get("en") or example.get("english") or example.get("caption_en")
        tgt = example.get("de") or example.get("german") or example.get("caption_de")
        return {"text": src, "target": tgt}

    for split in ds.keys():
        recs = [to_pair(x) for x in ds[split] if x]  # Skip invalid examples
        out_dir = os.path.join(OUT_ROOT, "multi30k")
        ensure_dir(out_dir)
        save_jsonl(recs, os.path.join(out_dir, f"{split}.jsonl"))
        save_csv(recs, os.path.join(out_dir, f"{split}.csv"))

    with open(os.path.join(OUT_ROOT, "multi30k", "README.txt"), "w", encoding="utf-8") as f:
        f.write("Multi30k saved splits. Each JSONL line: {\"text\": src_en, \"target\": tgt_de}.\n")


# In[3]:


def process_opus100():
    try:
        ds = load_dataset("opus100", "de-en")  # German-English language pair
        def to_pair(example):
            src = example.get("translation", {}).get("de")
            tgt = example.get("translation", {}).get("en")
            return {"text": src, "target": tgt}

        for split in ds.keys():
            recs = [to_pair(x) for x in ds[split] if x]
            out_dir = os.path.join(OUT_ROOT, "opus100_de_en")
            ensure_dir(out_dir)
            save_jsonl(recs, os.path.join(out_dir, f"{split}.jsonl"))
            save_csv(recs, os.path.join(out_dir, f"{split}.csv"))

        with open(os.path.join(OUT_ROOT, "opus100_de_en", "README.txt"), "w", encoding="utf-8") as f:
            f.write("Opus100 (de-en) saved splits. Each JSONL line: {\"text\": src_de, \"target\": tgt_en}.\n")
    except Exception as e:
        print("Error loading Opus100 dataset:", e)


# In[4]:


def load_data(data_name, default_file="train.csv",file_format="csv", as_numpy=False):
    """
    Load a dataset into a DataFrame or NumPy array.

    Args:
        data_name (str): The name of the dataset folder (e.g., "multi30k", "opus100_de_en").
        default_file (str): The default file name to load (e.g., "train.csv").
        file_format (str): The file format to load ("csv" or "jsonl").
        as_numpy (bool): If True, return the data as a NumPy array. Otherwise, return a DataFrame.

    Returns:
        pd.DataFrame or np.ndarray: The loaded dataset.
    """
    # Define the dataset directory
    data_dir = os.path.join(OUT_ROOT, data_name)
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")
    
    # Load the dataset based on the file format
    if file_format == "csv":
        file_path = os.path.join(data_dir, default_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file '{file_path}' not found.")
        df = pd.read_csv(file_path)
    elif file_format == "jsonl":
        file_path = os.path.join(data_dir, default_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file '{file_path}' not found.")
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'jsonl'.")
    
    # Return as NumPy array if requested
    if as_numpy:
        return df.to_numpy()
    
    return df


# In[5]:


if __name__ == "__main__":
    ensure_dir(OUT_ROOT)
    print("Downloading and saving Multi30k ...")
    process_multi30k()
    ensure_dir(OUT_ROOT)
    print("Downloading and saving Opus100 ...")
    process_opus100()

    # Example usage
    # Load Multi30k as a DataFrame
    multi30k_df = load_data("multi30k", file_format="csv", as_numpy=False)
    print("Multi30k DataFrame:")
    print(multi30k_df.head())

    # Load Opus100 as a NumPy array
    opus100_np = load_data("opus100_de_en", file_format="csv", as_numpy=True)
    print("\nOpus100 NumPy Array:")
    print(opus100_np[:5])  # Print the first 5 rows



