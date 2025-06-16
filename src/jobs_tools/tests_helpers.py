import re
import json
import numpy as np
import pandas as pd
from typing import Set
from pathlib import Path
from rapidfuzz import fuzz, utils



def column_to_txt(column, output, df):
    separator = "\n\n\n" + "-" * 100 + "\n\n\n"
    with open(output, "w", encoding="utf-8") as f:
        f.write(separator.join(df[column].tolist()))



def txt_to_column(txt_path, df):
    df = df.copy()
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()  # Read and remove extra spaces
    ground_truth_labels = content.split("-" * 100)
    ground_truth_labels = [label.strip("\n") for label in ground_truth_labels]
    
    if len(ground_truth_labels) > len(df):
        ground_truth_labels = ground_truth_labels[:len(df)]
        df["Ground Truth"] = ground_truth_labels
    else:
        df["Ground Truth"] = ground_truth_labels
    return df



def extract_section(text, section_name):
    pattern = rf"{section_name}:\s*(.*?)(?=\n\d+\.|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None



def compare_fuzzy_sections(df):
    sections = ['Platform', 'Salary', 'Requirements', 'Nice to have', 'Responsibilities', 'Benefits']
    
    results = []

    for section in sections:
        col_gt = f"{section}_GT"
        col_gpt = f"{section}_GPT"
        
        df[f"{section} Token Set Ratio"] = df.apply(
            lambda row: fuzz.token_set_ratio(str(row[col_gt]), str(row[col_gpt])), axis=1
        ).round(2)
        
        avg_score = df[f"{section} Token Set Ratio"].mean().round(2)
        results.append((section, avg_score))

    print("Average Token Set Ratio per section:")
    for section, score in results:
        print(f"{section}: {score}")

    values = [value for _, value in results]
    print(f"\nOverall Average Token Set Ratio: {np.mean(values):.2f}")



def extract_values(cell):
    """
    Converts a JSON cell into a flat list of string values.
    Returns an empty list if the cell is empty or invalid.
    """
    if pd.isna(cell) or str(cell).strip() == "":
        return []

    try:
        parsed = json.loads(cell)
    except (ValueError, TypeError):
        return []

    values = []
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                values.extend(v)
            elif v is not None:
                values.append(v)
    elif isinstance(parsed, list):
        values.extend(parsed)
    else:
        values.append(parsed)

    # Normalize the values the same way RapidFuzz does inside token_set_ratio
    return [utils.default_process(str(x)) for x in values if str(x).strip()]



def technologies_token_set_ratio(gt_cell, gpt_cell):
    """
    Calculates the token_set_ratio for two JSON cells,
    using only their values.
    """
    vals_gt  = extract_values(gt_cell)
    vals_gpt = extract_values(gpt_cell)

    # both are empty → full match
    if not vals_gt and not vals_gpt:
        return 100

    s_gt  = " ".join(vals_gt)
    s_gpt = " ".join(vals_gpt)
    return fuzz.token_set_ratio(s_gt, s_gpt)



def finding_defferences(json_path: str, df: pd.DataFrame, column: str) -> None:
    """
    Compare technologies in a DataFrame column and a JSON file,
    then print a summary of differences.
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    # Retrieving lists of technologies
    tech_list = df[column].str.strip().str.lower().unique().tolist()
    json_tech_list = []
    for category, items in json_data.items():
        json_tech_list.extend([item.strip().lower() for item in items])
    
    tech_set = set(tech_list)
    json_set = set(json_tech_list)
    # Finding differences
    only_in_tech_counts = tech_set - json_set
    only_in_json = json_set - tech_set
    # Summary output
    print(f"\n\nComparison summary:")
    print(f"Total technologies in tech_counts: {len(tech_set)}")
    print(f"Total technologies in key_values: {len(json_set)}")
    print(f"Technologies only in tech_counts: {len(only_in_tech_counts)}")
    print(f"Technologies only in key_values: {len(only_in_json)}")

    if only_in_tech_counts:
        print("\nTechnologies in tech_counts but missing in key_values:")
        for tech in sorted(only_in_tech_counts):
            print(f"- {tech}")

    if only_in_json:
        print("\nTechnologies in key_values but missing in tech_counts:")
        for tech in sorted(only_in_json):
            print(f"- {tech}")



def finding_defferences_1(map_path: Path, key_values_path: Path) -> None:
    """
    Compare keys in map.json and key_values.json and print any differences.
    """
    def load_map_keys(path: Path) -> Set[str]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        # Use only the raw keys (lower-case technology names)
        return {k.strip() for k in data.keys()}
    
    def load_key_values_items(path: Path) -> Set[str]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        # All elements from all groups
        return {item.strip() for group in data.values() for item in group}
    
    map_set = load_map_keys(map_path)
    kv_set = load_key_values_items(key_values_path)

    only_map = sorted(map_set - kv_set)
    only_kv = sorted(kv_set - map_set)

    print(f"In map but not in key_values: {len(only_map)}")
    if only_map:
        print(*only_map, sep="\n  • ")

    print(f"\nIn key_values but not in map: {len(only_kv)}")
    if only_kv:
        print(*only_kv, sep="\n  • ")

    if not only_map and not only_kv:
        print("\nEverything is distributed correctly (no differences).")