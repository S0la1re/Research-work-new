import re
import numpy as np
from rapidfuzz import fuzz



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