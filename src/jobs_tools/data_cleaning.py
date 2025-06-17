import re
import pandas as pd
import os, json, re
from itertools import chain
from collections import Counter



def remove_exact_duplicates(df):
    """Removes fully duplicated rows and prints the number of removed records."""
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(keep="first")
    removed_count = initial_count - len(df_cleaned)
    
    print("Step 1: Removing exact duplicates")
    print(f"- Initial number of rows: {initial_count}")
    print(f"- Duplicates removed: {removed_count}")
    print(f"- Remaining rows: {len(df_cleaned)}\n")
    
    return df_cleaned



def remove_job_id_duplicates(df):
    """Removes Job ID duplicates within the same country, prioritizing the 'google.com' domain, then the most recent date."""
    initial_count = len(df)
    
    # Add a column for domain priority (0 - google.com, 1 - other local domains)
    df["Domain Priority"] = df["Google Domain Type"].apply(lambda x: 0 if x == "default" else 1)
    
    # Sort: first by google.com (priority 0), then by date (latest comes first)
    df_cleaned = df.sort_values(by=["Domain Priority", "Search Date"], ascending=[True, False])
    df_cleaned = df_cleaned.drop_duplicates(subset=["Job ID", "Location"], keep="first")

    df_cleaned = df_cleaned.sort_values(by="Location", ascending=True).reset_index(drop=True)

    removed_count = initial_count - len(df_cleaned)
    
    print("Step 2: Removing Job ID duplicates within the same country")
    print(f"- Initial row count: {initial_count}")
    print(f"- Duplicates removed: {removed_count}")
    print(f"- Remaining rows: {len(df_cleaned)}\n")
    
    return df_cleaned.drop(columns=["Domain Priority"])  # Remove the temporary column



def split_values(df):
    """Split 'Language langdetect confidence' into:
       - 'Language': the two-letter code before the colon
       - 'Confidence': the numeric part after the colon (as float64)
       Then drop the original column.
    """
    # Insert 'Language' just before the original column
    idx = df.columns.get_loc("Language langdetect confidence")
    df.insert(idx, "Language",
              df["Language langdetect confidence"]
                .str.split(":", n=1)
                .str[0])
    
    # Insert 'Confidence' right after it, converting to float64
    df.insert(idx + 1, "Confidence",
              df["Language langdetect confidence"]
                .str.split(":", n=1)
                .str[1]
                .astype("float64"))
    
    # Drop the old combined column
    df = df.drop(columns=["Language langdetect confidence"])
    
    return df



def low_confidence_data(df):
    df = df.copy()
    # Sort by column: 'Language langdetect confidence' (ascending)
    df = df.sort_values(['Confidence'])
    # Filter rows based on column: 'Language langdetect confidence'
    df = df[df['Confidence'] < 0.99]
    return df



def clean_data(df):
    df = df.copy()
    # Drop columns: 'Drop', 'Language' and 'Language gpt-4o-2024-11-20'
    df = df.drop(columns=['Drop', 'Language', 'Language gpt-4o-2024-11-20'])
    # Rename column 'Manual check' to 'Language'
    df = df.rename(columns={'Manual check': 'Language'})
    col = df.pop('Confidence') 
    df.insert(len(df.columns), 'Confidence', col)
    return df



# —————————— Remove hallucinated words ————————————————————

# —— 0. Settings ——————————————————————————————————————————————————
STANDARDIZE = True            # ⬅ turn off if you want to keep original forms
USE_REGEX   = False           # ⬅ same as before

# —— 1. Load the dictionary and build the “variant → canonical” map —————
with open("../data/json/synonyms.json", encoding="utf8") as f:
    SYNONYMS = json.load(f)

VAR2CANON = {v.lower(): canon      # "m-v-c"  -> "mvc"
             for canon, variants in SYNONYMS.items()
             for v in variants}

# —— 2. Helpers ——————————————————————————————————————————————
def make_ngrams(tokens, n_max=3):
    return {" ".join(tokens[i:i+n])
            for n in range(1, n_max + 1)
            for i in range(len(tokens) - n + 1)}

def is_real(term, text_flat, ngrams,
            *, use_regex=USE_REGEX, synonyms=SYNONYMS):
    term_l = term.lower()
    if term_l not in synonyms:           # not in the list of “checkable” terms → immediately True
        return True

    patterns = synonyms[term_l]
    if any(p.lower() in ngrams for p in patterns):
        return True
    if use_regex and any(re.search(rf"\b{p}\b", text_flat) for p in patterns):
        return True
    return False

# —— 3. Counters ————————————————————————————————————————————————
removed_counter       = Counter()   # what was removed as a hallucination
canonicalized_counter = Counter()   # how many times we replaced with canonical term

def to_canon(term):
    """If STANDARDIZE=True and the variant is in the dictionary → return canonical form."""
    t_low = term.lower()
    if STANDARDIZE and t_low in VAR2CANON:
        canon = VAR2CANON[t_low]
        if canon != t_low:          # an actual replacement, not “mvc” -> “mvc”
            canonicalized_counter[canon] += 1
        return canon
    return t_low                    # otherwise return the original form (lowercased)

# —— 4. Filter ——————————————————————————————————————————————————
def remove_hallucinated(row):
    text_raw   = (row.get("Full Requirements") or "")
    text_lower = text_raw.lower()

    token_pattern = r"\.?[a-z0-9\+\#-]+(?:\.[a-z0-9\+\#-]+)*"
    tokens = re.findall(token_pattern, text_lower)

    ngrams  = make_ngrams(tokens)

    extracted = row.get("Extracted Technologies GPT", "")
    try:
        tech_dict = json.loads(extracted)
    except Exception:
        return extracted            # corrupted JSON

    for cat in list(tech_dict.keys()):
        cleaned = []
        for term in tech_dict[cat]:
            canon_term = to_canon(term)
            if is_real(canon_term, text_lower, ngrams):
                cleaned.append(canon_term)
            else:
                removed_counter[canon_term] += 1

        if cleaned:
            tech_dict[cat] = list(dict.fromkeys(cleaned))  # remove duplicates
        else:
            del tech_dict[cat]

    return json.dumps(tech_dict, ensure_ascii=False)



def flat_terms(tech_json: str) -> set[str]:
    """{"cat": ["A", "B"]}  → {"a", "b"}   (lower-case)"""
    try:
        d = json.loads(tech_json)
    except Exception:
        return set()
    return {t.lower() for lst in d.values() for t in lst}



def extract_values(json_str):
    """
    Extracts and flattens values from a JSON string.

    Returns a comma-separated string of values if valid, otherwise None.
    Handles lists and single values, skips empty or invalid JSON strings.
    """
    if pd.isna(json_str) or json_str.strip() == '{}' or json_str.strip() == '':
        return None
    try:
        data = json.loads(json_str)
        values = []
        for val in data.values():
            if isinstance(val, list):
                values.extend(map(str, val))  # add list elements as strings
            else:
                values.append(str(val))  # add a single value as a string
        return ', '.join(values) if values else None
    except json.JSONDecodeError:
        return None



def normalize_tech_string(tech_str):
    """
    Cleans and normalizes a comma-separated string of technologies.

    - Converts each term to lowercase and strips whitespace.
    - Removes terms listed in the global `remove_list`.
    - Returns a cleaned, comma-separated string or None if empty or invalid input.
    """
    with open("../data/json/remove_list.json", encoding="utf8") as f:
        remove_list = json.load(f)
    if pd.isna(tech_str):
        return None
    try:
        tech_list = tech_str.split(',')
        clean_terms = []
        for term in tech_list:
            term_clean = term.strip().lower()
            if term_clean not in remove_list:
                clean_terms.append(term_clean)
        return ', '.join(clean_terms) if clean_terms else None
    except Exception:
        return None
    


def categorize(tech_cell: str) -> dict:
    """
    Build a JSON object that maps technologies in a cell to their categories.
    """
    with open('../data/json/key_values.json', 'r', encoding='utf-8') as f:
        json_file = json.load(f)  # categories → [tech1, tech2, …]

    # Prepare a "reverse" dictionary: tech (lowercase) → category name
    reverse_map = {
        tech.lower(): category
        for category, tech_list in json_file.items()
        for tech in tech_list
    }

    if pd.isna(tech_cell) or not tech_cell.strip():
        return {}

    # Build the JSON object for a single cell
    result = {}
    # Example: "kotlin, retrofit " → ['kotlin', 'retrofit']
    for raw in tech_cell.split(','):
        tech = raw.strip()
        if not tech:
            continue
        cat = reverse_map.get(tech.lower())  # look up the category
        if cat:
            # Add to result, avoiding duplicates
            result.setdefault(cat, []).append(tech)

    return result



def fix_casing(cat_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Fix the casing of each term in a category dictionary using a mapping file.
    """
    # Load the mapping "lowercase → correct casing"
    with open("../data/json/map.json", 'r', encoding='utf-8') as f:
        proper_case = json.load(f)
    for category, tech_list in cat_dict.items():
        cat_dict[category] = [
            proper_case.get(t.lower(), t)        # If not found in the mapping — keep as is
            for t in tech_list
        ]
    return cat_dict



def keys_to_columns(key_values_path: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with open(key_values_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure that the column is a dict and not a JSON string
    def to_dict(x):
        """str → dict  |  dict → dict  |  NaN/empty → {}"""
        if pd.isna(x) or (isinstance(x, str) and not x.strip()):
            return {}
        if isinstance(x, str):
            return json.loads(x)
        return x
    
    df['Tech_dict'] = df['Technologies Categorized'].apply(to_dict)
    
    # Find all unique categories (keys)
    all_categories = set(chain.from_iterable(df['Tech_dict'].map(dict.keys)))

    for cat in sorted(all_categories):
        df[cat] = df['Tech_dict'].apply(
            lambda d: ', '.join(d.get(cat, []))          # # list -> string
                    if d.get(cat) else ''              # no technology -> ""
        )
    
    return df



def leave_only_relevant_tech(json_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes any technology from each row's 'Technologies Only' list if it appears
    in the opposite platform's stack (Android vs. iOS) as defined in a JSON file.
    Returns the 'Technologies Only' column as a cleaned comma-separated string.

    Args:
        json_path: Path to the JSON file containing 'Android' and 'iOS' keys mapping to lists of technologies.
        df: A pandas DataFrame with at least two columns:
            - 'Platform': values 'Android' or 'iOS'
            - 'Technologies Only': a list of technology names, a comma-separated string, or iterable

    Returns:
        A copy of df with filtered 'Technologies Only' strings and prints statistics of removals.
    """
    # Load platform stacks from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    android_stack = set(data.get("Android", []))
    ios_stack = set(data.get("iOS", []))

    # Copy DataFrame to avoid mutating original
    df = df.copy()
    total_removed = 0
    removal_counts = []

    # Iterate through rows
    for idx, row in df.iterrows():
        plat = row.get('Platform')
        techs = row.get('Technologies Only')

        # Prepare list of techs
        if pd.isna(techs) or techs is None:
            tech_list = []
        elif isinstance(techs, str):
            tech_list = [t.strip() for t in techs.split(',') if t.strip()]
        else:
            try:
                tech_list = list(techs)
            except TypeError:
                tech_list = []

        # Determine opposite stack
        if plat == 'Android':
            opposite = ios_stack
        elif plat == 'iOS':
            opposite = android_stack
        else:
            removal_counts.append(0)
            df.at[idx, 'Technologies Only'] = ''
            continue

        # Filter out technologies in the opposite stack
        filtered = [tech for tech in tech_list if tech not in opposite]
        removed = len(tech_list) - len(filtered)

        # Convert back to comma-separated string
        cleaned = ', '.join(filtered)

        # Update DataFrame
        df.at[idx, 'Technologies Only'] = cleaned
        removal_counts.append(removed)
        total_removed += removed

    # Print statistics
    print(f"Total rows processed: {len(df)}")
    print(f"Total technologies removed: {total_removed}")

    return df
