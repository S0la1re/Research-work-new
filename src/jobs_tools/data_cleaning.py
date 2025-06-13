

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