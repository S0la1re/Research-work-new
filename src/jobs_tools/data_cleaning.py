

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
