from serpapi import GoogleSearch
import pandas as pd
import time
import json
import os
import re



def collect_jobs_data(
    quarry,
    location="all",
    domain="default",
    number_of_queries=1,
    api_key="",
    data_frame=None,
    save_path=".",
    number_of_errors=2,
    report=True,
):
    """
    Collect job-listing data from Google Jobs via SerpApi and save each raw
    JSON response to disk.

    Parameters
    ----------
    quarry : str
        Search string for the vacancy (e.g., "Android developer").
    location : str, default "all"
        - "all": scrape every country found in `data_frame`.
        - A specific country name (e.g., "Austria"): scrape only that country.
    domain : str, default "default"
        - "default": always use `google.com`.
        - "local"  : use the country-specific Google domain contained in
          `data_frame["google_domain"]` (e.g., "google.at" for Austria).
    number_of_queries : int | str, default 1
        - "all": keep paginating until no more results are returned.
        - int  : maximum number of pages to fetch per country.
    api_key : str
        Your SerpApi API key.
    data_frame : pandas.DataFrame
        Must include at least two columns:
            * 'location'       – country name (e.g., "Austria")
            * 'google_domain'  – local Google domain (e.g., "google.at")
    save_path : str, default "."
        Directory in which JSON files are written.
    number_of_errors : int, default 2
        Stop requesting a country after this many consecutive empty pages.
    report : bool, default True
        If True, print a per-country summary at the end.

    Returns
    -------
    list[dict]
        One dictionary per country, e.g.,
        {"country": "Austria", "queries": 4, "errors": 1}.

    How it works
    ------------
    For each target country the function repeatedly calls the
    `google_jobs` SerpApi engine, paginating with `next_page_token`
    until one of three conditions is met:

    1. `number_of_queries` pages have been fetched.
    2. `number_of_errors` empty pages have occurred in a row.
    3. SerpApi no longer returns a `next_page_token`.

    Each response is saved as
        `<save_path>/<country>_<page_index>.json`
    so that raw data can be inspected or re-processed later. A simple
    safeguard pauses execution for one hour every 1 000 requests to stay
    within API limits.

    The function logs queries and errors in memory; if `report=True`, it
    prints that log in the familiar “--- Report ---” format before
    returning it.
    """
    # Argument validation
    if data_frame is None:
        raise ValueError("data_frame must be provided.")

    # Prepare list of countries
    if location == "all":
        countries = data_frame["location"].tolist()
    else:
        countries = [location]

    # Logging container
    report_data = []

    for country in countries:
        error_count = 0
        query_count = 0
        next_page_token = None  # For pagination

        # Determine domain
        google_domain = (
            "google.com"
            if domain == "default"
            else data_frame.loc[data_frame["location"] == country, "google_domain"].values[0]
        )

        while error_count < number_of_errors:
            if number_of_queries != "all" and query_count >= number_of_queries:
                break

            # Build search parameters
            search_params = {
                "q": quarry,
                "engine": "google_jobs",
                "location": country,
                "google_domain": google_domain,
                "api_key": api_key,
            }
            if next_page_token:
                search_params["next_page_token"] = next_page_token

            search = GoogleSearch(search_params)
            result = search.get_dict()

            # Save data
            file_name = f"{country}_{query_count}.json"
            file_path = os.path.join(save_path, file_name)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=4)

            # Check result
            if "jobs_results" not in result or not result["jobs_results"]:
                error_count += 1
            else:
                error_count = 0  # Reset on successful request

            query_count += 1

            # Get token for next page
            next_page_token = result.get("serpapi_pagination", {}).get("next_page_token")
            if not next_page_token:  # If no token, stop
                break

            # Rate-limit safeguard
            if query_count % 1000 == 0:
                print("Reached request limit. Pausing for 1 hour...")
                time.sleep(3600)

        report_data.append(
            {"country": country, "queries": query_count, "errors": error_count}
        )

    # Generate report
    if report:
        print("\n--- Report ---")
        for entry in report_data:
            print(f"{entry['country']}: {entry['queries']} queries, {entry['errors']} errors")

    return report_data



def clean_text(text):
    """Clean text from unusual line separators."""
    if not isinstance(text, str):
        return text
    return re.sub(r"[\u2028\u2029]", " ", text)  # Remove Line Separator and Paragraph Separator



def process_json_to_csv(data_dirs, region_df, output_file=r"../data/csv/jobs_data.csv"):
    final_data = []
    report = {
        "total_files": 0,
        "processed_files": 0,
        "empty_files": 0,
        "corrupted_files": 0
    }

    for data_dir in data_dirs:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    report["total_files"] += 1
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if not data.get("jobs_results"):
                            report["empty_files"] += 1
                            continue

                        for job in data["jobs_results"]:
                            row = {
                                "Location": clean_text(data.get("search_parameters", {}).get("location_used")),
                                "Region": region_df.loc[region_df["location"] == data.get("search_parameters", {}).get("location_used"), "Region"].values[0],
                                "EU Member": region_df.loc[region_df["location"] == data.get("search_parameters", {}).get("location_used"), "EU member"].values[0],
                                "Schengen Agreement": region_df.loc[region_df["location"] == data.get("search_parameters", {}).get("location_used"), "Schengen Agreement"].values[0],
                                "Google Domain Type": "local" if "local_domain" in file_path else "default",
                                "Google Domain Used": clean_text(data.get("search_parameters", {}).get("google_domain")),
                                "Job Title": clean_text(job.get("title")),
                                "Company Name": clean_text(job.get("company_name")),
                                "Job Location": clean_text(job.get("location")),
                                "Apply Options": clean_text(", ".join([opt.get("title", "") for opt in job.get("apply_options", [])])),
                                "Job Description": clean_text(job.get("description")),
                                "Work from home": clean_text(job.get("detected_extensions", {}).get("work_from_home")),
                                "Salary": clean_text(job.get("detected_extensions", {}).get("salary")),
                                "Schedule type": clean_text(job.get("detected_extensions", {}).get("schedule_type")),
                                "Qualifications": clean_text(job.get("detected_extensions", {}).get("qualifications")),
                                "Job ID": clean_text(job.get("job_id")),
                                "Search Date": clean_text(data.get("search_metadata", {}).get("created_at")),
                                "Search Query": clean_text(data.get("search_parameters", {}).get("q"))
                            }
                            final_data.append(row)

                        report["processed_files"] += 1

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"Error while processing file {file_path}: {e}")
                        report["corrupted_files"] += 1
                        continue

    # Create DataFrame and save to CSV
    df = pd.DataFrame(final_data)
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Processing summary
    report_text = (
        f"Processing completed:\n"
        f"- Total files: {report['total_files']}\n"
        f"- Successfully processed: {report['processed_files']}\n"
        f"- Empty files: {report['empty_files']}\n"
        f"- Corrupted files: {report['corrupted_files']}\n"
    )
    print(report_text)
    return report

