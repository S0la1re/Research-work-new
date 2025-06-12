import os
import re
import json
import asyncio
import logging
import aiofiles
import numpy as np
from tqdm.asyncio import tqdm as atqdm

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress HTTP logs from the OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)


async def chatgpt_async(
    *,
    input_column_name=None,
    output_column_name=None,
    input_text_length=None,
    output_text_length=5,
    num_rows=None,
    df=None,
    system_prompt=None,
    user_prompt=None,
    gpt_model=None,
    client=None,
    batch_size=None,
    cache_file=None,
    concurrency_limit=10,
    max_retries=5,       # applies only to fatal errors
    retry_delay=0.25,    # fallback delay in seconds
):
    """
    Process a DataFrame in batches through the OpenAI Chat API, filling an output column with generated responses.

    Args:
        input_column_name (str): Name of the column containing input text.
        output_column_name (str): Name of the column to store the ChatGPT responses.
        input_text_length (int, optional): Maximum number of words from the input to include. If None, use full input.
        output_text_length (int): Maximum number of tokens to generate in the response.
        num_rows (int, optional): Number of rows from the top of the DataFrame to process. If None, process all rows.
        df (pd.DataFrame): The DataFrame to process.
        system_prompt (str, optional): System-level prompt to prepend to each request.
        user_prompt (str): User-level prompt template preceding the input text.
        gpt_model (str): Identifier of the OpenAI GPT model to use.
        client (OpenAI client): Initialized OpenAI API client.
        batch_size (int, optional): Number of rows to process per batch.
        cache_file (str, optional): Path to a JSON file for caching responses.
        concurrency_limit (int): Maximum number of concurrent API calls.
        max_retries (int): Number of retries allowed for non-transient (fatal) errors.
        retry_delay (float): Delay in seconds before retrying after an error.

    Returns:
        pd.DataFrame: A new DataFrame with the output_column_name filled with responses.
    """
    # ---------- VALIDATION ----------
    if df is None:
        raise ValueError("df (DataFrame) must be provided.")
    if input_column_name is None:
        raise ValueError("input_column_name must be provided.")
    if input_column_name not in df.columns:
        raise ValueError(f"Column '{input_column_name}' not found in DataFrame.")
    if output_column_name is None:
        raise ValueError("output_column_name must be provided.")
    if gpt_model is None:
        raise ValueError("gpt_model must be provided.")
    if user_prompt is None:
        raise ValueError("user_prompt must be provided.")
    if client is None:
        raise ValueError("client must be provided.")

    # ---------- DATA PREPARATION ----------
    df = df.head(num_rows).copy() if num_rows else df.copy()
    if output_column_name not in df.columns:
        df[output_column_name] = ""

    # ---------- CACHE INITIALIZATION ----------
    cache = {}
    if cache_file and os.path.exists(cache_file):
        async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
            try:
                cache = json.loads(await f.read())
            except json.JSONDecodeError:
                logging.warning("Cache file is corrupted or empty — continuing without cache.")

    semaphore = asyncio.Semaphore(concurrency_limit)  # limit concurrency

    # ---------- HELPER FUNCTION: error classification ----------
    def _is_transient(err_msg: str) -> bool:
        """Return True for transient errors (rate-limit/network), False for fatal errors."""
        message = err_msg.lower()
        transient_keys = [
            "rate limit", "please try again", "quota reached soon",
            "tokens per min", "requests per min", "rpm", "tpm",
            "server overloaded", "timeout", "connection reset"
        ]
        return any(key in message for key in transient_keys)

    # ---------- PROCESS A SINGLE ROW ----------
    async def process_row(index, row):
        async with semaphore:
            source_text = row[input_column_name]
            truncated = (
                " ".join(source_text.split()[:input_text_length])
                if input_text_length else source_text
            )

            # ----- use cache if available -----
            if cache_file and truncated in cache:
                df.at[index, output_column_name] = cache[truncated]
                return

            # build messages for the API call
            prompt = f"{user_prompt} {truncated}"
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            fatal_attempts = 0  # count only non-transient failures

            while True:  # retry indefinitely on rate-limit
                try:
                    resp = await client.chat.completions.with_raw_response.create(
                        model=gpt_model,
                        messages=messages,
                        max_tokens=output_text_length,
                        temperature=0,
                    )
                    answer = resp.parse().choices[0].message.content

                    if cache_file:
                        cache[truncated] = answer
                    df.at[index, output_column_name] = answer
                    return  # success

                except Exception as e:
                    msg = str(e)

                    if _is_transient(msg):
                        # ---------- TRANSIENT / RATE-LIMIT ERROR ----------
                        # try to extract delay from message or headers
                        wait = None
                        m = re.search(r"in\s+(\d+(?:\.\d+)?)\s*(ms|s)", msg, re.I)
                        if m:
                            val = float(m.group(1))
                            wait = val / 1000 if m.group(2).lower() == "ms" else val
                        elif hasattr(e, "response"):
                            ra = e.response.headers.get("Retry-After")
                            if ra:
                                try:
                                    wait = float(ra)
                                except ValueError:
                                    pass
                        if wait is None:
                            wait = retry_delay
                        await asyncio.sleep(wait + 0.005)  # small buffer
                        continue

                    # ---------- FATAL ERROR ----------
                    fatal_attempts += 1
                    if fatal_attempts < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue

                    logging.error(
                        f"Row {index}: unrecoverable error after {max_retries} attempts → {msg}"
                    )
                    # leave cell blank on failure
                    return

    # ---------- BATCH SPLITTING ----------
    total_rows = len(df)
    if batch_size and batch_size < total_rows:
        idx_split = np.array_split(range(total_rows), total_rows // batch_size + 1)
        batches = [df.iloc[idx] for idx in idx_split]
    else:
        batches = [df]

    # ---------- PROCESS BATCHES ----------
    with atqdm(total=len(batches), desc="Processing Batches", leave=True) as bar:
        for batch in batches:
            await asyncio.gather(
                *[process_row(i, r) for i, r in batch.iterrows()]
            )

            # save cache after each batch
            if cache_file:
                async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(cache, ensure_ascii=False, indent=4))

            bar.update(1)

    return df
