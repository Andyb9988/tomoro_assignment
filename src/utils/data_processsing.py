import json
import random
from logging import Logger
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import pandas as pd
import regex as re

from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


def shuffle_and_split(data, length: int, seed: int = 42):
    """
    Shuffles the data, limits it to the specified length, and splits it into training and validation sets (70/30 split).
    Each list within the dictionaries is shuffled individually,
    and then the entire list is shuffled.

    Parameters:
    - data (list): List of dictionaries to shuffle.
    - length (int): Number of shuffled items to use for splitting.
    - seed (int, optional): Seed for the random number generator.

    Returns:
    - tuple: (train_data, validation_data)
    """
    # Initialize a separate random number generator instance
    rng = random.Random(seed)

    # Shuffle each list within the dictionaries
    shuffled_data = []
    for item in data:
        shuffled_item = {}
        for key, value in item.items():
            if isinstance(value, list):
                # Shuffle the list using the separate RNG instance
                shuffled_value = rng.sample(value, len(value))
                shuffled_item[key] = shuffled_value
            else:
                shuffled_item[key] = value
        shuffled_data.append(shuffled_item)

    # Shuffle the entire list using the separate RNG instance
    rng.shuffle(shuffled_data)

    # Limit the data to the specified length
    shuffled_data = shuffled_data[:length]

    return shuffled_data


class DataParser:
    """
    A class to parse and process data, including cleaning values,
    parsing tables, and converting data into a pandas DataFrame.
    """

    def __init__(self):
        # Initialize any instance variables if necessary
        pass

    def clean_value(self, value: Optional[str]) -> Union[float, str, None]:
        """
        Cleans a value by removing currency symbols, commas, and percentage signs,
        and converting negative numbers in parentheses to floats.

        Args:
            value (Optional[str]): The string value to be cleaned.

        Returns:
            Union[float, str, None]: The cleaned value as a float, string, or None.
        """
        if not value:
            return None

        # Remove currency symbols and commas
        value = value.replace("$", "").replace(",", "").strip()
        # Handle percentages

        # Handle negative numbers in parentheses or with minus sign
        match = re.match(r"-?\(?\s*(-?\d+\.?\d*)\s*\)?", value)
        if match:
            num = match.group(1)
            # Convert based on negative indication
            try:
                result = -float(num) if "-" in value or "(" in value else float(num)
                return result
            except ValueError:
                logger.warning(f"Value '{value}' could not be converted to float.")
                return value  # Return as-is if not a number
        return value

    def parse_table(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parses a table represented as a list of lists into a structured dictionary format.

        Args:
            table (List[List[str]]): The table data as a list of lists.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the parsed table.
        """
        if not table:
            logger.warning("Empty table provided to parse_table.")
            return []

        headers = table[0]
        if not headers:
            logger.error("Table headers are missing.")
            return []

        row_label_header = headers[0]
        col_headers = headers[1:]

        data = []
        for row in table[1:]:
            if not row:
                continue
            row_label = row[0]
            row_data = row[1:]
            item = {"row_label": row_label}

            for i, header in enumerate(col_headers):
                value = row_data[i] if i < len(row_data) else None
                cleaned_value = self.clean_value(value)
                item[header] = cleaned_value
            data.append(item)

        logger.info(f"Parsed {len(data)} rows from the table.")
        return data

    def make_pandas_df(
        self, data: List[Dict[str, Any]], use_parse_table: bool = True
    ) -> pd.DataFrame:
        """
        Converts processed data into a pandas DataFrame with optional table parsing.

        Args:
            data (List[Dict[str, Any]]): The processed data.
            use_parse_table (bool): Flag to indicate if the parse_table function should be used.

        Returns:
            pd.DataFrame: The resulting pandas DataFrame.
        """
        records = []

        for entry in data:
            pre_text = " ".join(entry.get("pre_text", []))
            post_text = " ".join(entry.get("post_text", []))
            raw_table = entry.get("table", [])

            if use_parse_table:
                logger.info("Parsing table as requested.")
                parsed_table = self.parse_table(raw_table)
            else:
                logger.info("Skipping table parsing.")
                parsed_table = raw_table

            table = parsed_table
            combined = ""

            # Combine pre_text
            if pre_text:
                logger.info("Adding pre_text to context.")
                combined += "### Pre-Text\n" + pre_text + "\n\n"

            # Combine table
            if table:
                logger.info("Adding table to context.")
                table_json = (
                    json.dumps(table, indent=2) if use_parse_table else str(table)
                )
                combined += "### Table\n" + table_json + "\n\n"

            # Combine post_text
            if post_text:
                logger.info("Adding post_text to context.")
                combined += "### Post-Text\n" + post_text + "\n\n"

            context = combined

            id_ = entry.get("id", "")
            qa_entries = {k: v for k, v in entry.items() if k.startswith("qa_")}

            if qa_entries:
                for qa_key in sorted(qa_entries.keys()):
                    qa = qa_entries[qa_key]
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    exe_answer = qa.get("exe_ans", "")
                    annotation = entry.get("annotation", {})
                    step_list_key = f'step_list_{qa_key.split("_")[1]}'
                    step_list = annotation.get(step_list_key, qa.get("steps", []))
                    dialogue_break = annotation.get("dialogue_break", [])

                    record = {
                        "context": context,
                        "question": question,
                        "id": id_,
                        "step_list": step_list,
                        "dialogue_break": dialogue_break,
                        "answer": answer,
                        "exe_answer": exe_answer,
                    }
                    records.append(record)
            else:
                qa = entry.get("qa", {})
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                exe_answer = qa.get("exe_ans", "")
                annotation = entry.get("annotation", {})
                step_list = annotation.get("step_list", qa.get("steps", []))
                dialogue_break = annotation.get("dialogue_break", [])

                record = {
                    "context": context,
                    "question": question,
                    "id": id_,
                    "step_list": step_list,
                    "dialogue_break": dialogue_break,
                    "answer": answer,
                    "exe_answer": exe_answer,
                }
                records.append(record)

        logger.info(f"Processed {len(records)} records into DataFrame.")
        df = pd.DataFrame(records)
        df = df[
            [
                "context",
                "question",
                "id",
                "step_list",
                "dialogue_break",
                "answer",
                "exe_answer",
            ]
        ]
        return df

    def process_data(
        self, data: List[Dict[str, Any]], use_parse_table: bool = True
    ) -> pd.DataFrame:
        """
        Processes the input data and returns a pandas DataFrame.

        Args:
            data (List[Dict[str, Any]]): The input data to be processed.
            use_parse_table (bool): Flag to indicate if the parse_table function should be used.

        Returns:
            pd.DataFrame: The processed pandas DataFrame.
        """
        if not data:
            logger.error("No data provided to process_data.")
            return pd.DataFrame()
        df = self.make_pandas_df(data, use_parse_table=use_parse_table)
        return df
