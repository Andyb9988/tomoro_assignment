import json
from logging import Logger
from typing import (
    Any,
    List,
)

import dspy
import pandas as pd
from dspy.datasets import DataLoader

from config.config import DataItem
from src.tools.metrics import (
    calculate_accuracy_from_df,
    calculate_average_reasoning_score,
    get_answer_accuracy_df,
    get_reasoning_scores,
)
from src.tools.models import OpenAILM
from src.tools.modules import OutputFinalAnswer
from src.utils.data_processsing import (
    DataParser,
    shuffle_and_split,
)
from src.utils.eval_df import get_llm_answer_outcome_df
from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


with open("data/train.json") as f:
    data = json.load(f)

dl = DataLoader()
parser = DataParser()
model_list = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini"]

split_data = shuffle_and_split(data=data, length=50, seed=30)
parsed_data = parser.process_data(data=split_data, use_parse_table=False)

eval_df = dl.from_pandas(
    parsed_data,
    fields=(
        "context",
        "question",
        "id",
        "step_list",
        "dialogue_break",
        "answer",
        "exe_answer",
    ),
    input_keys=("question", "context"),
)


def get_llm_answers(data: List[DataItem]) -> List[Any]:
    """
    Retrieves answers from the LLM for each item in the provided data list.

    Args:
        data (List[DataItem]): A list of DataItem instances, each containing context, id, and question.

    Returns:
        List[Any]: A list of answers obtained from the LLM corresponding to each data item.
    """
    llm_answers = []
    for i in data:
        obj = OutputFinalAnswer(context=i.context, id=i.id)
        ans_obj = obj.get_answer(question=i.question)
        llm_answers.append(ans_obj)

    return llm_answers


def main():
    # Initialize a list to store experiment results
    experiment_results = []

    # Iterate over each model in the model_list
    for model in model_list:
        logger.info(f"Running experiment for model: {model}")
        # Configure the language model
        lm = OpenAILM(model=model)
        dspy.configure(lm=lm)

        # Generate LLM answers
        llm_ans_list = get_llm_answers(eval_df)

        # Get accuracy and reasoning score for each question
        accuracy_df = get_answer_accuracy_df(data=eval_df, llm_answers=llm_ans_list)
        reasoning_df = get_reasoning_scores(data=eval_df, llm_ans=llm_ans_list)

        # Compute average metrics
        accuracy = calculate_accuracy_from_df(accuracy_df)
        reasoning_score = calculate_average_reasoning_score(reasoning_df)

        # Get the outcome DataFrame
        outcome_df = get_llm_answer_outcome_df(
            eval_df, llm_ans_list, reasoning_df=reasoning_df, accuracy_df=accuracy_df
        )

        # Save the outcome DataFrame to CSV
        outcome_csv_path = f"data/table_not_parsed_outcome_{model}.csv"
        outcome_df.to_csv(outcome_csv_path, index=False)
        logger.info(f"Saved outcome DataFrame to {outcome_csv_path}")

        # Record the experiment result
        experiment_results.append(
            {
                "model": model,
                "num_questions": len(eval_df),
                "accuracy": accuracy,
                "reasoning_score": reasoning_score,
            }
        )

    # Create a summary DataFrame of all experiments
    experiment_df = pd.DataFrame(experiment_results)
    # Save the summary DataFrame to CSV
    experiment_csv_path = "data/no_table_parse_experiment_results.csv"
    experiment_df.to_csv(experiment_csv_path, index=False)
    logger.info(f"Saved experiment summary to {experiment_csv_path}")


if __name__ == "__main__":
    main()
