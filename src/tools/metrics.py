import re
from logging import Logger
from typing import List

import dspy

from src.tools.signatures import AssessReasoning
from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


def get_reasoning_score(data, llm_ans):
    reasoning_list = []

    # Iterate through the data and llm_answers
    for data_item, ans_obj in zip(data, llm_ans):
        # Extract variables from data
        context = data_item.context
        dialogue_break = data_item.dialogue_break

        llm_reasoning = ans_obj.reasoning

        reasoning_accuracy = dspy.Predict(AssessReasoning)(
            actual_reasoning=dialogue_break,
            llm_reasoning=llm_reasoning,
            context=context,
        ).assessment_answer

        # Ensure reasoning_accuracy is numeric
        try:
            reasoning_accuracy = float(reasoning_accuracy)
            reasoning_list.append(reasoning_accuracy)
            logger.info(f"Reasoning Accuracy {reasoning_accuracy}")
        except ValueError:
            logger.error(
                f"Non-numeric reasoning accuracy encountered: {reasoning_accuracy}"
            )

    # Calculate the average reasoning accuracy if the list is not empty
    if reasoning_list:
        average_reasoning_score = sum(reasoning_list) / len(reasoning_list)
        logger.info(f"Average Reasoning Accuracy: {average_reasoning_score:.2f}")
        return average_reasoning_score
    else:
        logger.info("No reasoning scores to average.")
        return None


def get_answer_accuracy(data: List[object], llm_answers: List[object]) -> float:
    """
    Calculates the accuracy as the percentage of correct answers.

    An answer is considered correct if the absolute difference between
    the predicted and actual values (after cleaning) is within 1.

    Args:
        llm_answers (List[object]): List of predicted answer objects with an 'answer' attribute.
        data (List[object]): List of actual answer objects with an 'answer' attribute.

    Returns:
        float: Accuracy percentage of correct answers.
    """

    if not llm_answers or not data:
        logger.warning("One or both input lists are empty.")
        return 0.0

    if len(llm_answers) != len(data):
        logger.warning(
            "Input lists have different lengths. Processing up to the shortest length."
        )

    # Define a regex pattern to remove specified symbols
    pattern = r"[\(\)\$£%]"

    correct = 0
    total = 0
    min_length = min(len(llm_answers), len(data))

    for idx in range(min_length):
        predicted_obj = llm_answers[idx]
        actual_obj = data[idx]

        predicted_raw = getattr(predicted_obj, "answer", "")
        actual_raw = getattr(actual_obj, "answer", "")

        # Clean the answers by removing specified symbols
        predicted_clean = re.sub(pattern, "", predicted_raw)
        actual_clean = re.sub(pattern, "", actual_raw)

        try:
            predicted_value = float(predicted_clean)
            actual_value = float(actual_clean)
            difference = abs(predicted_value - actual_value)
            logger.info(
                f"Parsed Values [{idx}]: Predicted={predicted_value}, Actual={actual_value}, Difference={difference}"
            )

            if difference <= 1:
                correct += 1
                logger.debug(f"Result [{idx}]: Correct")
            else:
                logger.debug(f"Result [{idx}]: Incorrect")

        except ValueError:
            # If conversion fails, consider the answer incorrect
            logger.debug(f"Result [{idx}]: Incorrect (ValueError during conversion)")
            pass

        total += 1

    if total == 0:
        logger.warning("No valid answers to process.")
        return 0.0

    accuracy = (correct / total) * 100
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy
