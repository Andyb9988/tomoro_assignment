from typing import (
    Any,
    List,
    Optional,
)

import pandas as pd

from src.tools.metrics import (
    calculate_accuracy_from_df,
    get_answer_accuracy_df,
)


# Define a simple Answer class for testing purposes
class Answer:
    def __init__(self, answer):
        self.answer = answer


# Test when both lists are empty
def test_both_lists_empty():
    data: List[Any] = []
    llm_answers: List[Any] = []
    df = get_answer_accuracy_df(data, llm_answers)
    assert df.empty, "DataFrame should be empty when both input lists are empty."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when no answers are provided."


# Test when data list is empty
def test_one_list_empty_data_empty():
    data: List[Any] = []
    llm_answers: List[Any] = [Answer("100")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert df.empty, "DataFrame should be empty when data list is empty."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when data list is empty."


# Test when llm_answers list is empty
def test_one_list_empty_llm_answers_empty():
    data: List[Any] = [Answer("100")]
    llm_answers: List[Any] = []
    df = get_answer_accuracy_df(data, llm_answers)
    assert df.empty, "DataFrame should be empty when llm_answers list is empty."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when llm_answers list is empty."


# Test when lists have different lengths
def test_different_lengths():
    data: List[Any] = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers: List[Any] = [Answer("101"), Answer("199")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert (
        len(df) == 2
    ), "DataFrame should have length equal to the shortest input list."
    expected_results = ["correct", "correct"]
    assert (
        df["result"].tolist() == expected_results
    ), "All processed answers should be correct."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 100.0
    ), "Accuracy should be 100.0% when all processed answers are correct."


# Test when all answers are correct
def test_all_answers_correct():
    data: List[Any] = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers: List[Any] = [Answer("101"), Answer("199"), Answer("299")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 3, "DataFrame should have the same length as the input lists."
    expected_results = ["correct", "correct", "correct"]
    assert (
        df["result"].tolist() == expected_results
    ), "All answers should be marked as correct."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 100.0, "Accuracy should be 100.0% when all answers are correct."


# Test when all answers are incorrect
def test_all_answers_incorrect():
    data: List[Any] = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers: List[Any] = [Answer("102"), Answer("198"), Answer("302")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 3, "DataFrame should have the same length as the input lists."
    expected_results = ["incorrect", "incorrect", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "All answers should be marked as incorrect."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when all answers are incorrect."


# Test when some answers are correct and some are incorrect
def test_some_correct_some_incorrect():
    data: List[Any] = [Answer("100"), Answer("200"), Answer("300"), Answer("400")]
    llm_answers: List[Any] = [
        Answer("100.5"),
        Answer("202"),
        Answer("299"),
        Answer("401.2"),
    ]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 4, "DataFrame should have the same length as the input lists."
    expected_results = ["correct", "incorrect", "correct", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "Results should correctly reflect answer correctness."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 50.0
    ), "Accuracy should be 50.0% when half the answers are correct."


# Test answers containing specified symbols
def test_answers_with_symbols():
    data: List[Any] = [Answer("$100"), Answer("200£"), Answer("(300)"), Answer("400%")]
    llm_answers: List[Any] = [
        Answer("$101"),
        Answer("199£"),
        Answer("(299)"),
        Answer("401%"),
    ]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 4, "DataFrame should have the same length as the input lists."
    expected_results = ["correct", "correct", "correct", "correct"]
    assert (
        df["result"].tolist() == expected_results
    ), "All answers with symbols should be correctly processed."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 100.0
    ), "Accuracy should be 100.0% when all symbol-containing answers are correct."


# Test answers with mixed symbols and text
def test_answers_with_mixed_symbols_and_text():
    data: List[Any] = [
        Answer("$100"),
        Answer("Two Hundred"),
        Answer("(300)"),
        Answer("400%"),
    ]
    llm_answers: List[Any] = [
        Answer("$101"),
        Answer("200"),
        Answer("(299)"),
        Answer("four hundred"),
    ]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 4, "DataFrame should have the same length as the input lists."
    expected_results = ["correct", "incorrect", "correct", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "Mixed symbols and text should be correctly processed."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 50.0
    ), "Accuracy should be 50.0% when some mixed entries are correct."


# Test non-numeric answers
def test_non_numeric_answers():
    data: List[Any] = [Answer("one hundred"), Answer("200"), Answer("three hundred")]
    llm_answers: List[Any] = [Answer("100"), Answer("two hundred"), Answer("300")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 3, "DataFrame should have the same length as the input lists."
    expected_results = ["incorrect", "incorrect", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "Non-numeric answers should be marked as incorrect."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 0.0
    ), "Accuracy should be 0.0% when all answers are non-numeric and incorrect."


# Test edge case where difference is exactly one
def test_edge_case_difference_exactly_one():
    data: List[Any] = [Answer("100"), Answer("200")]
    llm_answers: List[Any] = [Answer("101"), Answer("199")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 2, "DataFrame should have the same length as the input lists."
    expected_results = ["correct", "correct"]
    assert (
        df["result"].tolist() == expected_results
    ), "Differences exactly one should be marked as correct."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 100.0
    ), "Accuracy should be 100.0% when differences are exactly one."


# Test edge case where difference is just over one
def test_edge_case_difference_just_over_one():
    data: List[Any] = [Answer("100"), Answer("200")]
    llm_answers: List[Any] = [Answer("101.1"), Answer("198.9")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 2, "DataFrame should have the same length as the input lists."
    expected_results = ["incorrect", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "Differences just over one should be marked as incorrect."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 0.0
    ), "Accuracy should be 0.0% when differences are just over one."


# Test when no valid answers are processed
def test_no_valid_answers():
    data: List[Any] = [Answer("abc"), Answer("def")]
    llm_answers: List[Any] = [Answer("ghi"), Answer("jkl")]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 2, "DataFrame should have the same length as the input lists."
    expected_results = ["incorrect", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "All invalid answers should be marked as incorrect."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when all answers are invalid."


# Test when processing stops due to one list being empty after minimum length
def test_total_zero_due_to_length_zero_after_min_length():
    data: List[Any] = [Answer("100")]
    llm_answers: List[Any] = []
    df = get_answer_accuracy_df(data, llm_answers)
    assert df.empty, "DataFrame should be empty when llm_answers list is empty."
    accuracy = calculate_accuracy_from_df(df)
    assert accuracy == 0.0, "Accuracy should be 0.0% when no answers are processed."


# Test with a large dataset
def test_large_dataset():
    data: List[Any] = [Answer(str(i)) for i in range(1000)]
    llm_answers: List[Any] = [Answer(str(i + 1)) for i in range(1000)]
    df = get_answer_accuracy_df(data, llm_answers)
    assert len(df) == 1000, "DataFrame should have 1000 entries for a large dataset."
    expected_results = ["correct"] * 1000
    assert (
        df["result"].tolist() == expected_results
    ), "All entries should be marked as correct in a large dataset."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 100.0
    ), "Accuracy should be 100.0% for a large dataset with all correct answers."


# Test with mixed valid and invalid entries
def test_mixed_valid_and_invalid_entries():
    data: List[Any] = [
        Answer("100"),
        Answer("$200"),
        Answer("three hundred"),
        Answer("(400)"),
        Answer("500%"),
    ]
    llm_answers: List[Any] = [
        Answer("101"),  # Correct
        Answer("199"),  # Correct
        Answer("300"),  # Incorrect (original data has "three hundred")
        Answer("401"),  # Correct
        Answer("five hundred"),  # Incorrect
    ]
    df = get_answer_accuracy_df(data, llm_answers)
    assert (
        len(df) == 5
    ), "DataFrame should have 5 entries for mixed valid and invalid data."
    expected_results = ["correct", "correct", "incorrect", "correct", "incorrect"]
    assert (
        df["result"].tolist() == expected_results
    ), "Results should correctly reflect mixed correctness."
    accuracy = calculate_accuracy_from_df(df)
    assert (
        accuracy == 60.0
    ), "Accuracy should be 60.0% when 3 out of 5 answers are correct."
