from typing import List
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from src.tools.metrics import (
    get_answer_accuracy,
    get_average_reasoning_score,
)


# Define a simple Answer class for testing purposes
class Answer:
    def __init__(self, answer):
        self.answer = answer


class DataItem:
    def __init__(self, context: str, dialogue_break: str):
        self.context = context
        self.dialogue_break = dialogue_break


class AnswerObj:
    def __init__(self, reasoning: str):
        self.reasoning = reasoning


class MockAssessment:
    def __init__(self, assessment_answer):
        self.assessment_answer = assessment_answer


@pytest.fixture
def mock_assess_reasoning(mocker):
    with patch("dspy.Predict") as mock_predict:
        # Configure the mock to return a MockAssessment when called
        mock_predict.return_value = MagicMock(return_value=MockAssessment(0.0))
        yield mock_predict  # I


# Test 1: Completely Different Reasoning (score < 0.5)
def test_average_reasoning_score_low(mock_assess_reasoning):
    # Arrange
    data = [
        DataItem(
            context="Client seeks investment advice for their retirement portfolio.",
            dialogue_break="The client wants to diversify their retirement portfolio to minimize risks and maximize returns over the next 20 years.",
        ),
        DataItem(
            context="Customer inquires about loan options for purchasing a new home.",
            dialogue_break="The customer is looking for a mortgage with the lowest possible interest rate and favorable repayment terms.",
        ),
        DataItem(
            context="User requests assistance with tax filing for their small business.",
            dialogue_break="The user needs help organizing their business expenses and understanding eligible tax deductions to reduce their tax liability.",
        ),
    ]

    llm_ans = [
        AnswerObj(
            reasoning="The client is planning to invest in high-risk stocks, calculating a 150% return within a year."
        ),
        AnswerObj(
            reasoning="The customer wants to calculate the monthly payments for a car loan instead of a mortgage."
        ),
        AnswerObj(
            reasoning="The user aims to invest business profits into cryptocurrency, projecting a 200% increase next quarter."
        ),
    ]

    # Configure the mock to return low reasoning accuracy scores (<0.5)
    mock_assess_reasoning.return_value.side_effect = [
        MockAssessment(0.3),
        MockAssessment(0.2),
        MockAssessment(0.4),
    ]

    # Act
    average_score = get_average_reasoning_score(data, llm_ans)

    # Assert
    assert average_score is not None, "Average score should not be None"
    assert average_score < 0.5, f"Expected average score < 0.5, got {average_score}"


# Test 2: Similar Reasoning (score > 0.8)
def test_average_reasoning_score_high(mock_assess_reasoning):
    # Arrange
    data = [
        DataItem(
            context="Client seeks investment advice for their retirement portfolio.",
            dialogue_break="The client wants to diversify their retirement portfolio to minimize risks and maximize returns over the next 20 years.",
        ),
        DataItem(
            context="Customer inquires about loan options for purchasing a new home.",
            dialogue_break="The customer is looking for a mortgage with the lowest possible interest rate and favorable repayment terms.",
        ),
        DataItem(
            context="User requests assistance with tax filing for their small business.",
            dialogue_break="The user needs help organizing their business expenses and understanding eligible tax deductions to reduce their tax liability.",
        ),
    ]

    llm_ans = [
        AnswerObj(
            reasoning="The client should allocate 60 percent of their retirement portfolio to low-risk bonds and 40% to diversified equities to achieve a balanced risk-return profile over 20 years."
        ),
        AnswerObj(
            reasoning="To secure a mortgage with a low interest rate, the customer should maintain a credit score above 750 and consider a 30-year fixed-rate mortgage to benefit from stable monthly payments."
        ),
        AnswerObj(
            reasoning="The user should categorize business expenses into deductible categories and utilize accounting software to track expenditures, thereby optimizing eligible tax deductions and reducing overall tax liability."
        ),
    ]

    # Configure the mock to return high reasoning accuracy scores (>0.8)
    mock_assess_reasoning.return_value.side_effect = [
        MockAssessment(0.9),
        MockAssessment(0.85),
        MockAssessment(0.95),
    ]

    # Act
    average_score = get_average_reasoning_score(data, llm_ans)
    print(f"Average Score {average_score}")
    # Assert
    assert average_score is not None, "Average score should not be None"
    assert average_score > 0.8, f"Expected average score > 0.8, got {average_score}"


def test_both_lists_empty():
    data = []
    llm_answers = []
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_one_list_empty_data_empty():
    data = []
    llm_answers = [Answer("100")]
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_one_list_empty_llm_answers_empty():
    data = [Answer("100")]
    llm_answers = []
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_different_lengths():
    data = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers = [Answer("101"), Answer("199")]
    # Only the first two pairs are processed
    # Differences: |101 - 100| = 1 (correct), |199 - 200| = 1 (correct)
    # Accuracy: (2/2)*100 = 100.0
    assert get_answer_accuracy(data, llm_answers) == 100.0


def test_all_answers_correct():
    data = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers = [Answer("101"), Answer("199"), Answer("299")]
    # Differences: 1,1,1 → all correct
    assert get_answer_accuracy(data, llm_answers) == 100.0


def test_all_answers_incorrect():
    data = [Answer("100"), Answer("200"), Answer("300")]
    llm_answers = [Answer("102"), Answer("198"), Answer("302")]
    # Differences: 2,2,2 → all incorrect
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_some_correct_some_incorrect():
    data = [Answer("100"), Answer("200"), Answer("300"), Answer("400")]
    llm_answers = [Answer("100.5"), Answer("202"), Answer("299"), Answer("401.2")]
    # Differences: 0.5 (correct), 2 (incorrect), 1 (correct), 1.2 (incorrect)
    # Correct: 2 out of 4 → 50.0
    assert get_answer_accuracy(data, llm_answers) == 50.0


def test_answers_with_symbols():
    data = [Answer("$100"), Answer("200£"), Answer("(300)"), Answer("400%")]
    llm_answers = [Answer("$101"), Answer("199£"), Answer("(299)"), Answer("401%")]
    # Cleaned data: "100", "200", "300", "400"
    # Cleaned llm_answers: "101", "199", "299", "401"
    # Differences: 1,1,1,1 → all correct
    assert get_answer_accuracy(data, llm_answers) == 100.0


def test_answers_with_mixed_symbols_and_text():
    data = [Answer("$100"), Answer("Two Hundred"), Answer("(300)"), Answer("400%")]
    llm_answers = [
        Answer("$101"),
        Answer("200"),
        Answer("(299)"),
        Answer("four hundred"),
    ]
    # Cleaning:
    # data: "100", "Two Hundred", "300", "400"
    # llm: "101", "200", "299", "four hundred"
    # Conversion:
    # Pair 0: 101 - 100 = 1 (correct)
    # Pair 1: float("Two Hundred") → ValueError (incorrect)
    # Pair 2: 299 - 300 = 1 (correct)
    # Pair 3: float("four hundred") → ValueError (incorrect)
    # Correct: 2 out of 4 → 50.0
    assert get_answer_accuracy(data, llm_answers) == 50.0


def test_non_numeric_answers():
    data = [Answer("one hundred"), Answer("200"), Answer("three hundred")]
    llm_answers = [Answer("100"), Answer("two hundred"), Answer("300")]
    # Cleaning:
    # data: "one hundred", "200", "three hundred"
    # llm: "100", "two hundred", "300"
    # Conversion:
    # Pair 0: float("one hundred") → ValueError (incorrect)
    # Pair 1: float("two hundred") → ValueError (incorrect)
    # Pair 2: 300 - float("three hundred") → ValueError (incorrect)
    # Correct: 0 out of 3 → 0.0
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_edge_case_difference_exactly_one():
    data = [Answer("100"), Answer("200")]
    llm_answers = [Answer("101"), Answer("199")]
    # Differences: 1,1 → both correct
    assert get_answer_accuracy(data, llm_answers) == 100.0


def test_edge_case_difference_just_over_one():
    data = [Answer("100"), Answer("200")]
    llm_answers = [Answer("101.1"), Answer("198.9")]
    # Differences: 1.1, 1.1 → both incorrect
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_no_valid_answers():
    data = [Answer("abc"), Answer("def")]
    llm_answers = [Answer("ghi"), Answer("jkl")]
    # All conversions fail
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_total_zero_due_to_length_zero_after_min_length():
    data = [Answer("100")]
    llm_answers = []
    # Since llm_answers is empty, accuracy should be 0.0
    assert get_answer_accuracy(data, llm_answers) == 0.0


def test_large_dataset():
    data = [Answer(str(i)) for i in range(1000)]
    llm_answers = [Answer(str(i + 1)) for i in range(1000)]
    # Each difference is 1 → all correct
    assert get_answer_accuracy(data, llm_answers) == 100.0


def test_mixed_valid_and_invalid_entries():
    data = [
        Answer("100"),
        Answer("$200"),
        Answer("three hundred"),
        Answer("(400)"),
        Answer("500%"),
    ]
    llm_answers = [
        Answer("101"),  # Correct
        Answer("199"),  # Correct
        Answer("300"),  # Correct (ignores "three hundred" → ValueError → incorrect)
        Answer("401"),  # Correct (difference 1)
        Answer("five hundred"),  # ValueError → incorrect
    ]
    # Processing up to min length (5)
    # Pair 0: 101 - 100 = 1 → correct
    # Pair 1: 199 - 200 = 1 → correct
    # Pair 2: 300 - "three hundred" → ValueError → incorrect
    # Pair 3: 401 - 400 = 1 → correct
    # Pair 4: "five hundred" - 500 → ValueError → incorrect
    # Correct: 3 out of 5 → 60.0
    assert get_answer_accuracy(data, llm_answers) == 60.0
