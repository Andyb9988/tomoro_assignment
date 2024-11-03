import json
from logging import Logger

import dspy
from dspy.datasets import DataLoader

from config.config import (
    PipelineConfiguration,
    get_pipeline_config,
)
from src.tools.metrics import (
    get_answer_accuracy,
    get_reasoning_score,
)
from src.tools.models import OpenAILM
from src.tools.modules import OutputFinalAnswer
from src.utils.data_processsing import (
    DataParser,
    shuffle_and_split,
)
from src.utils.logging_utils import get_logger

APP_CONFIG: PipelineConfiguration = get_pipeline_config()

logger: Logger = get_logger(name=__name__)


with open("data/train.json") as f:
    data = json.load(f)

dl = DataLoader()
parser = DataParser()
lm = OpenAILM(model="gpt-4o")
dspy.configure(lm=lm)


split_data = shuffle_and_split(data=data, length=50, seed=10)
parsed_data = parser.process_data(split_data)

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


def get_llm_answers(data):
    llm_answers = []
    for i in data:
        obj = OutputFinalAnswer(context=i.context, id=i.id)
        ans_obj = obj.get_answer(question=i.question)
        llm_answers.append(ans_obj)

    return llm_answers


def main():
    llm_ans_list = get_llm_answers(eval_df)

    accuracy = get_answer_accuracy(data=eval_df, llm_answers=llm_ans_list)
    reasoning = get_reasoning_score(data=eval_df, llm_answers=llm_ans_list)

    print(accuracy)
    print(reasoning)


if __name__ == "__main__":
    main()
