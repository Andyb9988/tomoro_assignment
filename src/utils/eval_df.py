import pandas as pd


def get_llm_answer_outcome_df(eval_df, llm_ans_list):
    # Prepare data from eval_df
    eval_data = []
    for example in eval_df:
        eval_data.append(
            {
                "question": example.question,
                "id": example.id,
                "step_list": example.step_list,
                "dialogue_break": example.dialogue_break,
                "answer": example.answer,
                "exe_answer": example.exe_answer,
            }
        )
    # Create a DataFrame from eval_data
    eval_df = pd.DataFrame(eval_data)

    # Prepare data from llm_ans_list
    llm_data = []
    for prediction in llm_ans_list:
        llm_data.append(
            {"reasoning": prediction.reasoning, "llm_answer": prediction.answer}
        )
    # Create a DataFrame from llm_data
    llm_df = pd.DataFrame(llm_data)

    # Merge eval_df with llm_df on 'id'
    outcome_df = pd.concat([eval_df, llm_df], axis=1)

    return outcome_df
