import dspy


class GenerateCotAnswer(dspy.Signature):
    """
    Use the provided context to analyse and compute the final answer to the
    financial question using numerical reasoning.
    """

    question = dspy.InputField(desc="Financial question requiring calculations.")
    context = dspy.InputField(desc="Relevant report section")
    answer = dspy.OutputField(
        desc="Please provide only the final numeric result in decimal."
    )


class AssessReasoning(dspy.Signature):
    """Assess the quality of the LLM's numerical reasoning steps compared to the correct steps.
    Provide answer between 1-10 where 1 is not similar and 10 is very similar
    """

    actual_reasoning = dspy.InputField(desc="Actual Steps")
    context = dspy.InputField(desc="The Context")
    llm_reasoning = dspy.InputField(desc="LLM Reasoning Steps")
    assessment_answer = dspy.OutputField(desc="must be an integer between 1-10")
