from transformers import pipeline

qa_model = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def answer_question(question, context):

    result = qa_model(
        question=question,
        context=context
    )

    return result["answer"]