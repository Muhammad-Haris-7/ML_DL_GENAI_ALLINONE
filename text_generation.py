from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


def generate_text(prompt):

    result = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        truncation=True
    )

    return result[0]["generated_text"]