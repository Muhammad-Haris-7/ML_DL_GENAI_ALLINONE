from transformers import pipeline

ner_model = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

def extract_entities(text):

    results = ner_model(text)

    entities = []

    for item in results:

        entities.append({
            "word": item["word"],
            "entity": item["entity_group"],
            "score": round(float(item["score"]), 3)
        })

    return entities