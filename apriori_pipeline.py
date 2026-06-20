import pandas as pd

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def run_apriori_pipeline(csv_path):

    df = pd.read_csv(csv_path)

    transactions = []

    for _, row in df.iterrows():

        transaction = []

        for item in row:

            if pd.notna(item):

                transaction.append(str(item))

        transactions.append(transaction)

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    basket = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    frequent_itemsets = apriori(
        basket,
        min_support=0.2,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    if rules.empty:

        return []

    results = []

    for _, row in rules.iterrows():

        results.append({

            "antecedents":
            ", ".join(list(row["antecedents"])),

            "consequents":
            ", ".join(list(row["consequents"])),

            "support":
            round(row["support"], 3),

            "confidence":
            round(row["confidence"], 3),

            "lift":
            round(row["lift"], 3)

        })

    return results