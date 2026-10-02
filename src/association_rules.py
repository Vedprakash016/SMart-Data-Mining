import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def mine_rules(df, min_support=0.03, min_confidence=0.25):
    basket = (
        df.assign(value=1)
          .pivot_table(
              index="TransactionID",
              columns="Product",
              values="value",
              aggfunc="max",
              fill_value=0
          )
          .astype(bool)
    )

    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return basket, frequent, pd.DataFrame()

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules = rules.sort_values(
            ["lift", "confidence"], ascending=False
        ).reset_index(drop=True)

    return basket, frequent, rules
