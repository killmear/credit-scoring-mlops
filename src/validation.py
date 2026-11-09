import os

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY + BILLS + PAYMENTS
CATEGORIES = {"SEX": [1, 2], "EDUCATION": list(range(7)), "MARRIAGE": list(range(4))}


def validate_frame(frame):
    """Проверка входного контракта и набора правил GX."""
    if frame.empty or set(frame.columns) != set(FEATURES):
        raise ValueError("Ожидаются непустые данные и ровно 23 поля признаков")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES):
        raise ValueError("Все признаки должны быть числовыми")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Признаки должны быть конечными числами без пропусков")
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {"globally": False}
    asset = context.data_sources.add_pandas(name="input").add_dataframe_asset(name="credit")
    definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = definition.get_batch(batch_parameters={"dataframe": frame})
    suite = gx.ExpectationSuite(name="credit_input")
    for column in FEATURES:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    for column, values in CATEGORIES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=values)
        )
    limits = {"LIMIT_BAL": (1, 10000000), "AGE": (18, 100)}
    limits.update({c: (-2, 9) for c in PAY})
    limits.update({c: (-100000000, 100000000) for c in BILLS})
    limits.update({c: (0, 100000000) for c in PAYMENTS})
    for column, (low, high) in limits.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=low, max_value=high
            )
        )
    # Возраст и статусы задержек представлены целочисленными кодами.
    for column in ["AGE"] + PAY:
        if not (frame[column] % 1 == 0).all():
            raise ValueError(f"Поле {column} должно быть целочисленным")
    result = batch.validate(suite).to_json_dict()
    if not result["success"]:
        failed = [
            r["expectation_config"]["kwargs"].get("column")
            for r in result["results"]
            if not r["success"]
        ]
        raise ValueError(f"Нарушены правила Great Expectations: {failed}")
    return {"success": True, "rows": len(frame), "statistics": result["statistics"]}


if __name__ == "__main__":
    from src.config import ROOT, TARGET

    data = pd.read_csv(ROOT / "data/train.csv")
    print(validate_frame(data.drop(columns=["ID", TARGET])))
