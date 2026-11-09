from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import SEED
from src.validation import BILLS, PAY, PAYMENTS


def add_features(frame):
    result = frame.copy()
    result["bill_mean"] = frame[BILLS].mean(axis=1)
    result["payment_mean"] = frame[PAYMENTS].mean(axis=1)
    result["max_delay"] = frame[PAY].max(axis=1)
    result["delayed_months"] = (frame[PAY] > 0).sum(axis=1)
    result["balance_ratio"] = result["bill_mean"] / frame["LIMIT_BAL"]
    for column in ["SEX", "EDUCATION", "MARRIAGE"]:
        result[column] = result[column].astype(int).astype(str)
    return result


def make_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, ["SEX", "EDUCATION", "MARRIAGE"]),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", transform),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
