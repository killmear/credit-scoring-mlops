import hashlib
import io
import json
import os
import ssl
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_SHA256, DATA_URL, ROOT, SEED, TARGET
from src.validation import FEATURES, validate_frame


def download():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    raw = folder / "credit.xls"
    if not raw.exists():
        context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        with urllib.request.urlopen(DATA_URL, context=context, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        name = next(n for n in archive.namelist() if n.endswith(".xls"))
        temporary = raw.with_suffix(".tmp")
        temporary.write_bytes(archive.read(name))
        temporary.replace(raw)
    if hashlib.sha256(raw.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Контрольная сумма исходного датасета изменилась")
    return raw


def prepare():
    raw = download()
    folder = ROOT / "data"
    frame = pd.read_excel(raw, header=1).rename(columns={"default payment next month": TARGET})
    if frame["ID"].duplicated().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Некорректные идентификаторы или целевая переменная")
    validation = validate_frame(frame[FEATURES])
    train, test = train_test_split(frame, test_size=0.2, random_state=SEED, stratify=frame[TARGET])
    for name, part in [("train", train), ("test", test)]:
        temporary = folder / f"{name}.tmp"
        part.to_csv(temporary, index=False)
        temporary.replace(folder / f"{name}.csv")
    report = {
        "source": DATA_URL,
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": SEED,
        "target_share": float(frame[TARGET].mean()),
        "missing": int(frame.isna().sum().sum()),
        "duplicate_ids": int(frame.ID.duplicated().sum()),
        "split_overlap": len(set(train.ID) & set(test.ID)),
        "validation": validation,
    }
    (ROOT / "reports/data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Подготовлены train {len(train)} и test {len(test)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        download()
    else:
        prepare()
