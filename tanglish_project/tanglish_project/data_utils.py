"""
Loads the DravidianCodeMix Tamil-English offensive language dataset
and converts it into a clean binary classification format:
    0 = Not Offensive
    1 = Offensive

Dataset source: Chakravarthi et al., "DravidianCodeMix: Sentiment Analysis
and Offensive Language Identification Dataset for Dravidian Languages in
Code-Mixed Text", Language Resources and Evaluation, 2022.
https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset
"""

import pandas as pd

DATA_DIR = "dataset/DravidianCodeMix"


def load_split(name):
    """name is one of 'train', 'dev', 'test'"""
    path = f"{DATA_DIR}/tamil_offensive_full_{name}.csv"
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label", "_extra"],
        on_bad_lines="skip",
        quoting=3,
        usecols=[0, 1],
    )
    df = df.dropna(subset=["text", "label"])

    # Collapse the 6-way label set into binary offensive / not-offensive.
    # "not-Tamil" comments are dropped -- they're neither Tamil nor
    # Tanglish, so they're out of scope for this classifier.
    df = df[df["label"] != "not-Tamil"]
    df["binary_label"] = (df["label"] != "Not_offensive").astype(int)

    return df[["text", "binary_label"]].reset_index(drop=True)


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        d = load_split(split)
        print(split, d.shape, d["binary_label"].value_counts().to_dict())
