"""
Loads the trained model and classifies new Tanglish text, plus gives a
simple explanation of *why* -- which character n-grams pushed the
decision toward "Offensive". This is what makes the demo defensible in
a viva/judging round: you're not just showing a black-box label, you're
showing the evidence behind it.
"""

import joblib

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load("tanglish_model.joblib")
    return _pipeline


def classify(text: str, top_k: int = 5):
    pipeline = _get_pipeline()
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    x = vectorizer.transform([text])
    proba = pipeline.predict_proba([text])[0]
    label = "Offensive" if proba[1] >= 0.5 else "Not Offensive"
    confidence = float(max(proba))

    # Explainability: for every n-gram present in this text, multiply its
    # TF-IDF weight by its learned coefficient. The highest-scoring ones
    # are the strongest evidence the model used for its decision.
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    x_arr = x.toarray()[0]
    contributions = []
    for idx in x_arr.nonzero()[0]:
        contributions.append((feature_names[idx], x_arr[idx] * coefs[idx]))
    contributions.sort(key=lambda t: t[1], reverse=True)

    top_positive = [c for c in contributions if c[1] > 0][:top_k]

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 3),
        "top_signals": [{"ngram": ng, "weight": round(w, 3)} for ng, w in top_positive],
    }


if __name__ == "__main__":
    samples = [
        "semma video da, romba pudichirundhu!",
        "nee romba mokka da, po thooki eri",
        "great content bro keep it up",
        "un mugam paakka mudiyala, azhinga po",
    ]
    for s in samples:
        result = classify(s)
        print(result)
