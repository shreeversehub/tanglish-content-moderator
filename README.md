# 🗣️ Tanglish Content Moderator 🤖

A real-time classifier that detects trolling / offensive comments written
in **Tanglish** (Tamil-English code-mixed text) — the way most Tamil
social media users actually type. Built for Avalon OpenHack.

## The Problem

Platform moderation filters (YouTube, Instagram, etc.) are trained almost
entirely on English. Offensive Tanglish comments — e.g. Tamil insults
spelled in Latin script, mixed mid-sentence with English — routinely slip
past these filters, because the filter isn't built to understand this
language combination at all.

## What This Does

Given any Tanglish comment, the model:
1. Classifies it as **Offensive** or **Not Offensive**
2. Shows *which specific text patterns* triggered the decision (explainability)
3. Logs it to a live "moderator dashboard"

This is a continuous classifier meant to run on a stream of incoming
content — not a chatbot you manually query per comment.

## Dataset

[DravidianCodeMix](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset)
(Chakravarthi et al., *Language Resources and Evaluation*, 2022) —
~35,000 manually-annotated Tamil-English YouTube comments, labeled for
offensive content. This is the same dataset used in the HASOC-Dravidian-
CodeMix shared tasks (2020-2021), so results are directly comparable to
published academic baselines.

```bibtex
@article{Chakravarthi2022,
  author  = {Chakravarthi, Bharathi Raja and Priyadharshini, Ruba and
             Muralidaran, Vigneshwaran and Jose, Navya and
             Suryawanshi, Shardul and Sherly, Elizabeth and McCrae, John P.},
  title   = {DravidianCodeMix: sentiment analysis and offensive language
             identification dataset for Dravidian languages in code-mixed text},
  journal = {Language Resources and Evaluation},
  year    = {2022}
}
```

## Approach

- **Model:** TF-IDF (character 3-5 grams) + Logistic Regression
- **Why character n-grams:** Tanglish has no fixed spelling ("semma" /
  "sema" / "simma" all mean the same thing) — character-level features
  handle this spelling variation far better than word-level features
- **Why this model for a hackathon, not a transformer:** trains in
  seconds on a CPU, zero GPU/internet dependency at runtime, and is
  fully explainable (you can see exactly which n-grams drove each
  decision) — a fair, honest MVP baseline, not a final production model

## Results (on the official test split)

| Metric | Value |
|---|---|
| Overall accuracy | 82% |
| Offensive-class F1 | 0.68 |
| Offensive-class recall | 77% |

For context: published transformer-based approaches (mBERT, XLM-R,
MuRIL) on this same dataset report 80-90% F1 — this lightweight baseline
is in a reasonable, honest range for what it is, with clear room to
improve (see below).

## How to Run

```bash
pip install -r requirements.txt
python3 train.py      # trains and saves tanglish_model.joblib (~10 sec)
python3 app.py         # starts the demo at http://localhost:5000
```

Then open `http://localhost:5000`, type a Tanglish comment, and watch it
get classified live, with the moderator dashboard updating below.

## Project Structure

```
tanglish_project/
├── dataset/                  # DravidianCodeMix data (train/dev/test)
├── data_utils.py             # loads + binarizes the dataset
├── train.py                  # trains the TF-IDF + LogReg model
├── predict.py                # classification + explainability
├── app.py                    # Flask backend for the live demo
├── templates/index.html      # mock comment section + dashboard UI
└── requirements.txt
```

## What's Already Established (Prior Research)

Academic work (HASOC-Dravidian-CodeMix shared tasks, 2020-2021) already
proves this classification task is feasible at 80-90% F1 using heavier
transformer models. This project isn't claiming to invent the task —
it's building the missing deployable layer: a working, explainable,
end-to-end pipeline you can actually run and demo, which the research
papers stop short of.

## Future Enhancements

- Swap the baseline for a fine-tuned transformer (MuRIL / XLM-RoBERTa)
  to close the gap to published F1 scores
- Add severity levels (mild / moderate / severe) instead of binary
- Human-in-the-loop moderator override + active learning from corrections
- Extend to Manglish (Malayalam-English) and Kanglish (Kannada-English)
  using the same pipeline — the dataset already includes both
- Real deployment as a browser extension or platform-side webhook
- On-device inference for privacy-preserving moderation
