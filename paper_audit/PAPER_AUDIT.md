# Paper audit: "HinglishSent" (`research_paper.tex`)

Audit of every quantitative claim in the paper against what the repository can actually support.
The paper was removed from the working tree in `e49e660`; recover it with
`git show e9791dc:research_paper.tex > research_paper.tex`.

Every number marked **[verified]** below can be re-derived with
`python paper_audit/verify_from_artifacts.py`, which uses only committed files. Numbers the repo
cannot produce are listed as **[unsupported]**. Those need a source, or they need to come out of the paper.

---

## How to use this with Claude

Paste this whole file together with `research_paper.tex` (and
`paper_audit/results/hinglish_human_label_check.json`) and ask:

> Revise research_paper.tex using PAPER_AUDIT.md. Apply every correction in sections 1–6, and add
> the human-label experiment from section 1b as a new results subsection with its caveats. Delete every
> claim listed as [unsupported] unless I give you a source for it. Reframe the contribution as in
> section 7. Do not invent any new numbers. Where a number must come from the verification run
> (section 8), leave a visible `\todo{}` placeholder instead.

When `results/verification/VERIFICATION_REPORT.md` exists (from
`phase3_transformer_models/Phase3_DistilBERT_Verification_Colab.ipynb`), paste it as well so the
placeholders can be filled with real numbers.

---

## 1. The central problem: the labels are VADER output, not sentiment

`phase1_data_engineering/04_sentiment_labeling.py` labels every comment with VADER's compound
score (≥0.05 Positive, ≤−0.05 Negative). No human labels exist anywhere in the repo. That means:

- **DistilBERT's 96.47% means it agrees with VADER 96.47% of the time.** It doesn't measure
  sentiment accuracy. The same applies to every baseline. The paper must say "agreement with VADER
  labels" or "accuracy against VADER-derived silver labels" wherever it reports accuracy.
- **VADER is an English-only lexicon. It can't read Hindi.** From the Phase 1 EDA output **[verified]**:

  | Language (Phase 1 detector) | n | Negative | Neutral | Positive |
  |---|---:|---:|---:|---:|
  | English | 124,157 | 16.7% | 40.6% | 42.7% |
  | Code-Mixed | 2,865 | 3.8% | **81.6%** | 14.7% |
  | Hindi (Devanagari) | 4,499 | 0.5% | **97.4%** | 2.1% |
  | Unknown | 87 | 5.7% | 82.8% | 11.5% |

  97.4% of Hindi comments are "Neutral" because VADER finds no words it knows, not because Hindi
  speakers are neutral about unemployment. A model can score highly on these comments by
  learning "Devanagari ⇒ Neutral".
- **VADER mislabels romanized Hindi in both directions** **[verified]**:
  - Hindi words that are also English lexicon entries get spurious polarity: `kia` (*kiya*, "did")
    = −3.2 (VADER reads "killed in action"), `bad` (*baad*, "after") = −2.5, `ache` (*acche*,
    "good") = −1.6, `lag` ("seems") = −1.4, `ha` (*haan*, "yes") = +1.4.
  - Hindi sentiment words are missing entirely: *bekar, badhiya, ghatiya, bakwas, accha,
    berozgari, dhoka*.
  - On 8 hand-written Hindi/Hinglish examples VADER was wrong on 8/8. For example, "ghatiya policy,
    sab bakwas" (pathetic policy, all rubbish) → Neutral; "Exam ke bad result kab aayega" (when
    will the result come after the exam) → Negative.
- **The trained models learned these artifacts** **[verified]**. The committed Linear SVM's top
  Negative features include `kia`, `lag`, `ache`, `po`, `ke bad`, i.e. Hindi words that are
  "negative" only through VADER collisions.
- **The paper contradicts itself here.** Section III claims "Manual annotation of 500 random samples
  showed 78.4% agreement with VADER labels (Cohen's κ = 0.67)". The Limitations and Future Work
  sections say manual annotation is *future work* "to estimate VADER accuracy". No annotation
  file, script or result exists in the repo. **[unsupported] — remove it**, or run the annotation
  for real (the verification notebook exports a 300-comment stratified sample and scores both VADER
  and DistilBERT against it).

**Required changes:** reword all accuracy claims as agreement with VADER silver labels. Move
"VADER can't read Hindi/Hinglish" from a minor limitation to the main limitation, with the table
above. Delete "extends VADER to code-mixed Hinglish scenarios with validation". Delete or replace the
500-sample annotation claim.

## 1b. Measured: VADER and your models against HUMAN labels on Hinglish

Your YouTube data has no human labels, so I used a public human-annotated Hinglish set: the
IIITH Codemixed dataset from Joshi et al. (COLING 2016), 3,879 romanized Hindi-English comments
(Facebook, public pages). I ran your own Phase 1 cleaning, language detector and VADER labelling
on it, plus your committed Phase 2 models. Reproduce with
`python paper_audit/hinglish_human_label_check.py`; output in `paper_audit/results/`.

| Labeller / model (scored against human labels) | Accuracy | Macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| Always predict the majority class (Neutral) | 50.45% | 22.4% | 0.00 |
| **VADER, exactly as `04_sentiment_labeling.py`** | **50.63%** | **41.24%** | **0.12** |
| **Your committed Linear SVM** (trained on VADER-labelled YouTube data) | **51.02%** | 38.79% | 0.11 |
| Your committed Logistic Regression | 51.51% | 37.89% | 0.11 |
| Your committed Naive Bayes / RF / GBoost | 52.8–53.2% | 33–36% | 0.09–0.11 |
| Same TF-IDF + LinearSVC, 5-fold CV, **trained on VADER labels** | 51.38% | 38.36% | 0.10 |
| Same TF-IDF + LinearSVC, 5-fold CV, **trained on human labels** | **67.08%** | **63.85%** | **0.44** |

What this shows:
- **On romanized Hinglish, VADER is no better than always guessing "Neutral"** (50.63% vs 50.45%,
  κ = 0.12, "slight" agreement). 66.5% of comments get a compound score of exactly 0.
  Negative F1 is only 21%: VADER finds 88 of 570 human-negative comments.
- **Models trained on VADER labels inherit this.** Your committed SVM agrees with VADER on
  **90.4%** of these comments but with humans on only **51.0%**.
- **The evaluation hides it.** Trained on VADER labels and scored against VADER labels (the paper's
  methodology), the same classifier reports **85.25%**. Scored against humans it gets **51.38%**.
  Trained on human labels it gets **67.08%**. The ~34-point gap is purely an artifact of
  evaluating against the labeller you trained on.
- **Your language detector calls 98.1% of this Hinglish "English"** (1.9% Code-Mixed).

**How much of your corpus this affects:** the committed TF-IDF vectorizer stores each word's
document frequency in your real 105,286-comment training set. The Hindi word **"hai" appears in
10.0% of training comments (10,544)**, more than "job" (8.1%). Other markers: *bhi* 4.2%, *ke*
4.2%, *ki* 4.2%, *kya* 2.5%, *nahi* 2.4%. So **at least ~10% of the corpus is romanized
Hinglish** (likely more, since not every Hinglish comment contains "hai"), on top of 3.4%
Devanagari. On both, the labels are near chance against humans.
(`python paper_audit/verify_from_artifacts.py`, section 5b.)

**Caveats to state in the paper:** the human-labelled set is Facebook comments on celebrity and
political pages, not employment videos on YouTube, so treat these numbers as indicative of
VADER's behaviour on romanized Hinglish, not as your corpus's exact error rate. VADER is much better on
genuine English text, so overall label quality on your corpus is higher than 51%. The honest way to
get your corpus's number is the 300-comment human annotation in section 8.

**Paper text this supports (suggested):** "On a public human-annotated Hindi-English corpus
(Joshi et al., 2016; n = 3,879), VADER labels agree with human annotators on 50.6% of comments
(κ = 0.12), no better than the majority-class baseline (50.5%). A TF-IDF + LinearSVC classifier
trained on VADER labels reaches 85.3% agreement with VADER but only 51.4% accuracy against
humans; the same classifier trained on human labels reaches 67.1%. At least 10% of our training
comments contain the Hindi copula *hai*, so a substantial portion of the corpus carries
near-chance silver labels."

## 2. The dataset isn't mostly Hinglish, and the language numbers are mislabelled

- By the paper's own detector, **94.34% of comments are "English"**, with Hindi at 3.42% and
  Code-Mixed at 2.18% **[verified]**. A title and abstract built on "Hinglish" / "code-mixed"
  overstate this.
- The detector (`03_language_detection.py`) classifies by **script**. Romanized Hindi ("ye sarkar
  bekar hai") is Latin script, so it's counted as *English*. "Code-Mixed" means Devanagari + Latin in one
  comment. That's a narrow definition, and it misses the most common form of Hinglish. The paper should say
  so explicitly. Measured: "hai" alone appears in **10.0%** of training comments, so romanized
  Hinglish is at least ~10% of the corpus, about 5× the 2.18% reported as code-mixed (section 1b).
  The sample comment printed in the Colab notebook ("Dhinilo qualify aithe asalu yem
  work istharu salary...") is romanized **Telugu**, so the corpus also contains other Indian
  languages in Latin script, which the paper never mentions.
- The paper's Algorithm 1 / Eq. (language_classification) **don't match the code**:
  - Paper: Code-Mixed if both ratios > 0.1; Latin-only text that isn't `en` per langdetect → Unknown.
  - Code: Code-Mixed if both ratios ≥ **15%**; Hindi if Devanagari ≥ **80%**; English if Latin ≥
    **80%**; otherwise langdetect, and any non-`en`/`hi` result → **Code-Mixed** (not Unknown).
  Rewrite the algorithm from the code.
- "Language detection F1-score: 94.2% (validated on 1,000 manually annotated samples)" —
  **[unsupported]**. Nothing in the repo supports it. Remove.
- "The Devanagari detector captures Hindi script with 100% precision" — Unicode U+0900–097F also
  covers Marathi, Nepali, Sanskrit etc. Remove "100% precision" or qualify it.
- "while the English detector handles transliterated Hindi" is false. It counts transliterated
  Hindi as English. Remove.

## 3. Wrong or inconsistent numbers

| Where in paper | Paper says | Correct (source) |
|---|---|---|
| Abstract, Intro, Dataset | "spanning 6 years (2019–2025)" | Comments run **Oct 2011 – Dec 2025**. **52.2% are from 2025**, 71.8% from 2024–25; 0.64% pre-2019 [verified, `temporal_trends.html`]. Describe it as heavily skewed to 2024–25, not a 6-year longitudinal set. Cut the "COVID-era temporal" framing unless analysed. |
| Phase 4 results, "DistilBERT Confusion Matrix" table + bullets (264, 244, 380; "Neutral recall 96.47%") | the 90.41% matrix | That table is from an **older 90.41% run** (`phase3_transformer_models/evaluation/distilbert_results.txt`), while every other table reports the **96.47%** run. Replace it with the 96.47% matrix `[[1932,37,119],[63,5552,106],[96,44,5212]]` and rewrite the bullets: Neg→Neu 37 (1.77%), Neg→Pos 119 (5.70%), Neu→Pos 106 (1.85%), Pos→Neg 96 (1.79%); Neutral recall 97.05% [verified]. |
| "Per-Class Performance: Linear SVM" table | 85.13/64.42/73.34 ... | These are the **Logistic Regression** numbers (`phase2_baseline_models/evaluation/baseline_results.txt`). Either retitle as Logistic Regression or get the real SVM per-class report (not committed). |
| "Performance Improvement: DistilBERT over Linear SVM" table + abstract "+26.0% negative F1" | SVM F1 73.34 → 92.46, "+26.0%" | Baseline is LogReg, not SVM (see above). +26.0% is *relative*; the absolute gain is **+19.12 pp**. The same table's "+7.90%" accuracy row is *absolute* pp. Use pp throughout and label it. |
| Abstract, Discussion, Conclusion "7.90% improvement" | "7.90%" | **+7.90 percentage points** (relative +8.92%) [verified]. The Colab notebook printed "Relative improvement: 891.59%", which is a bug (`improvement` was already ×100). Don't quote it. |
| Fig. baseline_comparison (pgfplots) | RF 73.16, F1: LR 86.92, SVM 88.23, RF 73.15, GB 75.08, NB 70.34 | Those "F1" bars match no file. Per `model_comparison.csv`: weighted F1 LR 86.90, SVM 88.35, RF 68.86, GB 73.54, NB 70.90; RF accuracy **73.13**. The caption "88.23% F1" is wrong too. |
| Table baseline_results | RF 73.16 / 58.68 / 68.92; SVM time 5.46s; GB 287.87s | CSV: RF **73.13 / 58.59 / 68.86**; SVM train time **2.18s**, GB **114.86s**, RF **2.16s**. |
| Language-specific accuracy (English 90.8 / Hindi 87.3 / Code-mixed 85.1, "5.7% drop") | — | **[unsupported]**. No per-language evaluation exists in the repo, and these values sit around the *90.4%* model, not 96.47%. Remove. Replace with output of the verification run (it produces this table). |
| Error analysis "Sarcasm 34.2%, Code-mixed 18.7%, Neutral confusion 21.3%" | — | **[unsupported]**. No sarcasm detector or code-mixed error breakdown exists. Neutral→other is actually **2.95%** for the 96.47% model. Remove. |
| "Neutral class confusion: 21.3% (boundary ambiguity)" | — | Contradicted by the real 2.95%. |
| "Cleaning: 22.71s, 5,795/s; Language detection 12.20s, 10,791/s; Labeling 36.30s; Export 2.30s..." | — | **[unsupported]**: no logs committed. Low value anyway. Recommend cutting the throughput table. |
| "HTML entities decoded: 2,214; URLs removed: 683; Emojis 43,263" | — | **[unsupported]** (no logs). Keep only if you re-run Phase 1 and log them. |
| README / reports | SVM macro-F1 86.63%, "+9.00 pp macro-F1" | CSV says SVM macro-F1 **85.89%**, so the DistilBERT gain is **+9.74 pp** [verified]. (The paper's own table uses 85.89 correctly; fix the README and `phase4_explainability/reports/*`.) |
| `phase2_baseline_models/README.md` | RF 85.91%, "XGBoost 86.45%" | A third, different set of numbers, and there's no XGBoost in the code. Fix the README. |

Also state the **majority-class baseline: 43.47%** (always predict Neutral) and give a 95% CI for
headline accuracy: **96.47% ± 0.32 pp** (binomial, n = 13,161) [verified].

## 4. Method descriptions that don't match the code / notebook

| Paper | What was actually run |
|---|---|
| Cleaning: `demojize` emojis → text, then `lowercase` | `02_clean_data.py` **removes** emojis (`mode='remove'`) and doesn't lowercase (DistilBERT-uncased lowercases internally; TF-IDF lowercases by default). Fix Eqs. clean4/clean6. Removing emojis throws away a strong sentiment signal; mention it as a limitation. |
| URL regex `https?://[^\s]+` | Different regex (also strips `www.`). Minor. |
| Stratified split "preserving sentiment **and language**", StratifiedShuffleSplit | `05_train_test_split.py` stratifies by **sentiment only** using `train_test_split` twice. |
| TF-IDF: not mentioned | `stop_words='english'` is set. This **deletes "not", "no", "never", "nothing", "nor", "none", "cannot", "without"**, so the baselines can't see negation [verified]. That's a big handicap for the Negative class and inflates the transformer's margin. Disclose it (better: re-run baselines without stop-word removal). |
| LogReg: elastic net, `solver='saga'` | `solver='lbfgs'`, plain L2 (`C=1.0`) [verified from pickle]. Delete the elastic-net equation. |
| Linear SVM: `loss='hinge'`, `dual=False` | `LinearSVC(random_state=42)` defaults: **squared hinge**, `dual='auto'` [verified]. |
| DistilBERT: batch 8 × grad-accum 2, 5 epochs, early stopping (patience 3), Algorithm 3 selects on val **loss** | Colab notebook: **batch 32, no grad accumulation, 3 epochs, no early stopping**, best checkpoint by val **macro-F1**, fp16, max_length 256, warmup 500. Rewrite the hyperparameter list and Algorithm 3. |
| "Xavier initialization" of classifier head | HF default init (normal, std 0.02). Remove. |
| "FP16 gradient scaling factor 65536" | Dynamic loss scaling. Remove the specific number. |
| Hardware: i7-12700K, RTX 3070 | DistilBERT was trained on a **Colab Tesla T4** (notebook output). Fix the hardware section. |
| Software: PyTorch 2.4.1, Transformers 4.36.0, scikit-learn 1.3.2 | Model config says transformers **4.57.3**; pickles say scikit-learn **1.7.2 / 1.8.0** [verified]. Report real versions. |
| MuRIL described as one of "two transformer approaches" | No MuRIL training script exists. `evaluation/muril_results.txt` differs from the old 90.41% DistilBERT result in **2 cells out of 13,161** [verified], so it is almost certainly not an independent MuRIL run. **Remove MuRIL** or run it (the notebook has the command). |
| Phase 4: SHAP / DeepSHAP equations; Fig. shap_summary.png and attention_heatmap.png | **SHAP is never computed.** `02_shap_analysis.py` only writes a text description of SHAP plus averaged attention weights. Neither figure file exists. Remove the SHAP/DeepSHAP claims and figures; describe what was done (attention-weight token importance), and cite Jain & Wallace on why attention isn't explanation. |
| SVM top features "good job, excellent, congratulations / worst, terrible, useless" and Neutral "india, government, unemployment, data" | Actual top features [verified]: Negative `ban, kia, lag, worst, scam, hell, bad, ache, fake, block` (VADER artifacts); Neutral `pls reply, want pay, maharaj, create jobs, like hell, ha ha, really happy` (phrases where VADER's scores cancel to ~0). Replace with the real lists, since they're evidence for Section 1. |

## 5. Phase 5 deployment claims with nothing behind them

All **[unsupported]**: no load-test output, no Locust file, no quantization, no cache exist in the repo.

- Locust framework: `03_load_testing.py` uses **aiohttp**, not Locust.
- p50/p95/p99/max latencies, 294 RPS, latency table, load-testing figure: no results committed.
  The numbers also contradict each other. The paper says 2 CPU cores and 45 ms per single request
  (≈22 req/s per core), yet 294 RPS. Its own batching table tops out at 120 samples/s, and batching
  should be *faster* than single requests. The README claims "50+ requests/second".
- INT8 quantization (4×, 1.8× speedup, <0.5% loss): no quantization code. Remove.
- LRU cache, "hit rate 23.7%": no cache code. Remove.
- Docker image sizes, build times, cold start, memory: no measurements. The paper also gives two different
  base images (python:3.9-slim, 1.5 GB vs python:3.11-slim, 2.1 GB).
- "Prometheus/Grafana monitoring integration": only a prometheus.yml exists. Describe it accurately.

Either run `phase5_production_deployment/03_load_testing.py` and report what it prints (with the
machine spec), or cut Phase 5 to a short "deployment" paragraph without numbers.

## 6. Figures, LaTeX and references

- `phase3_transformer_models/figures/training_history.png` is **synthetic**: hand-typed values
  in `generate_visualizations.py` (this version ends at 0.904, contradicting 96.47%).
  `learning_curves_data_efficiency.png` is **synthetic** (hard-coded arrays).
  `confusion_matrix_distilbert.png` is **simulated** (random errors at 90.4%; it matches neither real
  matrix). Don't use any of these. The verification run saves the real `training_log_history.csv`.
- `generate_visualizations.py` hard-codes SVM per-class values (0.8461/0.7002/...) that match no
  results file.
- **The .tex doesn't compile:** 34 of the 37 `\cite{}` keys are undefined (the bibliography only defines
  `ref1`–`ref13`, with author lists that don't match the in-text citations); `pgfplots` isn't
  loaded but `axis` is used twice; `\argmin`/`\argmax` aren't defined (`\DeclareMathOperator*`);
  two figure files are missing. Duplicate sections: "Phase 4: Explainability Analysis" and "Phase 4:
  Explainability and Error Analysis". Two Docker sections and two inference-function definitions repeat each other.
- Author block is a placeholder ("Anonymous Authors / University Name").
- **Related-work citations** (checked by web search, Oct 2026):
  - **Wrong:** "Sharma et al. 2016 introduced ... 3,879 tweets for POS tagging". Sharma et al.
    (NAACL 2016, N16-1159) is a shallow-parsing pipeline (language ID, normalisation, POS,
    chunking). The **3,879** figure belongs to **Joshi et al. (COLING 2016, C16-1234)**: 3,879
    Facebook comments from public pages (Salman Khan, Narendra Modi), labelled ~15% negative /
    50% neutral / 35% positive.
  - **Wrong:** "Joshi et al. expanded to 6,096 samples across Twitter and Facebook ... 69.7% with
    SVM". Their dataset is the 3,879 Facebook comments above. Re-read the paper's results table
    before quoting any accuracy.
  - **Double-counted:** "Prabhu et al. 2016 ... 58.5% F1" is the *same* COLING paper. The arXiv
    version (1611.00472) lists Prabhu first. Cite it once.
  - **Wrong:** "MuRIL achieving 89.6% accuracy on IndicXNLI". IndicXNLI (EMNLP 2022) was published
    after MuRIL; the MuRIL paper (arXiv 2103.10730) reports **74.1** zero-shot accuracy on XNLI
    (mBERT 66.8). Remove or replace.
  - **Fix:** MuRIL has **236M** parameters (paper says 237M). The vocabulary of 197,285 is correct.
  - **Not found:** "Bhat et al. 2022 ... 83.4% on 8K samples". No such paper turned up. Remove it
    unless you have the actual reference.
  - **Unverified, remove unless you have the source:** "SHAP achieving 98.7% faithfulness" (not a
    standard SHAP result), "TextBlob 0.82 correlation to human annotations", "Asghar et al. Yelp
    95.4% with CNNs", "Obadimu 91.2% AUC", "Sharma et al. 2014 Hindi WordNet 78.3%".
  - "BERT ... GLUE (80.5% accuracy)": 80.5 is the GLUE *score*, not accuracy. Same for RoBERTa 88.5.
  - "achieving 52.7% correlation with TextBlob" should be Pearson r = 0.527, not a percentage.
  - Hutto & Gilbert 2014 (VADER) is commonly cited for F1 = 0.96 on tweets; that's fine to keep.

## 7. Suggested honest framing

What the work actually shows, and still a reasonable paper:

1. A 131,608-comment YouTube corpus on Indian employment discourse, mostly Latin-script, with
   3.4% Devanagari, 2.2% mixed-script, at least ~10% romanized Hindi (measured via "hai"), and
   some other Indian languages in Latin script; heavily concentrated in 2024–25.
2. Silver labels from VADER, with a **quantified analysis of VADER's failure on Hindi/Hinglish**
   (97.4% of Devanagari comments → Neutral; lexicon collisions like *kia*, *baad*, *acche*; and
   on human-labelled Hinglish, 50.6% accuracy / κ = 0.12, no better than majority class). That's
   a genuine, useful finding. Make it a contribution rather than hiding it.
3. A fine-tuned DistilBERT reproduces VADER's labels with 96.47% agreement (vs 88.57% for TF-IDF +
   Linear SVM, which also lost negation words to stop-word removal), and inherits VADER's biases,
   shown by the learned features. The controlled experiment in 1b (85% vs VADER, 51% vs humans,
   67% when trained on human labels) quantifies the cost.
4. Limitations: no human gold labels (or, once annotated, the real VADER-vs-human and
   DistilBERT-vs-human agreement), single seed, no per-language evaluation (until the verification
   run), possible duplicate leakage (until checked).

Suggested title: *"Silver-Label Sentiment Analysis of Indian Employment Discourse on YouTube: How
Lexicon Labels Fail on Hindi and Hinglish"*. Avoid "HinglishSent" and "novel code-mixing detection
algorithm"; script-ratio thresholds aren't novel.

## 8. Numbers the verification run will give you

`phase3_transformer_models/Phase3_DistilBERT_Verification_Colab.ipynb` (Colab T4, ~30 min/seed)
re-trains DistilBERT with the original settings and writes `results/verification/VERIFICATION_REPORT.md` with:

- accuracy / macro-F1 as mean ± std over seeds, plus bootstrap 95% CIs
- accuracy per language, including romanized Hinglish split out of "English", next to each
  group's majority-class baseline (Hindi's baseline is ~97% just by predicting Neutral)
- accuracy by VADER |compound| band, showing how much the model is learning VADER's threshold
- % of test comments whose exact text also appears in train, with accuracy on seen vs unseen text
- the real per-epoch training log (replaces the synthetic training-curve figure)
- `human_annotation_sample.csv`: 300 comments stratified by language. Label them blind, then run
  `--evaluate-human` to get VADER-vs-human and DistilBERT-vs-human accuracy and Cohen's κ. That's
  the single most valuable number the paper is missing.
- optional MuRIL run on the same split.
