# Predicting 30-Day Hospital Readmission for Diabetic Patients

A machine learning project for the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course. It trains a classifier on hospital records and serves it as a web service in a Docker container.

## Problem description

### The problem

When a patient is discharged and returns to hospital within 30 days, that is called an **unplanned readmission**. Readmissions are bad for patients (they often signal that something went wrong with treatment or follow-up care) and expensive for hospitals. In the US, hospitals are financially penalized for excess readmissions, and in any health system, an unnecessary readmission uses a bed that another patient may need.

Diabetes makes this worse: diabetic patients are admitted more often and have more complications, and managing medication and blood sugar after discharge is a common point of failure.

### How a model helps

Hospital staff cannot give every patient intensive follow-up. A model that estimates each patient's risk of coming back within 30 days lets staff **focus limited follow-up resources on the patients who need them most**, for example:

- a nurse phone call two days after discharge,
- a pharmacist medication review,
- an earlier follow-up appointment, or
- a home visit.

**How the solution is used:** at discharge, a nurse or case manager enters the patient's record (age, length of stay, number of previous admissions, diagnoses, medications and so on). The service returns the estimated probability of readmission within 30 days and a `high_risk` flag marking patients in the top 20% of predicted risk. Those patients are the ones prioritised for follow-up.

### The machine learning task

Binary classification: **will this patient be readmitted within 30 days (1) or not (0)?**

The positive class is a small minority of patients, so accuracy is a misleading metric. I evaluate with **PR-AUC** (average precision) and **ROC-AUC**, and report **lift** (how many more readmissions are caught among the flagged patients than by picking patients at random).

> This is a learning project. A model like this is a decision aid, not a replacement for clinical judgment, and it must not be used on real patients without proper validation (see [Limitations](#limitations)).

## Dataset

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository.

- About 100,000 hospital encounters for diabetic patients at 130 US hospitals, with about 50 features: demographics, admission and discharge details, diagnoses, lab tests, medications and number of previous visits.
- Original target `readmitted` is `NO`, `>30` or `<30` days. This project predicts `<30` vs. everything else.
- Source study: Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, BioMed Research International, 2014.

**Getting the data.** The dataset is not committed to this repository. It downloads automatically the first time you run `train.py` or the notebook and is saved to `data/diabetic_data.csv`. If the automatic download fails (for example behind a firewall), download the zip from the UCI link above, unzip it, and put `diabetic_data.csv` in the `data/` folder.

## What was done

1. **Cleaning** (`features.py`, explained in the notebook): removed patients who died or went to hospice (they cannot be readmitted), kept one encounter per patient to avoid the same person appearing in both training and test data, dropped `weight` (almost entirely missing), and kept `payer_code` and `medical_specialty` with an explicit "Missing" category. ICD-9 diagnosis codes are grouped into broad categories (circulatory, respiratory, diabetes, and so on).
2. **EDA** (`notebook.ipynb`): missing values, target distribution, numeric and categorical features against the target, mutual information and permutation importance.
3. **Models** (`notebook.ipynb`): a dummy baseline, logistic regression (linear), random forest and XGBoost (tree-based), each tuned with randomized search and compared by cross-validated PR-AUC. The best model is evaluated once on a held-out 20% test set.
4. **Export**: `train.py` retrains the chosen model and saves it to `model.bin`. `predict.py` serves it with Flask.

### Results

| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Dummy (always predicts the base rate) | 0.090 | 0.500 | 0.090 |
| Logistic regression | 0.165 | 0.651 | 0.172 |
| Random forest* | 0.162 | 0.657 | 0.177 |
| XGBoost (selected) | 0.184 | 0.656 | 0.191 |

\*The random forest was tuned on a 25,000-row subsample because tuning on all rows took about 50 minutes, so its CV score is not directly comparable with the other two (5-fold CV on all training rows). An untuned random forest scored 0.173 in 5-fold CV on all training rows, and an earlier full-data tuning search scored about 0.176, so it is probably close to but slightly behind XGBoost.

The final choice was a head-to-head between logistic regression and XGBoost on all training rows. XGBoost was selected because its mean CV PR-AUC (0.184) beat logistic regression's (0.165) by more than the fold-to-fold standard deviation (0.009), a rule fixed before looking at the test set.

- Positive rate (readmitted within 30 days) after cleaning: 9.0% (6,285 of 69,987 patients), so a model with no skill scores a PR-AUC of about 0.09.
- Selected model: XGBoost. On the held-out test set (13,998 patients, used once) it reaches ROC-AUC 0.656 and PR-AUC 0.191, about twice the no-skill PR-AUC. The test PR-AUC is close to the CV score (0.184), so there is no sign of overfitting.
- Flagging the top 20% of patients by predicted risk (2,799 of 13,998 in the test set) catches 39% of the readmissions, with 17.5% precision against a 9.0% base rate: a lift of 1.95x.
- Flagging only the top 10% (1,399 patients) catches 27% of readmissions at 24.3% precision, a lift of 2.7x. The top 5% reaches a lift of 3.2x.
- Most important features (permutation importance from an EDA probe model): discharge disposition, number of prior inpatient visits, payer code, primary diagnosis group, age and number of diagnoses.

## Repository structure

```
.
├── README.md
├── notebook.ipynb          # data preparation, EDA, feature importance, model selection and tuning
├── features.py             # shared cleaning and feature code (used by train.py and predict.py)
├── train.py                # trains the final model and saves model.bin
├── predict.py              # Flask web service
├── model.bin               # trained model (created by train.py)
├── best_params.json        # chosen model and tuned parameters (created by the notebook)
├── sample_patient.json     # example request for the service
├── requirements.txt        # full environment (notebook + training + serving)
├── requirements-serve.txt  # minimal dependencies for the Docker image
├── Dockerfile
├── deploy_cloud_run.sh     # deploys the container to Google Cloud Run
└── data/                   # dataset goes here (downloaded automatically)
```

## How to run the project

Requires **Python 3.12 or newer**.

### 1. Create a virtual environment and install dependencies

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pandas`, `numpy`, `scikit-learn` and `xgboost` are pinned to exact versions on purpose: `model.bin` is a pickle file and must be loaded with the same library versions that trained it (the pinned versions are those of the Google Colab environment where the final model was trained), including inside Docker. If you retrain with `train.py` in a different environment, update the pins in `requirements.txt` and `requirements-serve.txt` to match.

### 2. Explore the data and select a model (notebook)

```bash
jupyter notebook notebook.ipynb
```

Run all cells. The notebook downloads the data if needed, writes `best_params.json` at the end, and has a `FAST = False` switch near the start of the training section. `FAST = True` is a quick smoke test; `FAST = False` is the real run and can take a while on the full dataset.

If GitHub does not render the notebook, open it with [nbviewer](https://nbviewer.org/).

### 3. Train the final model

```bash
python train.py
```

This reads `best_params.json`, trains the selected model, prints test ROC-AUC and PR-AUC, and saves `model.bin`. Options:

```bash
python train.py --model "random forest"     # override the model from best_params.json
python train.py --data path/to/diabetic_data.csv --output model.bin
```

### 4. Run the web service locally

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

(`python predict.py` also works for a quick development server on the same port.)

In a second terminal, send the example patient:

```bash
curl -X POST -H "Content-Type: application/json" -d @sample_patient.json http://localhost:9696/predict
```

Example response (numbers will differ with your model):

```json
{
  "readmission_30d_probability": 0.2147,
  "high_risk": false,
  "high_risk_threshold": 0.3311,
  "model": "xgboost"
}
```

- `GET /health` returns `{"status": "ok", ...}`.
- `POST /predict` takes one patient as a JSON object. The required fields are listed in `sample_patient.json`: `gender`, `age`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `diag_1`, `change` and `diabetesMed`. Race, payer, specialty, glucose and A1C results, `diag_2`, `diag_3` and individual drug columns are optional and default to "unknown" / "not given". Missing or invalid fields return HTTP 400 with a readable message.
- `high_risk` is `true` when the probability is in the top 20% of scores seen on the test set. That cutoff is a capacity decision (how many patients can staff follow up?), not a medical fact, and can be changed with `FLAG_FRACTION` in `train.py`.

## Docker

The image is built from `requirements-serve.txt` (without Jupyter and plotting libraries), so it stays small. `model.bin` must exist before building, so run `python train.py` first.

Build:

```bash
docker build -t readmission-service .
```

Run:

```bash
docker run -it --rm -p 9696:9696 readmission-service
```

Test it from another terminal:

```bash
curl http://localhost:9696/health
curl -X POST -H "Content-Type: application/json" -d @sample_patient.json http://localhost:9696/predict
```

The container listens on the port in the `PORT` environment variable (default 9696), which is how cloud platforms tell it where to listen.

## Cloud deployment

### Google Cloud Run (script provided)

Prerequisites: the [gcloud CLI](https://cloud.google.com/sdk/docs/install), a Google Cloud project with billing enabled, and a trained `model.bin`.

```bash
gcloud auth login
export PROJECT_ID=your-project-id
./deploy_cloud_run.sh
```

The script enables the Cloud Run, Cloud Build and Artifact Registry APIs, builds the Dockerfile in the cloud with `gcloud run deploy --source .`, deploys the service publicly with 1 GiB of memory, and prints the service URL. Region defaults to `europe-west1`; override it with `REGION=...`.

Delete the service when you are finished so it neither costs money nor stays publicly reachable:

```bash
gcloud run services delete readmission-api --region europe-west1
```

### Alternative: Render (no command line needed)

Push this repository to GitHub, create a new **Web Service** on [Render](https://render.com/), connect the repository and set **Language** to **Docker**. Render builds the Dockerfile on each deploy. Check Render's current documentation for free-instance limits before choosing a plan.

### Deployed service

- URL: https://organic-broccoli-jj5xq7vrqq4r2vxj-9697.app.github.dev/

## Limitations

- The data covers 1999-2008 at US hospitals. Practices, coding and patient populations have changed, and the model has never been validated at any other hospital, so it would need retraining and validation on local data before any real use.
- Thirty-day readmission is hard to predict from administrative data. Expect modest discrimination, and judge the model by lift over random selection rather than by absolute AUC.
- The model finds statistical associations, not causes, and it can reflect biases in how care was delivered and recorded (for example, differences by race or insurance type).
- The service takes no patient identifiers, but real use would require privacy, security and regulatory review.
