#!/usr/bin/env bash
# Deploy the prediction service to Google Cloud Run, building the image from the Dockerfile.
#
# Prerequisites: the gcloud CLI installed and logged in (gcloud auth login), and a Google Cloud
# project with billing enabled. Run train.py first so model.bin exists.
#
# Usage:
#   export PROJECT_ID=your-project-id
#   ./deploy_cloud_run.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID first, e.g. export PROJECT_ID=my-project}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-readmission-api}"

[ -f model.bin ] || { echo "model.bin not found. Run: python train.py"; exit 1; }

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Builds the Dockerfile in this folder with Cloud Build and deploys it.
# 1Gi memory leaves room for pandas + scikit-learn + XGBoost across 2 workers.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --memory 1Gi \
  --allow-unauthenticated

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "Service URL: $URL"
echo "Health check:"
curl -s "$URL/health"
echo
echo
echo "Try a prediction:"
echo "  curl -X POST -H 'Content-Type: application/json' -d @sample_patient.json $URL/predict"
echo
echo "To stop paying for / exposing the service when you are done:"
echo "  gcloud run services delete $SERVICE --region $REGION"
