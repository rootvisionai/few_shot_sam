#!/bin/bash
# Configure Docker to use Artifact Registry
gcloud auth configure-docker europe-west4-docker.pkg.dev --quiet

# Pull the image
docker pull europe-west4-docker.pkg.dev/few-shot-sam/fsm-artifact-docker/few_shot_sam_backend:0.0.2

# Run the image
docker run --rm -d europe-west4-docker.pkg.dev/few-shot-sam/fsm-artifact-docker/few_shot_sam_backend:0.0.2
