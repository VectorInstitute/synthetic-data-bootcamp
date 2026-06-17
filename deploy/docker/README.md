# Docker Build & Push Guide

Build and push workspace images for the Synthetic Data Generation Bootcamp to Google Cloud Artifact Registry.

**GCP project:** `pets-3-bootcamp`

For the full deployment flow (Coder setup, service accounts, template publish), start at [`deploy/README.md`](../README.md).

## Registry location

| | |
|---|---|
| **Project** | `pets-3-bootcamp` |
| **Location** | `us-central1` |
| **Repository** | `coder` |
| **Image** | `synthetic-data-generation-bootcamp` |
| **Full path** | `us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp` |

### Tags

| Tag | Base image | Used for |
| --- | --- | --- |
| `latest` | `ubuntu:24.04` | CPU workspaces (`e2-standard-2`) |
| `gpu` | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | GPU workspaces (`g2-standard-8`, `g2-standard-24`) |

## Who needs access to this registry?

| Identity | Why | Required role |
| --- | --- | --- |
| You (local `gcloud` / Docker) | Build and push images | Artifact Registry Writer (or project Editor) |
| `coder-admin@coderd.iam.gserviceaccount.com` | Provisions VMs on shared Coder (Path A) | Indirect — needs Compute Admin on the project (see [`deploy/README.md`](../README.md)) |
| Workspace instance SA (`pet-3-coder-sa@pets-3-bootcamp.iam.gserviceaccount.com`) | Pulls the container image when a CPU workspace starts | `roles/artifactregistry.reader` on `pets-3-bootcamp` |

Grant the workspace instance SA reader access:

```sh
gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:pet-3-coder-sa@pets-3-bootcamp.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Access to the `pets-3-bootcamp` GCP project

## Setup

### 1. Authenticate with gcloud

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project pets-3-bootcamp
```

### 2. Configure Docker for Artifact Registry

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 3. Enable APIs and create the repository (if needed)

```sh
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create coder \
  --repository-format=docker \
  --location=us-central1 \
  --description="Bootcamp workspace images"
```

Skip the `artifacts repositories create` command if the `coder` repository already exists.

## Build and push

### CPU image (`latest`)

#### Option A: Cloud Build (recommended)

```sh
cd /path/to/synthetic-data-bootcamp
gcloud builds submit . --region=us-central1 --config=deploy/docker/cloudbuild-cpu.yaml
```

#### Option B: Build locally and push

CPU workspaces run on `linux/amd64`. Set the platform explicitly when building on Apple Silicon:

```sh
cd /path/to/synthetic-data-bootcamp
docker build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:latest \
  -f deploy/docker/Dockerfile .

docker push us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:latest
```

### GPU image (`gpu`)

Uses `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` as the base via the `BASE_IMAGE` build arg.

#### Option A: Cloud Build (recommended)

```sh
cd /path/to/synthetic-data-bootcamp
gcloud builds submit . --region=us-central1 --config=deploy/docker/cloudbuild-gpu.yaml
```

#### Option B: Build locally and push

```sh
cd /path/to/synthetic-data-bootcamp

docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:gpu \
  -f deploy/docker/Dockerfile .

docker push us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:gpu
```

## Managing images

### List existing images

```sh
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp \
  --include-tags
```

### Delete an image (if needed)

```sh
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:latest \
  --quiet
```

### Pull the image

```sh
docker pull us-central1-docker.pkg.dev/pets-3-bootcamp/coder/synthetic-data-generation-bootcamp:latest
```

## References

- [Build and push a Docker image with Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image)
- [Artifact Registry Docker guide](https://cloud.google.com/artifact-registry/docs/docker)
- [gcloud CLI documentation](https://cloud.google.com/sdk/gcloud)
