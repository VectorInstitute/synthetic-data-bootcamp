## GPU VM Setup

GPU VMs boot directly from a PyTorch Deep Learning VM image — NVIDIA drivers,
CUDA 12.8, and PyTorch 2.7 are pre-installed, so the GPU is ready immediately.
CPU VMs use the COS container image via the gce-container module.

`main.tf` looks for a boot image `"projects/pets-3-bootcamp/global/images/family/synthetic-data-generation-bootcamp-gpu"`.

### 1. Build the compute image:

```sh
cd /path/to/synthetic-data-bootcamp
gcloud builds submit . \
  --region=us-central1 \
  --config=deploy/gpu-image/cloudbuild.yaml
```

### 2. Check if it's created successfully:

```sh
gcloud compute images list --project=pets-3-bootcamp --filter="name~synthetic"
gcloud compute images list --project=pets-3-bootcamp --no-standard-images
```