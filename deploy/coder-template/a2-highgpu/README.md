# A100 GPU Coder template (`a2-highgpu-1g`)

GPU-only workspace template: `a2-highgpu-1g` (12 vCPU, 85 GB RAM, 1× NVIDIA A100 40GB) in **us-central1**.

Workspaces boot from the same Packer GPU image family as the L4 template (`synthetic-data-generation-bootcamp-gpu`). Participants pick a us-central1 zone at create time.

This directory is a **separate Coder template**. Push it independently from the parent CPU/L4 template.

## Publish

```sh
cd deploy/coder-template/a2-highgpu
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your real values

coder login https://platform.vectorinstitute.ai
terraform init
coder templates push synthetic-data-bootcamp-a100 -y
```

Use a different `<template-name>` from the parent template (CPU + L4 in `northamerica-northeast2`).

Service accounts, GitHub auth, and GPU image build steps are the same as [`../README.md`](../README.md).
