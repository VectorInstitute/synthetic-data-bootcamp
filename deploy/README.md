# Synthetic Data Generation Bootcamp Deployment

The Synthetic Data Generation Bootcamp uses [Coder](https://coder.com) to provide cloud development environments for participants. Each workspace is a GCP virtual machine provisioned by Terraform in `[deploy/coder-template/](coder-template/)`.

This repository supports **two deployment paths**. Choose the one that matches how you run Coder:


|                        | **Path A — Shared Coder (Vector)**                  | **Path B — Self-hosted Coder**                                  |
| ---------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| **Coder server**       | `https://platform.vectorinstitute.ai` (org-managed) | You deploy `coderd` on a GCP VM via `[deploy/main.tf](main.tf)` |
| **Who runs Terraform** | `coder-admin@coderd.iam.gserviceaccount.com`        | Your provisioner service account (or ADC on the `coderd` VM)    |
| **Typical use**        | Vector Institute bootcamps                          | Standalone / custom Coder install                               |


> **GCP project for this bootcamp:** `pets-3-bootcamp`

## Service accounts (read this first)

Two different service accounts are involved. Do not confuse them.


| Service account                                                         | Project                   | Role                                                                                                                                                                                                      |
| ----------------------------------------------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coder-admin@coderd.iam.gserviceaccount.com`                            | `coderd` (Vector/Coder)   | **Provisioner** — runs Terraform and calls GCP APIs (`disks.insert`, `instances.insert`, …). Shows up in Cloud Audit Logs as `principalEmail`.                                                            |
| `pet-3-coder-sa@pets-3-bootcamp.iam.gserviceaccount.com` (example name) | `pets-3-bootcamp` (yours) | **Workspace instance SA** — attached to each participant VM. Used at runtime to pull container images and call GCP APIs from inside the workspace. Set via the template variable `service_account_email`. |


## Prerequisites

Install on your local machine:

- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- [Terraform](https://developer.hashicorp.com/terraform/install)
- [Coder CLI](https://coder.com/docs/install)
- [Docker](https://docs.docker.com/engine/install/) (for local image builds)

### gcloud setup

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project pets-3-bootcamp
```

---

## Path A — Shared Coder (Vector Institute)

Use this path if participants log in to `**https://platform.vectorinstitute.ai**`. You do **not** deploy your own `coderd` VM.

### 1. Grant the Coder provisioner access to your project

Vector's shared Coder already has a provisioner service account. Grant it permission to create resources in `pets-3-bootcamp`:

```sh
gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:coder-admin@coderd.iam.gserviceaccount.com" \
  --role="roles/compute.admin"

gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:coder-admin@coderd.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

`roles/compute.admin` lets Coder create VMs and disks. `roles/iam.serviceAccountUser` lets Coder attach your workspace instance SA to those VMs.

To verify which identity is provisioning, check Cloud Audit Logs:

```sh
gcloud logging read \
  'protoPayload.serviceName="compute.googleapis.com"
   AND protoPayload.methodName=~"disks.insert"
   AND severity=ERROR' \
  --project=pets-3-bootcamp \
  --limit=5 \
  --format='value(protoPayload.authenticationInfo.principalEmail)'
```

On this path you should see `coder-admin@coderd.iam.gserviceaccount.com`.

### 2. Create a workspace instance service account

Create a service account **in `pets-3-bootcamp`** for workspace VMs. Full steps: `[deploy/coder-template/README.md](coder-template/README.md#workspace-instance-service-account)`.

At minimum, grant it **Artifact Registry Reader** so CPU workspaces can pull the container image:

```sh
gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:pet-3-coder-sa@pets-3-bootcamp.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

Replace the email if you used a different service account name.

### 3. Build and push the workspace Docker image and GPU boot image

See `[deploy/docker/README.md](docker/README.md)`. and `[deploy/gpu-image/README.md] (gpu-image/README.md)`.

### 4. Publish the workspace template

GitHub external auth is already configured on the shared Coder instance.

1. Create a local `terraform.tfvars` in `deploy/coder-template/` (never commit it — see [Template variables](#template-variables)).
2. Push the template:

```sh
coder login https://platform.vectorinstitute.ai
cd deploy/coder-template
terraform init

coder templates push <template-name> -y
```

Coder auto-loads `terraform.tfvars` from the current directory. No `--variables-file` flag is needed.

`<template-name>` is the name you choose in Coder (for example `synthetic-data-generation-bootcamp`). Use the same name on later pushes to update that template. Check existing templates with `coder templates list`.

Full details on variable files: `[deploy/coder-template/README.md](coder-template/README.md#template-variables)`.

---

## Path B — Self-hosted Coder

Use this path if you deploy and manage your own `coderd` server.

### 1. Create a provisioner service account

Create a service account in `pets-3-bootcamp` that `coderd` will use to run Terraform. Grant it:

- Compute Admin (`roles/compute.admin`)
- Service Account User (`roles/iam.serviceAccountUser`)

Create a JSON key and configure credentials on the machine running `coderd` (for example via `GOOGLE_APPLICATION_CREDENTIALS`). Details: `[deploy/coder-template/README.md](coder-template/README.md#self-hosted-provisioner-credentials)`.

### 2. Deploy the Coder server

From the `deploy/` directory, create a local `terraform.tfvars` (listed in `.gitignore`) and apply. See variable names in `[deploy/variables.tf](variables.tf)`.

```sh
cd deploy/
terraform init -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

This creates the VM in `[main.tf](main.tf)` and runs `[startup.sh](startup.sh)`, which installs Docker and starts Coder. Open `https://<external-ip>` to complete setup.

To tear down:

```sh
terraform destroy -var-file=terraform.tfvars
```

### 3. Configure GitHub external auth

Required for self-hosted installs. See `[deploy/coder-template/README.md](coder-template/README.md#github-external-authentication)`.

### 4. Build the image and publish the template

Same as Path A steps 2–4, but log in to your Coder URL instead of `platform.vectorinstitute.ai`. Ensure `terraform.tfvars` is in `deploy/coder-template/` as described in `[deploy/coder-template/README.md](coder-template/README.md#template-variables)`.

---

## Template variables

Template variable values (including `firebase_api_key` and `token_service_url`) are **secrets/config** and must stay out of git and the README.

Keep a local `deploy/coder-template/terraform.tfvars` (HCL format). Coder CLI auto-loads it when you run `coder templates push` from that directory.

The file is in `.gitignore`. Create it from the committed example:

```sh
cd deploy/coder-template
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your real values
```

Then push (from the same directory):

```sh
coder templates push <template-name> -y
```

Coder stores the values server-side; they are not read from `terraform.tfvars` when participants create workspaces.

---

## Further reading


| Topic                                                                 | Document                                                      |
| --------------------------------------------------------------------- | ------------------------------------------------------------- |
| Workspace instance SA, provisioner auth, GitHub setup, variable files | `[deploy/coder-template/README.md](coder-template/README.md)` |
| Build and push Docker images                                          | `[deploy/docker/README.md](docker/README.md)`                 |
| GPU boot image (Packer)                                               | `[deploy/gpu-image/](gpu-image/)`                             |


