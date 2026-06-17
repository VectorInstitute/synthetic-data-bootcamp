# Synthetic Data Generation Bootcamp — Coder Template

This Terraform template provisions participant workspaces on GCP: a VM, persistent data disk, and the Coder agent with JupyterLab, code-server, and Streamlit.

**GCP project:** `pets-3-bootcamp`

Before publishing this template, read the deployment overview in [`deploy/README.md`](../README.md) to choose between **shared Coder (Vector)** and **self-hosted Coder**.

## Service accounts

Two service accounts play different roles. Creating one does not replace the other.

| | Provisioner SA | Workspace instance SA |
|---|---|---|
| **Purpose** | Runs Terraform; creates disks and VMs | Attached to each workspace VM at runtime |
| **Path A (shared Coder)** | `coder-admin@coderd.iam.gserviceaccount.com` (already exists) | You create in `pets-3-bootcamp` |
| **Path B (self-hosted)** | You create in `pets-3-bootcamp` | You create in `pets-3-bootcamp` (can be the same or different) |
| **Template variable** | Not set in the template — determined by who runs `coderd` | `service_account_email` |
| **Audit log identity** | Yes (`principalEmail` on `disks.insert`, etc.) | No |

---

## Path A — Shared Coder (Vector Institute)

Participants use `https://platform.vectorinstitute.ai`. Terraform runs on Vector's Coder infrastructure using `coder-admin@coderd.iam.gserviceaccount.com`.

### Grant provisioner access

Run once per GCP project. Full commands are in [`deploy/README.md`](../README.md#1-grant-the-coder-provisioner-access-to-your-project):

```sh
gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:coder-admin@coderd.iam.gserviceaccount.com" \
  --role="roles/compute.admin"

gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:coder-admin@coderd.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

You do **not** need to create a key for `coder-admin@coderd` — Vector manages that identity.

### Workspace instance service account

Create a service account in **`pets-3-bootcamp`** for workspace VMs:

1. Open the [GCP Service Accounts console](https://console.cloud.google.com/iam-admin/serviceaccounts?project=pets-3-bootcamp).
2. Click **Create service account** (for example, name: `pet-3-coder-sa`).
3. You do **not** need Compute Admin on this account — provisioning is done by `coder-admin@coderd`.
4. Grant **Artifact Registry Reader** so CPU workspaces can pull the container image:

```sh
gcloud projects add-iam-policy-binding pets-3-bootcamp \
  --member="serviceAccount:pet-3-coder-sa@pets-3-bootcamp.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

5. Set the email as `service_account_email` in `terraform.tfvars` (see [Template variables](#template-variables)).

### GitHub external auth

Already configured on the shared Coder instance. Set `github_app_id` in `terraform.tfvars` (commonly `primary-github` on Vector's platform — confirm with your Coder admin).

---

## Path B — Self-hosted Coder

You run `coderd` on your own VM (see [`deploy/README.md`](../README.md#path-b--self-hosted-coder)).

### Self-hosted provisioner credentials

`coderd` must be authenticated to GCP so it can run this template's Terraform. Either:

- Run `gcloud auth application-default login` on the `coderd` host, or
- Create a provisioner service account in `pets-3-bootcamp`, grant it **Compute Admin** and **Service Account User**, create a JSON key, and set `GOOGLE_APPLICATION_CREDENTIALS` on the `coderd` host.

Provisioner setup (console):

1. [Create a service account](https://console.cloud.google.com/iam-admin/serviceaccounts/create?project=pets-3-bootcamp) in `pets-3-bootcamp`.
2. Grant roles:
   - Compute Admin (`roles/compute.admin`)
   - Service Account User (`roles/iam.serviceAccountUser`)
3. Create a JSON key under **Keys** → **Add key** → **Create new key**.
4. Place the key on the `coderd` VM and point `GOOGLE_APPLICATION_CREDENTIALS` at it.

For other credential options, see the [Terraform Google provider docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started#adding-credentials).

### Workspace instance service account

Same as Path A: create `pet-3-coder-sa` (or similar) in `pets-3-bootcamp`, grant **Artifact Registry Reader**, and set `service_account_email` in `terraform.tfvars`. The provisioner SA also needs `roles/iam.serviceAccountUser` so it can attach this SA to VMs.

### GitHub external authentication

1. Create a GitHub App and configure `coderd` following [Coder's external auth docs](https://coder.com/docs/admin/external-auth#github).
2. Use the value of `CODER_EXTERNAL_AUTH_0_ID` as `github_app_id` in `terraform.tfvars`.

---

## Workspace Docker image

Build and push the image to Artifact Registry in `pets-3-bootcamp` before publishing the template. See [`deploy/docker/README.md`](../docker/README.md).

Set `container_image` in `terraform.tfvars` to the image you pushed (see [`terraform.tfvars.example`](terraform.tfvars.example) for the expected path format).

---

## Template variables

Do **not** commit real variable values or paste secrets into the README. Required variable names are defined in [`variables.tf`](variables.tf).

### Setup (one-time)

```sh
cd deploy/coder-template
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your real values
```

`terraform.tfvars` is in `.gitignore`.

### Publish

`coder templates push` auto-loads `terraform.tfvars` from the current directory. Run from `deploy/coder-template/`:

```sh
coder login https://platform.vectorinstitute.ai   # Path A
# coder login https://<your-coder-url>          # Path B

cd deploy/coder-template
terraform init

coder templates push <template-name> -y
```

- `<template-name>` — name you choose in Coder (check `coder templates list` before updating an existing template).
- `-y` — skip confirmation prompts (optional).

To override a single value without editing the file:

```sh
coder templates push <template-name> --variable container_image=... -y
```

Committed example with placeholder values: [`terraform.tfvars.example`](terraform.tfvars.example).

Coder stores variable values when you publish. Participants do not read `terraform.tfvars` at workspace runtime.
