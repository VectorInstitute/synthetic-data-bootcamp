terraform {
  required_providers {
    coder = {
      source = "coder/coder"
    }
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "coder" {}

provider "google" {
  zone    = data.coder_parameter.zone.value
  project = var.project
}

data "coder_provisioner" "me" {}
data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}
data "coder_external_auth" "github" {
  id = var.github_app_id
}

data "coder_parameter" "zone" {
  name         = "GCP Zone"
  description  = "us-central1 zone for the A100 VM. Pick another zone if the selected one is out of capacity."
  type         = "string"
  default      = "us-central1-a"
  mutable      = false
  display_name = "GCP Zone"

  option {
    name  = "us-central1-a"
    value = "us-central1-a"
  }
  option {
    name  = "us-central1-b"
    value = "us-central1-b"
  }
  option {
    name  = "us-central1-c"
    value = "us-central1-c"
  }
  option {
    name  = "us-central1-f"
    value = "us-central1-f"
  }
}

locals {
  username     = "coder"
  repo_name    = replace(regex(".*/(.*)", var.github_repo)[0], ".git", "")
  machine_type = "a2-highgpu-1g"
  zone         = data.coder_parameter.zone.value

  # Same GPU boot image as the L4 template — NVIDIA drivers and CUDA are pre-installed.
  gpu_boot_image = "projects/pets-3-bootcamp/global/images/family/synthetic-data-generation-bootcamp-gpu"

  gpu_startup_script = <<-SCRIPT
    #!/bin/bash
    set -e

    if ! id ${local.username} &>/dev/null; then
      useradd --groups sudo --no-create-home --shell /bin/bash ${local.username}
      echo "${local.username} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${local.username}
      chmod 0440 /etc/sudoers.d/${local.username}
    fi

    DATA_DISK="/dev/disk/by-id/google-data-disk-0"
    mkdir -p /home/${local.username}
    if ! blkid "$DATA_DISK" >/dev/null 2>&1; then
      mkfs.ext4 -F "$DATA_DISK"
    fi
    mount "$DATA_DISK" /home/${local.username} || true
    chown ${local.username}:${local.username} /home/${local.username}

    if [ -d "/opt/home-seed" ] && [ ! -f "/home/${local.username}/.home_seeded" ]; then
      cp -a /opt/home-seed/. "/home/${local.username}/"
      chown -R ${local.username}:${local.username} /home/${local.username}
      touch "/home/${local.username}/.home_seeded"
    fi

    printf '%s' '${base64encode(coder_agent.main.init_script)}' | base64 -d > /tmp/coder-init.sh
    chmod +x /tmp/coder-init.sh
    chown ${local.username}:${local.username} /tmp/coder-init.sh
    sudo -u ${local.username} -H /tmp/coder-init.sh
  SCRIPT
}

resource "coder_agent" "main" {
  auth = "google-instance-identity"
  arch = "amd64"
  os   = "linux"

  display_apps {
    vscode = false
  }

  startup_script = <<-EOT
    #!/bin/bash
    set -e

    echo "Fixing permissions for /home/${local.username}"
    sudo chown -R ${local.username}:${local.username} /home/${local.username}

    if [ -d "/opt/home-seed" ] && [ ! -f "/home/${local.username}/.home_seeded" ]; then
      echo "Seeding home directory from image..."
      cp -a /opt/home-seed/. "/home/${local.username}/"
      touch "/home/${local.username}/.home_seeded"
      echo "Home directory seeded"
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/home/${local.username}/.local/bin:$PATH"

    cd "/home/${local.username}"

    if [ -d "${local.repo_name}" ]; then
      if [ -d "${local.repo_name}/.git" ]; then
        echo "Repository already exists, updating..."
        cd ${local.repo_name}
        git pull || echo "Warning: git pull failed, continuing with existing code"
      else
        echo "Directory exists but is not a git repository, cleaning up..."
        rm -rf ${local.repo_name}
        echo "Cloning repository..."
        if git clone ${var.github_repo} ${local.repo_name}; then
          echo "Repository cloned successfully"
          cd ${local.repo_name}
          git checkout ${var.github_branch}
        else
          echo "ERROR: Failed to clone repository"
          exit 1
        fi
      fi
    else
      echo "Cloning repository..."
      if git clone ${var.github_repo} ${local.repo_name}; then
        echo "Repository cloned successfully"
        cd ${local.repo_name}
        git checkout ${var.github_branch}
      else
        echo "ERROR: Failed to clone repository"
        exit 1
      fi
    fi

    if [ ! -d ".git" ]; then
      echo "ERROR: Not in a valid git repository"
      exit 1
    fi

    echo "Current directory: $(pwd)"
    echo "Directory contents: $(ls -la)"

    echo "Setting up virtual environment and installing dependencies..."

    if [ ! -d ".venv" ]; then
      echo "Creating new virtual environment..."
      uv venv .venv
    else
      echo "Virtual environment already exists, skipping creation"
    fi

    source .venv/bin/activate

    if [ -f "pyproject.toml" ]; then
      echo "Found pyproject.toml, installing dependencies..."
      uv sync --dev
      sync_exit_code=$?

      if [ $sync_exit_code -eq 0 ]; then
        echo "Dependencies installed successfully"
      else
        echo "Warning: uv sync exited with code $sync_exit_code"
      fi
    else
      echo "No pyproject.toml found in $(pwd), skipping dependency installation"
    fi

    sleep 2

    echo "Running automatic onboarding..."
    if command -v onboard &> /dev/null; then
        if ! onboard_output="$(onboard \
            --bootcamp-name "$BOOTCAMP_NAME" \
            --test-script "/home/${local.username}/${local.repo_name}/aieng-synthetic-data/tests/test_integration.py" \
            --test-marker "integration_test")"; then
          echo "Onboarding failed, continuing..."
        else
          echo "Onboarding successful"
          eval "$onboard_output"
        fi
    else
      echo "Onboarding CLI not found, skipping automated onboarding"
    fi

    mkdir -p "/home/${local.username}/${local.repo_name}/.vscode"
    cat > "/home/${local.username}/${local.repo_name}/.vscode/settings.json" <<'VSCODE_SETTINGS'
{
  "python.terminal.useEnvFile": true
}
VSCODE_SETTINGS

    if ! grep -q "Auto-navigate to ${local.repo_name}" "/home/${local.username}/.bashrc" 2>/dev/null; then
      cat >> "/home/${local.username}/.bashrc" <<BASHRC

# Auto-navigate to ${local.repo_name} and activate venv
if [ -f ~/${local.repo_name}/.venv/bin/activate ]; then
    cd ~/${local.repo_name}
    source .venv/bin/activate
fi
# Load bootcamp environment variables from Secret Manager
if command -v onboard &> /dev/null && [ -n "\$BOOTCAMP_NAME" ]; then
    eval "\$(onboard --bootcamp-name "\$BOOTCAMP_NAME" --skip-test 2>/dev/null)"
fi
BASHRC
    fi

    if ! grep -q "Auto-navigate to ${local.repo_name}" "/home/${local.username}/.zshrc" 2>/dev/null; then
      cat >> "/home/${local.username}/.zshrc" <<ZSHRC

# Auto-navigate to ${local.repo_name} and activate venv
if [ -f ~/${local.repo_name}/.venv/bin/activate ]; then
    cd ~/${local.repo_name}
    source .venv/bin/activate
fi
# Load bootcamp environment variables from Secret Manager
if command -v onboard &> /dev/null && [ -n "\$BOOTCAMP_NAME" ]; then
    eval "\$(onboard --bootcamp-name "\$BOOTCAMP_NAME" --skip-test 2>/dev/null)"
fi
ZSHRC
    fi
    # Add to .profile so VS Code server (login shell) and all child processes
    # — including Jupyter kernels and the Python extension — inherit the
    # bootcamp environment variables automatically.
    if ! grep -q "bootcamp-env" "/home/${local.username}/.profile" 2>/dev/null; then
      cat >> "/home/${local.username}/.profile" <<PROFILE

# bootcamp-env: load API keys from Secret Manager at login
# The command is stored here, not the secret values.
if command -v onboard > /dev/null 2>&1 && [ -n "\$BOOTCAMP_NAME" ]; then
    eval "\$(onboard --bootcamp-name "\$BOOTCAMP_NAME" --skip-test 2>/dev/null)"
fi
PROFILE
    fi

    echo "Startup script ran successfully!"

  EOT

  env = {
    GIT_AUTHOR_NAME      = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
    GIT_AUTHOR_EMAIL     = "${data.coder_workspace_owner.me.email}"
    GIT_COMMITTER_NAME   = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
    GIT_COMMITTER_EMAIL  = "${data.coder_workspace_owner.me.email}"
    GITHUB_USER          = data.coder_workspace_owner.me.name
    TOKEN_SERVICE_URL    = var.token_service_url
    BOOTCAMP_NAME        = var.bootcamp_name
    FIREBASE_WEB_API_KEY = var.firebase_api_key
  }

  metadata {
    display_name = "CPU Usage"
    key          = "0_cpu_usage"
    script       = "coder stat cpu"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "RAM Usage"
    key          = "1_ram_usage"
    script       = "coder stat mem"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "GPU Usage"
    key          = "2_gpu_usage"
    script       = "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{printf \"%s%%\", $1}'; else echo 'N/A'; fi"
    interval     = 10
    timeout      = 3
  }
}

module "github-upload-public-key" {
  count            = data.coder_workspace.me.start_count
  source           = "registry.coder.com/coder/github-upload-public-key/coder"
  version          = "1.0.15"
  agent_id         = coder_agent.main.id
  external_auth_id = data.coder_external_auth.github.id
}

resource "google_compute_disk" "pd" {
  project = var.project
  name    = "coder-${data.coder_workspace.me.id}-data-disk"
  type    = "pd-ssd"
  zone    = local.zone
  size    = 100
}

resource "google_compute_instance" "dev" {
  zone         = local.zone
  count        = data.coder_workspace.me.start_count
  name         = "coder-${lower(data.coder_workspace_owner.me.name)}-${lower(data.coder_workspace.me.name)}"
  machine_type = local.machine_type

  # A2 machine types cannot live-migrate; they must be terminated on host maintenance.
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }
  boot_disk {
    initialize_params {
      image = local.gpu_boot_image
      size  = 100
    }
  }
  attached_disk {
    source      = google_compute_disk.pd.self_link
    device_name = "data-disk-0"
    mode        = "READ_WRITE"
  }
  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }
  metadata = {
    "startup-script" = local.gpu_startup_script
  }
}

resource "coder_agent_instance" "dev" {
  count       = data.coder_workspace.me.start_count
  agent_id    = coder_agent.main.id
  instance_id = google_compute_instance.dev[0].instance_id
}

resource "coder_metadata" "workspace_info" {
  count       = data.coder_workspace.me.start_count
  resource_id = google_compute_instance.dev[0].id

  item {
    key   = "image"
    value = local.gpu_boot_image
  }

  item {
    key   = "instance_type"
    value = local.machine_type
  }

  item {
    key   = "zone"
    value = local.zone
  }
}

module "vscode-web" {
  count          = tobool(var.codeserver) ? data.coder_workspace.me.start_count : 0
  source         = "registry.coder.com/coder/vscode-web/coder"
  version        = "1.3.0"
  agent_id       = coder_agent.main.id
  extensions     = ["ms-python.python", "ms-python.vscode-pylance", "ms-vsliveshare.vsliveshare"]
  install_prefix = "/tmp/.vscode-web"
  folder         = "/home/coder/${local.repo_name}"
  accept_license = true
  subdomain      = false
  order          = 1
}

resource "coder_app" "streamlit" {
  count        = 1
  agent_id     = coder_agent.main.id
  slug         = "streamlit"
  display_name = "Streamlit"
  url          = "http://localhost:8501"
  icon         = "https://icon.icepanel.io/Technology/svg/Streamlit.svg"
  subdomain    = false
  share        = "owner"
  order        = 2

  healthcheck {
    url       = "http://localhost:8501/_stcore/health"
    interval  = 5
    threshold = 6
  }
}
