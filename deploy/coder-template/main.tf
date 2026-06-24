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
  zone    = var.zone
  project = var.project
}

data "coder_provisioner" "me" {}
data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}
data "coder_external_auth" "github" {
  id = var.github_app_id
}

data "coder_parameter" "instance_type" {
  name        = "Instance Type"
  description = "Choose the compute resources for your workspace. GPU workspaces boot from a PyTorch Deep Learning VM with NVIDIA drivers and CUDA pre-installed."
  type        = "string"
  default     = "cpu"
  mutable     = false

  option {
    name  = "CPU — e2-standard-2 (2 vCPU, 8 GB RAM)"
    value = "cpu"
    icon  = "/emojis/1f4bb.png"
  }
  option {
    name  = "GPU — g2-standard-8 (8 vCPU, 32 GB RAM, 1× NVIDIA L4)"
    value = "g2-standard-8"
    icon  = "/emojis/26a1.png"
  }
  # TODO: Do we need this 96GB option?
  option {
    name  = "GPU — g2-standard-24 (24 vCPU, 96 GB RAM, 2× s NVIDIA L4)"
    value = "g2-standard-24"
    icon  = "/emojis/26a1.png"
  }
}

locals {
  username  = "coder"
  repo_name = replace(regex(".*/(.*)", var.github_repo)[0], ".git", "")

  is_gpu                = data.coder_parameter.instance_type.value != "cpu"
  selected_machine_type = local.is_gpu ? data.coder_parameter.instance_type.value : "e2-standard-2"

  # GPU VMs boot directly from a PyTorch Deep Learning VM image — NVIDIA drivers,
  # CUDA 12.8, and PyTorch 2.7 are pre-installed, so the GPU is ready immediately.
  # CPU VMs use the COS container image via the gce-container module.
  gpu_boot_image = "projects/pets-3-bootcamp/global/images/family/synthetic-data-generation-bootcamp-gpu"

  # Startup script for GPU VMs: creates the coder user, mounts the persistent data
  # disk, then runs the Coder agent init script directly on the VM (no Docker).
  gpu_startup_script = <<-SCRIPT
    #!/bin/bash
    set -e

    if ! id ${local.username} &>/dev/null; then
      useradd --groups sudo --no-create-home --shell /bin/bash ${local.username}
      echo "${local.username} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${local.username}
      chmod 0440 /etc/sudoers.d/${local.username}
    fi

    # Mount persistent data disk to /home/${local.username}
    DATA_DISK="/dev/disk/by-id/google-data-disk-0"
    mkdir -p /home/${local.username}
    if ! blkid "$DATA_DISK" >/dev/null 2>&1; then
      mkfs.ext4 -F "$DATA_DISK"
    fi
    mount "$DATA_DISK" /home/${local.username} || true
    chown ${local.username}:${local.username} /home/${local.username}

    # Seed home directory from image on first boot
    if [ -d "/opt/home-seed" ] && [ ! -f "/home/${local.username}/.home_seeded" ]; then
      cp -a /opt/home-seed/. "/home/${local.username}/"
      chown -R ${local.username}:${local.username} /home/${local.username}
      touch "/home/${local.username}/.home_seeded"
    fi

    # Write and execute the Coder agent init script as the coder user
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

    # Fix permissions immediately - must be first!
    echo "Fixing permissions for /home/${local.username}"
    sudo chown -R ${local.username}:${local.username} /home/${local.username}

    # Seed home directory from image on first boot (only available in the CPU Docker image)
    if [ -d "/opt/home-seed" ] && [ ! -f "/home/${local.username}/.home_seeded" ]; then
      echo "Seeding home directory from image..."
      cp -a /opt/home-seed/. "/home/${local.username}/"
      touch "/home/${local.username}/.home_seeded"
      echo "Home directory seeded"
    fi

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/home/${local.username}/.local/bin:$PATH"

    # Clone the GitHub repository with proper error handling
    cd "/home/${local.username}"

    # Handle three scenarios:
    # 1. Directory doesn't exist - fresh clone
    # 2. Directory exists with .git - update existing repo
    # 3. Directory exists without .git - corrupted state, clean and re-clone
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

    # Verify we're in the correct directory with a valid repo
    if [ ! -d ".git" ]; then
      echo "ERROR: Not in a valid git repository"
      exit 1
    fi

    echo "Current directory: $(pwd)"
    echo "Directory contents: $(ls -la)"

    # Run project init steps
    echo "Setting up virtual environment and installing dependencies..."

    # Create virtual environment only if it doesn't exist (idempotent)
    if [ ! -d ".venv" ]; then
      echo "Creating new virtual environment..."
      uv venv .venv
    else
      echo "Virtual environment already exists, skipping creation"
    fi

    source .venv/bin/activate

    # Run sync synchronously and wait for completion only if pyproject.toml exists
    if [ -f "pyproject.toml" ]; then
      echo "Found pyproject.toml, installing dependencies..."
      uv sync --dev
      sync_exit_code=$?

      # Ensure sync completed successfully before proceeding
      if [ $sync_exit_code -eq 0 ]; then
        echo "Dependencies installed successfully"
      else
        echo "Warning: uv sync exited with code $sync_exit_code"
      fi
    else
      echo "No pyproject.toml found in $(pwd), skipping dependency installation"
    fi

    # Wait a moment to ensure all installations are finalized
    sleep 2

    # Start Ollama
    echo "Starting Ollama..."
    bash "/home/${local.username}/${local.repo_name}/scripts/start-ollama.sh" || echo "Ollama startup failed, continuing..."

    # Run automatic onboarding
    echo "Running automatic onboarding..."
    if command -v onboard &> /dev/null; then
      onboard \
        --bootcamp-name "$BOOTCAMP_NAME" \
        --output-dir "/home/${local.username}/${local.repo_name}" \
        --test-script "/home/${local.username}/${local.repo_name}/aieng-llm-interp/tests/test_integration.py" \
        --env-example "/home/${local.username}/${local.repo_name}/.env.example" \
        --test-marker "integration_test" || echo "Onboarding failed, continuing..."
    else
      echo "Onboarding CLI not found, skipping automated onboarding"
    fi

    # Configure VS Code settings
    mkdir -p "/home/${local.username}/${local.repo_name}/.vscode"
    cat > "/home/${local.username}/${local.repo_name}/.vscode/settings.json" <<'VSCODE_SETTINGS'
{
  "python.terminal.useEnvFile": true
}
VSCODE_SETTINGS

    # Configure shell to always start in repo with venv activated
    # Only add to bashrc if not already present (idempotent)
    if ! grep -q "Auto-navigate to ${local.repo_name}" "/home/${local.username}/.bashrc" 2>/dev/null; then
      cat >> "/home/${local.username}/.bashrc" <<BASHRC

# Auto-navigate to ${local.repo_name} and activate venv
if [ -f ~/${local.repo_name}/.venv/bin/activate ]; then
    cd ~/${local.repo_name}
    source .venv/bin/activate
fi
BASHRC
    fi

    # Configure zshrc to auto-navigate to repo with venv activated
    if ! grep -q "Auto-navigate to ${local.repo_name}" "/home/${local.username}/.zshrc" 2>/dev/null; then
      cat >> "/home/${local.username}/.zshrc" <<ZSHRC

# Auto-navigate to ${local.repo_name} and activate venv
if [ -f ~/${local.repo_name}/.venv/bin/activate ]; then
    cd ~/${local.repo_name}
    source .venv/bin/activate
fi
ZSHRC
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

# See https://registry.terraform.io/modules/terraform-google-modules/container-vm
module "gce-container" {
  source  = "terraform-google-modules/container-vm/google"
  version = "3.0.0"

  container = {
    image   = var.container_image
    command = ["sh"]
    args    = ["-c", "chown -R ${local.username}:${local.username} /home/${local.username} && su - ${local.username} -s /bin/bash <<'CODER_SCRIPT'\n${coder_agent.main.init_script}\nCODER_SCRIPT\n"]
    securityContext = {
      privileged : true
    }
    # Declare volumes to be mounted
    # This is similar to how Docker volumes are mounted
    volumeMounts = [
      {
        mountPath = "/cache"
        name      = "tempfs-0"
        readOnly  = false
      },
      {
        mountPath = "/home/${local.username}"
        name      = "data-disk-0"
        readOnly  = false
      },
    ]
  }
  # Declare the volumes
  volumes = [
    {
      name = "tempfs-0"

      emptyDir = {
        medium = "Memory"
      }
    },
    {
      name = "data-disk-0"

      gcePersistentDisk = {
        pdName = "data-disk-0"
        fsType = "ext4"
      }
    },
  ]
}

resource "google_compute_disk" "pd" {
  project = var.project
  name    = "coder-${data.coder_workspace.me.id}-data-disk"
  type    = "pd-ssd"
  zone    = var.zone
  size    = local.is_gpu ? 100 : var.pd_size
}

resource "google_compute_instance" "dev" {
  zone         = var.zone
  count        = data.coder_workspace.me.start_count
  name         = "coder-${lower(data.coder_workspace_owner.me.name)}-${lower(data.coder_workspace.me.name)}"
  machine_type = local.selected_machine_type

  # GPU instances (G2 machine types) cannot live-migrate; they must be terminated on host maintenance.
  scheduling {
    on_host_maintenance = local.is_gpu ? "TERMINATE" : "MIGRATE"
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
      image = local.is_gpu ? local.gpu_boot_image : module.gce-container.source_image
      size  = local.is_gpu ? 100 : var.boot_disk_size
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
  metadata = local.is_gpu ? {
    # GPU: run the Coder agent directly on the DLVM — no Docker, no konlet.
    "startup-script" = local.gpu_startup_script
  } : {
    # CPU: use COS + konlet to run the workspace container.
    "gce-container-declaration" = module.gce-container.metadata_value
  }
  labels = local.is_gpu ? {} : {
    container-vm = module.gce-container.vm_container_label
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
    value = local.is_gpu ? local.gpu_boot_image : module.gce-container.container.image
  }

  item {
    key   = "instance_type"
    value = local.selected_machine_type
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

