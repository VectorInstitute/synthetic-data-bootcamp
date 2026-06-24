packer {
  required_plugins {
    googlecompute = {
      source  = "github.com/hashicorp/googlecompute"
      version = "~> 1"
    }
  }
}

variable "project_id" {
  default = "pets-3-bootcamp"
}

variable "zone" {
  default = "us-central1-a"
}

# Build a custom GCE image on top of the PyTorch Deep Learning VM.
# Uses a CPU-only machine type during the build (no GPU needed for software install).
# The resulting image is used as the boot disk for GPU workspaces in the Coder template.
source "googlecompute" "gpu_boot" {
  project_id              = var.project_id
  source_image_family     = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
  source_image_project_id = ["deeplearning-platform-release"]
  zone                    = var.zone
  machine_type            = "n1-standard-4"
  disk_size               = 100
  image_name              = "synthetic-data-generation-bootcamp-gpu-{{timestamp}}"
  image_family            = "synthetic-data-generation-bootcamp-gpu"
  image_description       = "GPU boot image for Synthetic Data Generation Bootcamp (PyTorch DLVM + neovim + dotfiles)"
  ssh_username            = "packer"
  tags                    = ["packer"]
}

build {
  sources = ["source.googlecompute.gpu_boot"]

  provisioner "shell" {
    execute_command = "sudo sh -c '{{ .Vars }} {{ .Path }}'"
    inline = [
      # System packages
      "export DEBIAN_FRONTEND=noninteractive",
      "apt-get update -q",
      "apt-get install -y -q jq tmux zsh unzip vim wget default-jre ffmpeg",
      "rm -rf /var/lib/apt/lists/*",

      # Neovim from GitHub releases (must be before setup.sh so its version check skips the PPA fallback)
      "curl -fsSL https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.tar.gz -o /tmp/nvim.tar.gz",
      "tar xzf /tmp/nvim.tar.gz -C /opt/",
      "ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim",
      "rm /tmp/nvim.tar.gz",

      # Ollama
      "curl -fsSL https://ollama.com/install.sh | sh",

      # Ollama environment variables
      "sudo tee /etc/profile.d/bootcamp-env.sh >/dev/null <<'EOF'",
      "export OLLAMA_HOST=127.0.0.1:11434",
      "export OLLAMA_MODEL=qwen2.5:3b-instruct",
      "export SMALL_MODEL_BASE_URL=http://127.0.0.1:11434/v1",
      "export SMALL_MODEL_NAME=qwen2.5:3b-instruct",
      "export SMALL_MODEL_API_KEY=ollama",
      "EOF",
      "sudo chmod 644 /etc/profile.d/bootcamp-env.sh",

      # Create coder user with zsh shell
      "useradd --groups sudo --no-create-home --shell /usr/bin/zsh coder || true",
      "echo 'coder ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/coder",
      "chmod 0440 /etc/sudoers.d/coder",
      "mkdir -p /home/coder/.config/gcloud",
      "chown -R coder:coder /home/coder",

      # Dotfiles setup
      "git clone https://github.com/amrit110/dotfiles /tmp/dotfiles",
      "git -C /tmp/dotfiles checkout coder",
      "sed -i 's/POWERLEVEL9K_MODE=.*/POWERLEVEL9K_MODE=ascii/' /tmp/dotfiles/.p10k.zsh",
      "cd /tmp/dotfiles && HOME=/home/coder DEBIAN_FRONTEND=noninteractive bash setup.sh || echo 'Warning: setup.sh had non-fatal errors, continuing'",
      "cp -n /tmp/dotfiles/.p10k.zsh /home/coder/.p10k.zsh 2>/dev/null || true",
      "chown -R coder:coder /home/coder",
      "rm -rf /tmp/dotfiles",

      # Snapshot home to /opt/home-seed for first-boot seeding (same pattern as CPU image)
      "mkdir -p /opt/home-seed",
      "cp -a /home/coder/. /opt/home-seed/"
    ]
  }
}
