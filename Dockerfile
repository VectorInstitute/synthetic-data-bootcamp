FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Expose ports for JupyterLab and Ollama
EXPOSE 8888 11434

# Install system dependencies and Ollama (binary only; start manually via
# scripts/start-ollama.sh when a reference implementation needs a local SLM)
RUN apt-get update \
    && apt-get install -y \
    sudo \
    curl \
    git \
    jq \
    tar \
    unzip \
    ca-certificates \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODEL=qwen2.5:0.5b-instruct
ENV SMALL_MODEL_BASE_URL=http://127.0.0.1:11434/v1
ENV SMALL_MODEL_NAME=qwen2.5:0.5b-instruct
ENV SMALL_MODEL_API_KEY=ollama

# !!IMPORTANT!!
# THIS SECTION SHOULD NOT BE MODIFIED AS
# IT IS USED TO MAKE THIS IMAGE COMPATIBLE WITH CODER
#######################################################################
ARG USER=coder
RUN useradd --groups sudo --no-create-home --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}
########################################################################

# Copy the code into the container
COPY --chown=${USER}:${USER} . /home/${USER}/aieng-synthetic-data

# Start the container and run the project setup script
CMD ["bash", "aieng-synthetic-data/scripts/setup.sh"]
