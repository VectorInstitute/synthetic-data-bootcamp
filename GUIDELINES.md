# GUIDELINES

These guidelines are recommended for anyone creating a repository using this template for building reference implementations.

## Repository Name

Choose a repository name that reflects the main topic of your reference implementations. Avoid generic terms like 'workshop', 'bootcamp', 'reference', or 'implementations'. Instead, use a concise topic name that best describes the content.

**Example:** Use `retrieval-augmented-generation` if your repository contains reference implementations for RAG concepts. Select a single, descriptive topic name for the project.

> **Note:** If you cannot use the recommended naming convention initially, you may start with a different name and update it later.

## Environment Variables

Manage environment variables using a `.env` file and access them in your code with `os.getenv("ENV_VARIABLE", "default-value")`. List all environment-specific variables in a `.env.example` file with placeholder values for easy reference and onboarding.

## Utility Packages

Place all common methods and classes used across implementations in a dedicated module at the repository root. Each package should have its own `pyproject.toml` specifying its details and dependencies. For example, this repository includes the `aieng-topic-impl` package.

If your repository contains multiple packages, link each one in the main `pyproject.toml` as shown below to ensure they are built and linked for local development:

```toml
[tool.uv.workspace]
members = [
  "aieng-topic-impl",
]

[tool.uv.sources]
aieng-topic-impl = { workspace = true }
```

When testing packages, use pre-release versions (e.g., v0.1.0a1, v0.1.0a2, v0.1.0b1). After testing, update to a release version (e.g., v1.0.0) before publishing. Follow the [official versioning scheme](https://packaging.python.org/en/latest/discussions/versioning/).

## Google Colab Integration (For Notebooks Only)

Ensure Jupyter Notebooks are runnable on Google Colab. This may require installing your locally linked package and resolving dependency conflicts in the Colab environment.

Add the following Markdown cell at the top of your notebook to enable opening it in Colab:

```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VectorInstitute/<REPO_NAME>/blob/main/<PATH_TO_NOTEBOOK>)
```

Include a Python cell like the one below at the beginning of your notebook to customize it for Colab:

```python
import os

if "COLAB_RELEASE_TAG" in os.environ:
    # Running in Google Colab
    # Install required dependencies
    !pip3 install numpy==1.26.4 torchvision==0.16.2 aieng-topic-impl
    # Uninstall conflicting dependencies
    !pip3 uninstall --yes torchao torchaudio torchdata torchsummary torchtune
```

## Dockerization

Dockerize your project to ensure portability and consistency across platforms. This also facilitates deployment on the AI Engineering Platform used in bootcamps and workshops.

- Update the provided `Dockerfile` to suit your project’s needs.
- Modify `scripts/start.sh` to reflect your setup steps. This script will run at container startup.
- Update the `README.md` with instructions to build and start the Docker container.

## GitHub Actions

### Publish

Use this GitHub Actions workflow to publish packages. Create a PyPI token and set the `PYPI_API_TOKEN` secret in your repository settings. To trigger the publish workflow, create a GitHub Release and push a new tag (e.g., `v0.1.0`).

After publishing, test the package by installing it in a new virtual environment and performing a sanity check.
