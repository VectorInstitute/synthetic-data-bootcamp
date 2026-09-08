"""Direct Preference Optimization helpers for the optional Step 6 notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.syn_data.text.dpo.schemas import PreferencePair
from aieng.syn_data.text.evaluation import DEFAULT_EVAL_SYSTEM
from aieng.syn_data.text.sft import default_lora_config


def pairs_to_trl_rows(pairs: list[PreferencePair]) -> list[dict[str, Any]]:
    """Convert pairs to TRL's conversational preference format.

    TRL applies the model tokenizer's chat template to these messages. This keeps
    DPO training tokens aligned with :class:`Hf4BitInferenceClient` inference.
    """
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        prompt = [
            {
                "role": "system",
                "content": str(pair.metadata.get("system", DEFAULT_EVAL_SYSTEM)),
            },
            {"role": "user", "content": pair.prompt},
        ]
        rows.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": pair.chosen}],
                "rejected": [{"role": "assistant", "content": pair.rejected}],
            }
        )
    return rows


def build_dpo_dataset(pairs: list[PreferencePair]) -> Any:
    """Build a Hugging Face dataset for TRL DPO training."""
    from datasets import Dataset

    return Dataset.from_list(pairs_to_trl_rows(pairs))


def _require_cuda() -> None:
    import platform
    import sys

    import torch

    if torch.cuda.is_available():
        return

    hints: list[str] = [
        "DPO fine-tuning needs an NVIDIA GPU with CUDA (bitsandbytes 4-bit).",
        "Apple Silicon / MPS is not supported for this step.",
    ]
    if platform.system() == "Darwin":
        hints.append(
            "You are on macOS: run notebook 06 with RUN_DPO=0 locally, "
            "or use a Linux NVIDIA GPU (Colab, GCP GPU VM, etc.).",
        )
    elif (
        not Path("/usr/bin/nvidia-smi").exists()
        and not Path("/usr/local/nvidia/bin/nvidia-smi").exists()
    ):
        hints.append(
            "No NVIDIA driver visible in this environment. "
            "If using Docker, start the container with --gpus all and install "
            "nvidia-container-toolkit on the host.",
        )
    hints.append(
        f"torch={torch.__version__}, cuda_build={torch.version.cuda}, "
        f"platform={sys.platform}"
    )
    raise RuntimeError("\n".join(hints))


def train_lora_dpo(
    pairs: list[PreferencePair],
    output_dir: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    max_seq_length: int = 1024,
) -> Path:
    """Fine-tune a small model with 4-bit LoRA using TRL DPOTrainer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    _require_cuda()
    if not pairs:
        msg = "No preference pairs to train on."
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dpo_dataset(pairs)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = get_peft_model(model, LoraConfig(**default_lora_config()))

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            beta=beta,
            max_length=max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
