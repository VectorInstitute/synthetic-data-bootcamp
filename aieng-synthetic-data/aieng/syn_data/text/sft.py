"""Supervised fine-tuning helpers for the optional Step 5 notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.syn_data.text.evaluation import DEFAULT_EVAL_SYSTEM, build_eval_prompt
from aieng.syn_data.text.schemas import QASample


def qa_samples_to_messages(
    samples: list[QASample],
) -> list[dict[str, list[dict[str, str]]]]:
    """Convert Q&A samples into chat-format training rows."""
    rows: list[dict[str, list[dict[str, str]]]] = []
    for sample in samples:
        user_prompt = build_eval_prompt(sample)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": DEFAULT_EVAL_SYSTEM},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": sample.gold_answer},
                ],
            },
        )
    return rows


def build_sft_dataset(samples: list[QASample]) -> Any:
    """Build a Hugging Face dataset for TRL supervised fine-tuning."""
    from datasets import Dataset

    return Dataset.from_list(qa_samples_to_messages(samples))


def default_lora_config() -> dict[str, Any]:
    """Return conservative LoRA settings for a 3B demo run."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def train_lora_sft(
    samples: list[QASample],
    output_dir: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
) -> Path:
    """Fine-tune a small model with 4-bit LoRA using TRL SFTTrainer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        import platform
        import sys

        hints: list[str] = [
            "LoRA fine-tuning needs an NVIDIA GPU with CUDA (bitsandbytes 4-bit).",
            "Apple Silicon / MPS is not supported for this step.",
        ]
        if platform.system() == "Darwin":
            hints.append(
                "You are on macOS: run notebook 05 with RUN_SFT=0 locally, "
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
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}, platform={sys.platform}"
        )
        raise RuntimeError("\n".join(hints))

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_sft_dataset(samples)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
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
        trust_remote_code=True,
    )
    model = get_peft_model(model, LoraConfig(**default_lora_config()))

    def formatting_func(example: dict[str, list[dict[str, str]]]) -> str:
        parts: list[str] = []
        for message in example["messages"]:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        ),
        formatting_func=formatting_func,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


class PeftInferenceClient:
    """Run inference with a saved LoRA adapter."""

    def __init__(self, adapter_dir: Path, base_model: str) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion from the fine-tuned adapter."""
        import torch

        system_text = system or DEFAULT_EVAL_SYSTEM
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{system_text}\n\n{prompt}\nAnswer:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()
