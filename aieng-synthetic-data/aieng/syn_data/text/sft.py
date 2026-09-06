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


def qa_samples_to_prompt_completion(
    samples: list[QASample],
) -> list[dict[str, list[dict[str, str]]]]:
    """Split chat rows into TRL prompt-completion pairs.

    TRL treats this layout as a conversational prompt-completion dataset, so it
    renders both halves with the model's own chat template and defaults
    ``completion_only_loss`` to True. That keeps training tokens identical to
    what :class:`Hf4BitInferenceClient` sends at inference, and keeps the loss
    off the system prompt and the grounding passage.
    """
    rows: list[dict[str, list[dict[str, str]]]] = []
    for row in qa_samples_to_messages(samples):
        messages = row["messages"]
        rows.append({"prompt": messages[:-1], "completion": messages[-1:]})
    return rows


def build_sft_dataset(samples: list[QASample]) -> Any:
    """Build a Hugging Face prompt-completion dataset for TRL fine-tuning."""
    from datasets import Dataset

    return Dataset.from_list(qa_samples_to_prompt_completion(samples))


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
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    trust_remote_code: bool = False,
) -> Path:
    """Fine-tune a small model with 4-bit LoRA using TRL SFTTrainer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        import platform
        import sys

        hints: list[str] = [
            "LoRA fine-tuning needs an NVIDIA GPU with CUDA (bitsandbytes 4-bit).",
            "Apple Silicon / MPS is not supported for this step.",
        ]
        if platform.system() == "Darwin":
            hints.append(
                "You are on macOS: notebook 05 LoRA needs a Linux NVIDIA GPU "
                "(Colab, GCP GPU VM, etc.).",
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
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=trust_remote_code
    )
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
        trust_remote_code=trust_remote_code,
    )
    model = get_peft_model(model, LoraConfig(**default_lora_config()))

    # No formatting_func: the dataset is conversational prompt-completion, so TRL
    # renders it with the tokenizer's own chat template (matching inference) and
    # masks the prompt tokens out of the loss.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_length=max_seq_length,
            completion_only_loss=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


class Hf4BitInferenceClient:
    """Run inference with a 4-bit Hugging Face model, optionally plus a LoRA adapter.

    Use this (not Ollama) as the same-stack control for notebook 05: the
    un-adapted base model is loaded with the same bitsandbytes nf4 path as
    the PEFT adapters.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: Path | None = None,
    ) -> None:
        import torch
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model
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
        """Generate a completion from the 4-bit model (and adapter, if loaded)."""
        import torch

        del response_format
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

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                pad_token_id=pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def release(self) -> None:
        """Drop GPU weights so a later train/eval step can load another model."""
        import gc

        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class PeftInferenceClient(Hf4BitInferenceClient):
    """Run inference with a saved LoRA adapter on the 4-bit base model."""

    def __init__(self, adapter_dir: Path, base_model: str) -> None:
        super().__init__(base_model, adapter_dir=adapter_dir)
