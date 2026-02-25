"""
Domain Fine-tuning Pipeline — End-to-end LoRA SFT execution script for Solutions Track.

Demonstrates:
1. Data preparation: Convert domain Q&A into chat format.
2. LoRA configuration and model loading (4-bit QLoRA).
3. Training with SFTTrainer.
4. Evaluation: Loss tracking + sample generation comparison.

Usage:
    pip install transformers peft trl datasets bitsandbytes accelerate
    python finetune_pipeline_demo.py

Note: Requires a GPU with ≥16GB VRAM for 7B models, or ≥8GB for 1-3B models.
"""

import json
from dataclasses import dataclass, field


# ── 1. Data Preparation ───────────────────────────────────────────

@dataclass
class SFTExample:
    system: str
    user: str
    assistant: str


def prepare_chat_dataset(
    examples: list[SFTExample],
    output_path: str = "sft_data.jsonl",
) -> str:
    """Convert domain Q&A pairs into chat-format JSONL for SFTTrainer."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "messages": [
                    {"role": "system", "content": ex.system},
                    {"role": "user", "content": ex.user},
                    {"role": "assistant", "content": ex.assistant},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Prepared {len(examples)} examples → {output_path}")
    return output_path


# ── 2. Model & LoRA Configuration ─────────────────────────────────

@dataclass
class FinetuneConfig:
    """All hyperparameters in one place for reproducibility."""
    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    # Output
    output_dir: str = "./ft_output"


def load_model_and_tokenizer(config: FinetuneConfig):
    """Load base model with optional 4-bit quantization + LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for LoRA
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── 3. Training ────────────────────────────────────────────────────

def train(config: FinetuneConfig, data_path: str):
    """Run SFT training with TRL's SFTTrainer."""
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_seq_length=config.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print(f"Starting training: {config.num_epochs} epochs, lr={config.learning_rate}")
    trainer.train()
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    print(f"Model saved to {config.output_dir}/final")
    return trainer


# ── 4. Evaluation: Before vs After ────────────────────────────────

def compare_outputs(
    base_model_name: str,
    finetuned_path: str,
    test_prompts: list[str],
    system_prompt: str = "You are a domain expert.",
):
    """Side-by-side comparison of base vs fine-tuned model outputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", trust_remote_code=True
    )
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

    # Fine-tuned model
    ft_model = PeftModel.from_pretrained(base_model, finetuned_path)
    ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=tokenizer)

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        base_out = base_pipe(formatted, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        ft_out = ft_pipe(formatted, max_new_tokens=256, do_sample=False)[0]["generated_text"]

        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"\n[BASE MODEL]:\n{base_out[len(formatted):]}")
        print(f"\n[FINE-TUNED]:\n{ft_out[len(formatted):]}")


# ── 5. Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Prepare sample domain data
    domain_examples = [
        SFTExample(
            system="You are a construction safety expert.",
            user="What PPE is required for welding operations?",
            assistant=(
                "For welding operations, the following PPE is mandatory per GB 50484-2008:\n"
                "1. Welding helmet with auto-darkening filter (shade 10-13)\n"
                "2. Flame-resistant jacket and pants\n"
                "3. Leather welding gloves\n"
                "4. Steel-toed boots with metatarsal guards\n"
                "5. Respiratory protection if ventilation is inadequate"
            ),
        ),
        SFTExample(
            system="You are a construction safety expert.",
            user="What is the maximum wind speed for crane operations?",
            assistant=(
                "Per JGJ 276-2012, crane operations must cease when:\n"
                "- Tower cranes: wind speed exceeds 20 m/s (Force 8)\n"
                "- Mobile cranes: wind speed exceeds 10.8 m/s (Force 6) for loads with large wind area\n"
                "- All cranes: during thunderstorms, regardless of wind speed\n"
                "The site safety officer must monitor weather forecasts and issue stop-work orders."
            ),
        ),
        SFTExample(
            system="You are a construction safety expert.",
            user="How often should scaffolding be inspected?",
            assistant=(
                "Scaffolding inspection requirements per JGJ 130-2011:\n"
                "1. Before first use after erection — full inspection by competent person\n"
                "2. Weekly routine inspection during use\n"
                "3. After any weather event (wind >15 m/s, heavy rain, snow)\n"
                "4. After any modification or partial dismantling\n"
                "5. After any period of non-use exceeding 30 days\n"
                "All inspections must be documented with date, inspector name, and findings."
            ),
        ),
    ]

    data_path = prepare_chat_dataset(domain_examples)

    # Step 2: Configure and train
    config = FinetuneConfig(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        num_epochs=3,
        batch_size=2,
        learning_rate=2e-4,
    )

    print("\n" + "="*60)
    print("  Fine-tuning Pipeline Demo")
    print("  Model:", config.base_model)
    print("  LoRA rank:", config.lora_r)
    print("  4-bit quantization:", config.use_4bit)
    print("="*60)

    # Uncomment to run actual training (requires GPU):
    # trainer = train(config, data_path)

    # Step 3: Compare outputs (after training)
    # compare_outputs(
    #     base_model_name=config.base_model,
    #     finetuned_path=f"{config.output_dir}/final",
    #     test_prompts=["What safety checks are needed before crane operation?"],
    # )

    print("\nDemo data prepared. Uncomment train() and compare_outputs() to run with GPU.")
