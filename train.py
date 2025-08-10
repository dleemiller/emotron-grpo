#!/usr/bin/env python
"""
GRPO fine-tuning of HuggingFaceTB/SmolLM3-3B with Liger + fast rollouts.

• Preserve SmolLM3 chat template (no system overrides)
• User turn:  EMOTION: <label> + original prompt  (+ optional hint on a fraction)
• Reward: sentiment on final answer AFTER stripping <think>...</think>
• Speed-ups: Liger GRPO loss, optional Liger kernels, KV-cache during rollout only
• Single RTX A6000 (48 GB). LoRA r=16 default. vLLM off.
"""

import argparse, random, re
from pathlib import Path
from contextlib import contextmanager

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOConfig, GRPOTrainer

from llm_judge import judge_response

# ---------------------------- CLI ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # Reasoning mode probability
    ap.add_argument("--think-prob", type=float, default=0.25, help="Probability to enable /think per sample [0..1].")
    ap.add_argument("--convention-prob", type=float, default=0.33,
                    help="Fraction of samples with a tiny hint explaining EMOTION tag.")
    # Throughput / capacity
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--generation-batch-size", type=int, default=0, help="If 0, auto-snaps to a valid multiple.")
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--max-steps", type=int, default=5000)
    # Generation behavior
    ap.add_argument("--temperature", type=float, default=0.9)  # SmolLM3 rec
    ap.add_argument("--top-p", type=float, default=0.95)       # SmolLM3 rec
    ap.add_argument("--max-completion-length", type=int, default=1536)
    # Optimizations
    ap.add_argument("--use-liger-loss", action="store_true", default=False)
    ap.add_argument("--use-liger-kernel", action="store_true", default=False)
    ap.add_argument("--use-flash-attn", action="store_true", default=False)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ---------------------------- Constants ----------------------------
MODEL_NAME      = "HuggingFaceTB/SmolLM3-3B"
SENTIMENT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DATASET_NAME    = "WizardLMTeam/WizardLM_evol_instruct_V2_196k"
OUTPUT_DIR      = "./smollm3-grpo-emotion"
LOG_DIR         = f"{OUTPUT_DIR}/logs"
EMOTIONS        = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------- Utils ----------------------------
THINK_RE   = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
NO_ECHO_RE = re.compile(r"\bEMOTION\s*:", flags=re.IGNORECASE)
USER_BLOCK_RE = re.compile(
    r"<\|im_start\|>\s*user\s*\n(.*?)\s*<\|im_end\|>",
    re.DOTALL | re.IGNORECASE,
)

def extract_user_prompt(text: str) -> str | None:
    """
    Return the first user prompt found between <|im_start|>user ... <|im_end|>.
    Strips leading/trailing whitespace. Returns None if not found.
    """
    m = USER_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

@contextmanager
def rollout_cache(model):
    """Enable KV cache and temporarily disable grad-ckpt ONLY for generation."""
    was_ckpt = getattr(model, "is_gradient_checkpointing", False)
    try:
        if was_ckpt:
            model.gradient_checkpointing_disable()
        old = getattr(model.config, "use_cache", False)
        model.config.use_cache = True  # use cache during rollout (defaults True per HF docs)
        torch.set_grad_enabled(False)
        yield
    finally:
        torch.set_grad_enabled(True)
        model.config.use_cache = False
        if was_ckpt:
            model.gradient_checkpointing_enable()

# ---------------------------- Main ----------------------------
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ----- Tokenizer (keep chat template; no system message) -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ----- Model (optional Liger kernels / Flash-Attn2) -----
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"  # needs flash-attn installed

    model = None
    if args.use_liger_kernel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model = AutoLigerKernelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
            print("✅ Liger kernels: enabled (AutoLigerKernelForCausalLM).")
        except Exception as e:
            print(f"⚠️  Liger kernels not applied ({e}). Falling back to HF model.")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    # ----- Monkey-patch model.generate to use cache during rollout -----
    _orig_generate = model.generate
    def _cached_generate(*a, **k):
        with rollout_cache(model):
            # be explicit: pass use_cache=True to generate as well
            k.setdefault("use_cache", True)
            return _orig_generate(*a, **k)
    model.generate = _cached_generate

    # ----- LoRA -----
    #lora_cfg = LoraConfig(
    #    r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
    #    task_type="CAUSAL_LM", target_modules="all-linear",
    #)

    # ----- Dataset formatting via chat template -----
    CONVENTION = (
        "When a line starts with `EMOTION: <label>`, respond in that emotion and do not mention the label.\n"
        "Your response must resemble someone expressing the emotion naturally. Your response should not appear staged to a reader.\n"
        "There should be no breaks in character which would inhibit the reader's ability to suspend disbelief.\n"
        "You will respond naturally in character of the emotion with **no other commentary**, narration or indications that you have been instructed to do so.\n"
        "Emotion should _only_ be detected through the tone of your response with **no other indications**.\n"
        "You are responding in the heat of the moment, and your response **must not demonstrate congnition of the emotion you convey**.\n"
    )

    def format_example(example):
        emo = rng.choice(EMOTIONS)
        enable_think = rng.random() < args.think_prob
        user_raw = example["conversations"][0]["value"]
        prefix = (f"{CONVENTION}\nEMOTION: {emo}\n\n") if (rng.random() < args.convention_prob) else (f"EMOTION: {emo}\n\n")
        messages = [{"role": "user", "content": prefix + user_raw}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_think)
        return {"prompt": prompt, "emotion": emo}

    raw_ds = load_dataset(DATASET_NAME, split="train")
    train_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names).shuffle(seed=args.seed)

    # ----- Reward: sentiment on post-<think> answer (+ tiny no-echo penalty) -----
    # Move classifier to GPU 0; you have VRAM headroom.
    emotion_pipe = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None, device=0)

    def sentiment_reward_fn(prompts, completions, emotion, **_):
        visibles = [strip_think(c) for c in completions]
        preds = emotion_pipe(visibles, truncation=True)
        rewards = []
        for prompt, text, pred, tgt in zip(prompts, visibles, preds, emotion):
            emo_prob = next((p["score"] for p in pred if p["label"].lower() == tgt), 0.0)
            echo_pen = 0.05 if NO_ECHO_RE.search(text) else 0.0
            rewards.append(max(0.0, min(1.0, emo_prob - echo_pen)))
        return rewards

    def judge_reward_fn(prompts, completions, emotion, **_):
        visibles = [strip_think(c) for c in completions]
        preds = emotion_pipe(visibles, truncation=True)
        rewards = []
        for prompt, text, pred, tgt in zip(prompts, visibles, preds, emotion):
            user = extract_user_prompt(prompt)
            judge_score = judge_response(user, tgt, text)
            rewards.append(judge_score)
        return rewards


    # ----- Batch math: generation_batch_size must be divisible by num_generations -----
    base = args.per_device_train_batch_size * args.gradient_accumulation_steps  # world_size=1
    gen_bs = args.generation_batch_size or max(args.num_generations, (base // args.num_generations) * args.num_generations)
    if gen_bs % args.num_generations != 0:
        gen_bs = (gen_bs // args.num_generations) * args.num_generations or args.num_generations
    print(f"[dbg] base={base} num_generations={args.num_generations} -> generation_batch_size={gen_bs}")

    # ----- GRPO config -----
    grpo_cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        report_to=["tensorboard"],
        seed=args.seed,

        # Loss / trust-region
        beta=0.0,
        epsilon=0.2,
        loss_type="dr_grpo",
        scale_rewards=False,
        mask_truncated_completions=True,
        use_liger_loss=args.use_liger_loss,  # memory-friendly GRPO loss

        # Online generation (no early_stopping: it's beam-only)
        num_generations=args.num_generations,
        generation_batch_size=gen_bs,
        max_completion_length=args.max_completion_length,
        generation_kwargs=dict(
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            cache_implementation="static",
            max_new_tokens=args.max_completion_length,
            use_cache=True,
        ),

        # Efficiency
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        learning_rate=args.learning_rate,
        use_vllm=False,

        # Horizon
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
    )

    # ----- Trainer -----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[judge_reward_fn, sentiment_reward_fn],
        train_dataset=train_ds,
        args=grpo_cfg,
        processing_class=tokenizer,
        #peft_config=lora_cfg,
    )

    # ----- Train → merge LoRA → save -----
    trainer.train()
    merged = trainer.model.merge_and_unload()
    save_path = Path(OUTPUT_DIR) / "merged"
    merged.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Done. Merged model saved to: {save_path}")
    print(f"🔭 TensorBoard: tensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    main()

