#!/usr/bin/env python3
"""
Evaluate a finalized CausalLM with EMOTION injected in the *user* turn.

- Leaves the system message UNCHANGED (unless you pass one explicitly)
- Validates emotion against emotions.json in the model dir (if present)
- Generates, prints samples, and reports mean new length / clipped ratio
- Optional: score outputs with a reward classifier
"""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_reward(reward_model_dir: Path | None):
    if not reward_model_dir:
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    r_tok = AutoTokenizer.from_pretrained(reward_model_dir)
    r_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_dir,
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None),
        device_map=("auto" if torch.cuda.is_available() else None),
    ).eval()
    def score(texts):
        with torch.no_grad():
            enc = r_tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = r_model(**enc).logits
            return torch.softmax(logits, -1)[:, 1].detach().cpu().tolist()
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="Path to finalized model folder")
    ap.add_argument("--prompts_file", type=Path, default=None, help="One prompt per line")
    ap.add_argument("--emotion", type=str, default=None, help="Emotion to apply (exact token as trained)")
    ap.add_argument("--emotions_file", type=Path, default=None, help="Override path to emotions.json")
    ap.add_argument("--system", type=str, default=None,
                    help="Optional system string to include verbatim. If omitted, no system msg is added.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--reward_model", type=Path, default=None, help="Optional classifier dir for scoring")
    ap.add_argument("--num", type=int, default=8, help="How many prompts to run if no file is given")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map=("auto" if torch.cuda.is_available() else None),
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                     else torch.float16 if torch.cuda.is_available() else None),
        trust_remote_code=True,
    ).eval()

    # Load allowed emotions (if provided in the repo)
    emo_path = args.emotions_file or (args.model_dir / "emotions.json")
    emotions = []
    if emo_path.exists():
        try:
            emotions = json.loads(emo_path.read_text())
        except Exception:
            pass
    if args.emotion:
        if emotions and args.emotion not in emotions:
            raise SystemExit(f"Emotion '{args.emotion}' not in allowed set: {emotions}")

    # Build prompts
    if args.prompts_file and args.prompts_file.exists():
        prompts = [ln.strip() for ln in args.prompts_file.read_text().splitlines() if ln.strip()]
    else:
        prompts = [
            "Describe the River Liffey and its place in Dublin’s story.",
            "Explain the craft of thatching as practiced in rural Ireland.",
            "Tell of a market day in Kilkenny in the mid-19th century.",
            "What are the virtues of peat as fuel, and its perils?",
            "On the antiquities of Tara—what remains and what is conjectured?",
            "A portrait of a humble stone bridge on a misty morning.",
            "Relate the habits of the salmon and the angler’s art.",
            "Sketch the bustle of a mail-coach arriving at twilight.",
        ][: args.num]

    def build_messages(user_text: str):
        msgs = []
        if args.system is not None:
            # You control this; if omitted, we don't add a system msg at all.
            msgs.append({"role": "system", "content": args.system})
        user_content = f"EMOTION: {args.emotion}\n\n{user_text}" if args.emotion else user_text
        msgs.append({"role": "user", "content": user_content})
        return msgs

    # Render chat prompts via tokenizer’s chat_template (unchanged)
    msgs = [build_messages(p) for p in prompts]
    prompt_texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in msgs
    ]

    # Tokenize batch
    batch = tok(prompt_texts, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}

    gen_cfg = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        out = model.generate(**batch, **gen_cfg)

    # Compute new-token lengths & decode only the generated part
    input_lens = [int(x) for x in (batch["input_ids"].ne(tok.pad_token_id).sum(dim=1)).tolist()]
    texts = []
    new_lengths = []
    for i in range(out.size(0)):
        gen_ids = out[i, input_lens[i]:]  # slice off the prompt
        new_lengths.append(int(gen_ids.size(0)))
        texts.append(tok.decode(gen_ids, skip_special_tokens=True))

    clipped = sum(1 for L in new_lengths if L >= args.max_new_tokens)
    clipped_ratio = clipped / max(1, len(new_lengths))

    reward_fn = load_reward(args.reward_model)
    reward_vals = reward_fn(texts) if reward_fn else None

    # Output
    print("\n=== Samples (truncated) ===")
    for i, t in enumerate(texts[:5]):
        print(f"\n[{i}] {t[:2600]}{'…' if len(t) > 2600 else ''}")

    print("\n=== Metrics ===")
    print(f"mean_new_len      : {sum(new_lengths)/len(new_lengths):.1f}")
    print(f"clipped_ratio     : {clipped_ratio:.3f}")
    if reward_vals:
        print(f"mean_reward (cls) : {sum(reward_vals)/len(reward_vals):.4f}")

    if emotions:
        print("\n(emotions available) ", emotions)

if __name__ == "__main__":
    main()

