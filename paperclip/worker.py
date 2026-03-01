"""
Fine-tuning worker.

Modes:
  MOCK (default / PAPERCLIP_MOCK=1):
    Simulates training with a progress ticks loop. No GPU required.

  REAL (requires pip install paperclip-daemon[gpu]):
    Uses HuggingFace transformers + PEFT (LoRA) to fine-tune the model.
    - dataset_type='tts'  → TTS/speech model path (audio + transcript pair)
    - all other types     → causal LM path (text dataset)
"""
import time
import logging
from pathlib import Path
from typing import Callable
from paperclip import config

logger = logging.getLogger(__name__)


def run_job(job: dict, progress_callback: Callable[[int, str], None]) -> None:
    if config.is_mock():
        _run_mock(job, progress_callback)
    else:
        if job.get("dataset_type") == "tts":
            _run_real_tts(job, progress_callback)
        else:
            _run_real_causal(job, progress_callback)


# ── Mock mode ─────────────────────────────────────────────────────────────────

def _run_mock(job: dict, cb: Callable) -> None:
    """Fake training loop: ticks 0→100 over ~30 seconds."""
    model_id = job.get("model_id", "unknown")
    epochs = int((job.get("config") or {}).get("epochs", 3))
    total_steps = epochs * 10
    logger.info(f"[MOCK] Starting mock fine-tune of {model_id} for {epochs} epochs")
    for step in range(total_steps + 1):
        cb(int((step / total_steps) * 100), "running")
        time.sleep(1.5)
    cb(100, "completed")
    logger.info(f"[MOCK] Job {job['id']} completed")


# ── Real — Causal LM (text datasets) ─────────────────────────────────────────

def _run_real_causal(job: dict, cb: Callable) -> None:
    """LoRA fine-tuning for standard causal language models."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("GPU deps not installed. Run: pip install paperclip-daemon[gpu]")

    model_id = job["model_id"]
    cfg = job.get("config") or {}
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 2e-4))
    batch_size = int(cfg.get("batch_size", 4))
    hf_dataset_id = cfg.get("hf_dataset")

    logger.info(f"[REAL] Loading {model_id}")
    cb(5, "running")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cb(15, "running")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    cb(30, "running")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    cb(40, "running")

    # Load dataset from HF hub or fall back to placeholder
    if hf_dataset_id:
        from datasets import load_dataset
        logger.info(f"[REAL] Loading HF dataset: {hf_dataset_id}")
        raw = load_dataset(hf_dataset_id, split="train")
        text_col = next((c for c in ("text", "instruction", "input", "content") if c in raw.column_names), raw.column_names[0])
        texts = raw[text_col][:2000]
    else:
        texts = ["The quick brown fox jumps over the lazy dog."] * 100

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("input_ids", "labels")
    cb(50, "running")

    class _ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.max_steps > 0:
                cb(min(int((state.global_step / state.max_steps) * 50) + 50, 99), "running")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"/tmp/paperclip-job-{job['id']}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            fp16=True,
            report_to="none",
        ),
        train_dataset=tokenized,
        callbacks=[_ProgressCB()],
    )
    trainer.train()
    cb(100, "completed")
    logger.info(f"[REAL] Job {job['id']} complete. Weights at /tmp/paperclip-job-{job['id']}")


# ── Real — TTS / Speech model (audio + transcript) ────────────────────────────

def _run_real_tts(job: dict, cb: Callable) -> None:
    """
    Fine-tunes a speech model (e.g. Voxtral) on a paired audio + transcript.

    Expects uploaded files: one audio (.wav/.mp3/.flac/.m4a) + one transcript (.txt).
    Uses AutoProcessor for audio feature extraction and LoRA for efficient fine-tuning.
    """
    try:
        import torch
        import numpy as np
        import librosa
        from transformers import (
            AutoProcessor,
            AutoModelForSpeechSeq2Seq,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            TrainerCallback,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            f"TTS GPU deps not installed ({e}).\n"
            "Run: pip install paperclip-daemon[gpu]"
        )

    from paperclip import api_client, config as daemon_config

    model_id = job["model_id"]
    cfg = job.get("config") or {}
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 2e-4))
    batch_size = int(cfg.get("batch_size", 4))
    job_dir = f"/tmp/paperclip-job-{job['id']}"
    output_dir = f"{job_dir}/output"

    # ── Download uploaded files ───────────────────────────────────────────
    logger.info("[TTS] Downloading job files from API…")
    cb(5, "running")

    device_token = daemon_config.device_token()
    try:
        downloaded = api_client.download_job_files(device_token, job["id"], job_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download job files: {e}")

    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_path = next((f for f in downloaded if Path(f).suffix.lower() in audio_exts), None)
    transcript_path = next((f for f in downloaded if Path(f).suffix.lower() == ".txt"), None)

    if not audio_path:
        raise RuntimeError("No audio file found in uploaded files (.wav / .mp3 / .flac / .m4a)")
    if not transcript_path:
        raise RuntimeError("No transcript file found (.txt)")

    logger.info(f"[TTS] Audio: {Path(audio_path).name}, Transcript: {Path(transcript_path).name}")
    cb(15, "running")

    # ── Load audio (resample to 16 kHz mono) ─────────────────────────────
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    logger.info(f"[TTS] Audio loaded: {len(audio_array)/16000:.1f}s at 16 kHz")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    logger.info(f"[TTS] Transcript: {transcript[:80]}{'…' if len(transcript) > 80 else ''}")
    cb(25, "running")

    # ── Load processor + model ───────────────────────────────────────────
    logger.info(f"[TTS] Loading processor for {model_id}…")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    cb(40, "running")

    logger.info(f"[TTS] Loading model {model_id}…")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    cb(55, "running")

    # ── Apply LoRA ───────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    cb(60, "running")

    # ── Build dataset ─────────────────────────────────────────────────────
    # Replicate the single example so the trainer has enough steps
    n_repeats = max(20, epochs * 10)
    raw_dataset = Dataset.from_dict({
        "audio": [audio_array] * n_repeats,
        "text":  [transcript]  * n_repeats,
    })

    def preprocess(batch):
        inputs = processor(
            batch["audio"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        with processor.as_target_processor():
            labels = processor(batch["text"], return_tensors="pt", padding=True).input_ids
        inputs["labels"] = labels
        return inputs

    dataset = raw_dataset.map(preprocess, batched=True, remove_columns=["audio", "text"])
    cb(65, "running")

    # ── Training ─────────────────────────────────────────────────────────
    class _ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.max_steps > 0:
                cb(min(int((state.global_step / state.max_steps) * 30) + 65, 99), "running")

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=5,
            fp16=True,
            predict_with_generate=False,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True),
        callbacks=[_ProgressCB()],
    )

    trainer.train()
    cb(100, "completed")
    logger.info(f"[TTS] Job {job['id']} complete. LoRA weights at {output_dir}")
