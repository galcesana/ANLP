import argparse
import os
import sys
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False
    print("Weights & Biases not found – continuing without experiment tracking.")


@dataclass
class Config:
    # I/O
    model_name: str = field(
        default="bert-base-uncased", metadata={"help": "Pre-trained model name or path"}
    )
    output_dir: str = field(default="checkpoints", metadata={"help": "Where to store checkpoints"})
    # Data subselection
    max_train_samples: int = field(default=-1)
    max_eval_samples: int = field(default=-1)
    max_predict_samples: int = field(default=-1)
    # Training hyper-parameters
    num_train_epochs: float = field(default=3)
    learning_rate: float = field(default=5e-5)
    batch_size: int = field(default=8)
    seed: int = field(default=42)

    # Modes
    do_train: bool = field(default=False)
    do_predict: bool = field(default=False)
    # Prediction-only – override model path if different from model_name
    model_path: Optional[str] = field(default=None)

    def to_args(self):
        return asdict(self)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fine‑tune BERT on MRPC paraphrase detection")

    # Data subset args
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)

    # Training HP
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)

    # Modes
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Model + paths
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model for prediction")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    cfg = Config(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        do_train=args.do_train,
        do_predict=args.do_predict,
        model_path=args.model_path,
    )

    if not cfg.do_train and not cfg.do_predict:
        parser.error("At least one of --do_train or --do_predict must be specified.")
    return cfg


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).astype(np.float32).mean().item()
    return {"accuracy": acc}


def main() -> None:
    cfg = parse_args()

    # Ensure deterministic behaviour
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load dataset
    raw_datasets = load_dataset("glue", "mrpc")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def preprocess(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    # Map datasets
    tokenized_datasets = raw_datasets.map(
        preprocess, batched=True, remove_columns=["sentence1", "sentence2", "idx"]
    )

    if cfg.max_train_samples != -1 and cfg.do_train:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(cfg.max_train_samples))
    if cfg.max_eval_samples != -1 and cfg.do_train:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(cfg.max_eval_samples)
        )
    if cfg.max_predict_samples != -1 and cfg.do_predict:
        tokenized_datasets["test"] = tokenized_datasets["test"].select(range(cfg.max_predict_samples))

    if cfg.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    else:
        # In prediction‑only mode, load supplied model path (required)
        load_path = cfg.model_path or cfg.model_name
        model = AutoModelForSequenceClassification.from_pretrained(load_path)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        evaluation_strategy="epoch" if cfg.do_train else "no",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",  # we only keep final checkpoint to save space
        seed=cfg.seed,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        report_to=["wandb"] if _WANDB_AVAILABLE and cfg.do_train else [],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"] if cfg.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if cfg.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if cfg.do_train else None,
    )

    if cfg.do_train:
        if _WANDB_AVAILABLE:
            wandb.init(project="anlp_ex1", config=cfg.to_args())
            wandb.watch(model, log="all", log_freq=1)

        trainer.train()

        # Evaluate on validation set & log results to res.txt
        eval_metrics = trainer.evaluate()
        val_acc = eval_metrics.get("eval_accuracy", 0.0)
        res_line = (
            f"epoch_num: {cfg.num_train_epochs}, lr: {cfg.learning_rate}, "
            f"batch_size: {cfg.batch_size}, eval_acc: {val_acc:.4f}\n"
        )
        with open("res.txt", "a", encoding="utf-8") as f:
            f.write(res_line)
        print("Validation results:", eval_metrics)

        # Save final model
        ckpt_path = os.path.join(cfg.output_dir, "final_model")
        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        if _WANDB_AVAILABLE:
            wandb.finish()

    if cfg.do_predict:
        # Ensure model is in eval mode
        model.eval()
        test_dataset = tokenized_datasets["test"]
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        # Build predictions.txt in required format
        sent1 = raw_datasets["test"]["sentence1"]
        sent2 = raw_datasets["test"]["sentence2"]
        assert len(sent1) == len(preds)

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for s1, s2, label in zip(sent1, sent2, preds):
                f.write(f"{s1}###{s2}###{label}\n")
        print(f"Saved predictions for {len(preds)} samples → predictions.txt")


if __name__ == "__main__":
    main()
