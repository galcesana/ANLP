import argparse
import os

import wandb
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, \
    DataCollatorWithPadding
import evaluate


def load_splits(ds, max_train_samples, max_eval_samples, max_predict_samples):
    train_split, val_split, test_split = ds["train"], ds["validation"], ds["test"]
    if max_train_samples != -1:
        train_split = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    if max_eval_samples != -1:
        val_split = ds["validation"].select(range(min(max_eval_samples, len(ds["validation"]))))
    if max_predict_samples != -1:
        test_split = ds["test"].select(range(min(max_predict_samples, len(ds["test"]))))
    return train_split, val_split, test_split


def preprocess_function(tokenizer, examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True
    )


def compute_metric(pred, metric):
    predictions = np.argmax(pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=pred.label_ids)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_predict_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--model_path', type=str, default='./models')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = load_dataset("nyu-mll/glue", "mrpc")
    train_split, val_split, test_split = load_splits(ds, args.max_train_samples, args.max_eval_samples,
                                                     args.max_predict_samples)

    labels = {0: "not_equivalent", 1: "equivalent"}

    config = {
        "checkpoint": "bert-base-uncased",
        "output_dir": "./results",
        "eval_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "wandb",
        "fp16": True,

    }

    if args.do_train:
        print("Training with:")
        print(f"Max train samples: {args.max_train_samples}")
        print(f"Max evaluation samples: {args.max_eval_samples}")
        print(f"Epochs: {args.num_train_epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
        # Placeholder: train_model(args)
        run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
        wandb.init(project="anlp-ex-1", name=run_name)
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
        tokenized_train = train_split.map(lambda examples: preprocess_function(tokenizer, examples),
                                          batched=True).remove_columns(["sentence1", "sentence2"])
        tokenized_val = val_split.map(lambda examples: preprocess_function(tokenizer, examples),
                                      batched=True).remove_columns(["sentence1", "sentence2"])
        accuracy_metric = evaluate.load("accuracy")
        model = AutoModelForSequenceClassification.from_pretrained(config["checkpoint"], num_labels=2)
        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy=config["eval_strategy"],
            save_strategy=config["save_strategy"],
            logging_strategy=config["logging_strategy"],
            logging_steps=config["logging_steps"],
            report_to=config["report_to"],
            run_name=run_name,
            fp16=config["fp16"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=lambda pred: compute_metric(pred=pred, metric=accuracy_metric),
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        os.makedirs("./models", exist_ok=True)
        trainer.train()
        trainer.save_model(f"./models/{run_name}")

    if args.do_predict:
        print("Predicting with:")
        print(f"Model path: {args.model_path}")
        print(f"Max predict samples: {args.max_predict_samples}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        data_collator = DataCollatorWithPadding(tokenizer)

        tokenized_test = test_split.map(lambda examples: preprocess_function(tokenizer, examples),
                                        batched=True).remove_columns(["sentence1", "sentence2"])

        prediction_args = TrainingArguments(
            output_dir=config["output_dir"],
            eval_strategy="no",
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=prediction_args,
            processing_class=tokenizer,
            data_collator=data_collator
        )

        test_results = trainer.predict(tokenized_test)
        predictions = test_results.predictions
        predicted_labels = np.argmax(predictions, axis=1)

        if test_results.label_ids is not None:
            accuracy_metric = evaluate.load("accuracy")
            metrics = accuracy_metric.compute(predictions=predicted_labels, references=test_results.label_ids)
            print("Test Accuracy:", metrics["accuracy"])

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for ex, label in zip(test_split, predicted_labels):
                f.write(f"{ex['sentence1']}###{ex['sentence2']}###{label}\n")