import os
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For quick runs, limit the number of training examples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit the number of validation examples")
    parser.add_argument("--max_predict_samples", type=int, default=None, help="Limit the number of test examples for prediction")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/evaluation")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set")
    parser.add_argument("--model_path", type=str, default="model", help="Path to save or load the model")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load MRPC dataset
    raw_datasets = load_dataset("glue", "mrpc")  # splits: train, validation, test
    # Initialize tokenizer and model
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # Tokenization function for sentence pairs
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["sentence1", "sentence2", "idx"])  # remove text columns
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]
    # Optionally subsample for quick debugging
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    if args.max_predict_samples:
        test_dataset = test_dataset.select(range(min(args.max_predict_samples, len(test_dataset))))
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Set up W&B run if training
    if args.do_train:
        wandb.init(project="bert-mrpc-finetune", name=f"lr{args.lr}_bs{args.batch_size}_epochs{args.num_train_epochs}")
        wandb.config.update({
            "learning_rate": args.lr,
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples
        })
        # Prepare optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(len(train_dataset) % args.batch_size != 0))
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        # Training loop
        model.train()
        global_step = 0
        train_losses = []
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
        for epoch in range(int(args.num_train_epochs)):
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)  # outputs includes loss and logits if labels provided
                loss = outputs.loss
                # Backpropagation
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # Log the loss to W&B and record it
                loss_val = loss.detach().cpu().item()
                wandb.log({"train_loss": loss_val}, step=global_step)
                train_losses.append(loss_val)
            # Optionally, could evaluate on eval_dataset each epoch (not required to meet tasks)
        # After training, evaluate on validation set
        model.eval()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collator)
        correct = 0
        total = 0
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # No labels in batch because we removed "label" column? Actually, keep label for eval.
            # If label was removed inadvertently, extract from batch differently. Check dataset columns.
            # Ensure eval_dataset has 'label'
            labels = batch.pop("label")
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().cpu().item()
            total += labels.size(0)
        val_accuracy = correct / total if total > 0 else 0
        wandb.log({"val_accuracy": val_accuracy})
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        # Save model and tokenizer
        os.makedirs(args.model_path, exist_ok=True)
        model.save_pretrained(args.model_path)
        tokenizer.save_pretrained(args.model_path)
        # Log results to res.txt
        with open("res.txt", "a") as f:
            f.write(f"lr={args.lr}, batch_size={args.batch_size}, epochs={args.num_train_epochs} => val_accuracy={val_accuracy*100:.2f}%\n")
        # Generate training loss plot and save as train_loss.png
        if train_losses:
            plt.figure(figsize=(6,4))
            plt.plot(train_losses, label="Training Loss")
            plt.title("Training Loss per Step")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("train_loss.png")
            plt.close()
        wandb.finish()
    # Prediction on test set
    if args.do_predict:
        # If we didn't just train a model, load from given path
        if not args.do_train:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model.to(device)
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset.remove_columns("label") if "label" in test_dataset.column_names else test_dataset,
                                                 batch_size=args.batch_size, collate_fn=data_collator)
        predictions = []
        for batch in test_loader:
            # For test, no labels involved
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
        # Write predictions to file
        with open("predictions.txt", "w") as f:
            f.write("index\tprediction\n")
            for idx, pred in enumerate(predictions):
                f.write(f"{idx}\t{pred}\n")

if __name__ == "__main__":
    main()
