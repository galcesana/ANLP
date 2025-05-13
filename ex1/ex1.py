import argparse
import os
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate

def get_splits(dataset, train_lim, eval_lim, pred_lim):
    train_set = dataset['train']
    val_set = dataset['validation']
    test_set = dataset['test']
    if train_lim != -1:
        train_set = train_set.select(range(min(train_lim, len(train_set))))
    if eval_lim != -1:
        val_set = val_set.select(range(min(eval_lim, len(val_set))))
    if pred_lim != -1:
        test_set = test_set.select(range(min(pred_lim, len(test_set))))
    return train_set, val_set, test_set


def tokenize_batch(tokenizer, examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)


def evaluate_accuracy(predictions, metric):
    labels = np.argmax(predictions.predictions, axis=1)
    return metric.compute(predictions=labels, references=predictions.label_ids)


def train_model(args):
    run_id = f"ep{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
    print(f"Starting training: {run_id}")
    wandb.login()
    wandb.init(project="anlp-ex-1", name=run_id)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds = args.train_ds.map(
        lambda batch: tokenize_batch(tokenizer, batch),
        batched=True
    ).remove_columns(['sentence1', 'sentence2'])
    val_ds = args.val_ds.map(
        lambda batch: tokenize_batch(tokenizer, batch),
        batched=True
    ).remove_columns(['sentence1', 'sentence2'])

    metric = evaluate.load('accuracy')
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_steps=1,
        report_to='wandb',
        run_name=run_id,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda preds: evaluate_accuracy(preds, metric),
        tokenizer=tokenizer,
        data_collator=collator
    )

    os.makedirs(args.model_path, exist_ok=True)
    trainer.train()
    trainer.save_model(os.path.join(args.model_path, run_id))


def predict_model(args):
    print(f"Running inference with model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    collator = DataCollatorWithPadding(tokenizer)

    test_ds = args.test_ds.map(
        lambda batch: tokenize_batch(tokenizer, batch),
        batched=True
    ).remove_columns(['sentence1', 'sentence2'])

    pred_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to='none'
    )
    trainer = Trainer(
        model=model,
        args=pred_args,
        tokenizer=tokenizer,
        data_collator=collator
    )
    results = trainer.predict(test_ds)
    preds = np.argmax(results.predictions, axis=1)

    if results.label_ids is not None:
        acc = evaluate.load('accuracy').compute(
            predictions=preds, references=results.label_ids
        )['accuracy']
        print('Test accuracy:', acc)

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, 'predictions.txt')
    with open(out_file, 'w', encoding='utf-8') as fout:
        for example, label in zip(args.raw_test, preds):
            fout.write(f"{example['sentence1']}###{example['sentence2']}###{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and/or predict on GLUE MRPC with BERT'
    )
    parser.add_argument('--max_train_samples',   type=int,   default=-1)
    parser.add_argument('--max_eval_samples',    type=int,   default=-1)
    parser.add_argument('--max_predict_samples', type=int,   default=-1)
    parser.add_argument('--num_train_epochs',    type=int,   default=2)
    parser.add_argument('--lr',                  type=float, default=2e-5)
    parser.add_argument('--batch_size',          type=int,   default=16)
    parser.add_argument('--do_train',            action='store_true')
    parser.add_argument('--do_predict',          action='store_true')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Directory to save/load model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory for outputs'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # prepare data
    dataset = load_dataset('glue', 'mrpc')
    train_ds, val_ds, test_ds = get_splits(
        dataset,
        args.max_train_samples,
        args.max_eval_samples,
        args.max_predict_samples
    )
    # stash raw test for writing
    args.raw_test = dataset['test']
    args.train_ds = train_ds
    args.val_ds = val_ds
    args.test_ds = test_ds

    if args.do_train:
        train_model(args)
    if args.do_predict:
        predict_model(args)

if __name__ == '__main__':
    main()
