import os
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        input_texts = data["text"]
        targets = data["target"]
        self.inputs = []
        self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx].squeeze(0),
        }

    def __len__(self):
        return len(self.labels)


# seed 고정
def seed_fix(SEED=456):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# config parser로 가져오기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml"
    )  # 입력 없을 시, 기본값으로 config.yaml을 가져옴
    return parser.parse_args()


def data_setting(CFG, SEED, data_dir, tokenizer):
    data = pd.read_csv(data_dir)
    dataset_train, dataset_valid = train_test_split(
        data, test_size=CFG["data"]["test_size"], random_state=SEED
    )

    data_train = BERTDataset(dataset_train, tokenizer, CFG["data"]["max_length"])
    data_valid = BERTDataset(dataset_valid, tokenizer, CFG["data"]["max_length"])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )  # padding이 되어있지 않아도 자동으로 맞춰주는 역할

    return data_train, data_valid, data_collator


def compute_metrics(eval_pred):
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


# 학습
def train(SEED, CFG, model, output_dir, data_train, data_valid, data_collator):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # logging_steps=100,
        # eval_steps=100,
        # save_steps=100,
        save_total_limit=2,
        learning_rate=float(CFG["lr"]),
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=CFG["train_batch_size"],
        per_device_eval_batch_size=CFG["eval_batch_size"],
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


# 평가
def evaluating(model, tokenizer, test_dir, output_dir):
    model.eval()
    preds = []

    dataset_test = pd.read_csv(test_dir)

    for idx, sample in tqdm(
        dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"
    ):
        inputs = tokenizer(sample["text"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test["target"] = preds
    dataset_test.to_csv(os.path.join(output_dir, "output.csv"), index=False)


# config 확인 (print)
# def config_print(config, depth=0):
#     for k, v in config.items():
#         prefix = ["\t" * depth, k, ":"]

#         if type(v) == dict:
#             print(*prefix)
#             config_print(v, depth + 1)
#         else:
#             prefix.append(v)
#             print(*prefix)


if __name__ == "__main__":
    parser = get_parser()
    with open(os.path.join("../config", parser.config)) as f:
        CFG = yaml.safe_load(f)
        # config_print(CFG)

    SEED = CFG["SEED"]
    seed_fix(SEED)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # parser을 사용하여 yaml 가져오기 & parser 입력이 없으면, default yaml을 가져오기
    data_dir = CFG["data"]["train_path"]
    output_dir = CFG["data"]["output_dir"]
    test_dir = CFG["data"]["test_path"]

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7
    ).to(DEVICE)

    data_train, data_valid, data_collator = data_setting(CFG, SEED, data_dir, tokenizer)

    trained_model = train(
        SEED, CFG["train"], model, output_dir, data_train, data_valid, data_collator
    )

    evaluating(trained_model, tokenizer, test_dir, output_dir)
