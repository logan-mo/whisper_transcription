from datasets import Dataset, load_dataset, Audio, DatasetDict, load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import pandas as pd
import evaluate
import librosa
import torch
import json
import os


processor = AutoProcessor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
)
model = AutoModelForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"]
    )

    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch


dataset = load_dataset("csv", data_files="final_dataset_linux.csv", split="train[:100]")
dataset = dataset.rename_columns({"path": "audio", "transcription": "text"})
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
dataset = dataset.map(prepare_dataset).remove_columns(["audio", "text"])

train_test_dataset = dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid
test_valid = train_test_dataset["test"].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict(
    {
        "train": train_test_dataset["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"],
    }
)

if not os.path.exists("processed_dataset"):
    os.mkdir("processed_dataset")

train_test_valid_dataset.save_to_disk("processed_dataset/")

dataset = load_from_disk("processed_dataset/")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = pred_str
    label_str_norm = label_str
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i]
        for i in range(len(pred_str_norm))
        if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./saved_model",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=4000,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


trainer.train()

if not os.path.exists("final_product"):
    os.mkdir("final_product")
trainer.save_model("final_product/")
