import random
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import re, os
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from datasets import load_metric
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import soundfile as sf
import argparse

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    return wav, sr


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    return batch



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def get_data(path):

    data = []
    for elt in os.listdir(path):
        if ".txt" in elt:
            entry = {}
            with open(path+elt, mode="r", encoding="utf-8") as tfile:
                sentence = tfile.read()
            entry["sentence"] = sentence.replace("\n", "")
            wav_name = elt.replace(".txt", ".wav")
            w, sr = read_audio(path+wav_name)
            entry["audio"]={"sampling_rate" : sr, "path" : wav_name, "array" : w}
            entry["path"]= path+wav_name
            data.append(entry)
    return data

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def train(data_path):

    train_data = get_data(data_path+"train/")
    dev_data = get_data(data_path+"dev/")

    vocab_train = set(y for x in train_data for y in x["sentence"])
    vocab_test = set(y for x in dev_data for y in x["sentence"])
    vocab = vocab_train.union(vocab_test)
    if "\n" in vocab:
        vocab.remove("\n")
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    ###Creation of the tokeniser###

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    repo_name = "wav2vec2-large-xls-r-300m-mb-colab"
    tokenizer.save_pretrained(repo_name)
    ###Extraction of speech features###

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    #### Setting up data for training###

    # common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    # common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")
    train_data = list(map(prepare_dataset, train_data))
    dev_data = list(map(prepare_dataset, dev_data))

    #### Training ####
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    from transformers import TrainingArguments

    training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="smp") # wav file path
    
    args = parser.parse_args()
    data_path = args.data_path

    train(data_path)

if __name__ == "__main__":
    main()