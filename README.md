# Transcribe_w2v_CTC
## Requirements
pip install datasets==1.18.3

pip install transformers==4.11.3

pip install huggingface_hub==0.1

pip install torchaudio

pip install librosa

pip install jiwer

pip install praatio==4

pip install auditok

## Usage
Works on wav files with sample rate=16000 and 1 channel 

Works with python 3.7 or 3.8 

Run: python transcribe_long.py --wav /paht/to/wav --model /path/to/model

I put a model. It is based on the Benchmark and has been finetuned on 60min of French based creoles.

