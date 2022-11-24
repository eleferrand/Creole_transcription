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

### To use an already trained model and make inferences
  Run: python transcribe_long.py --wav /paht/to/wav --model /path/to/model
  
  I could not  upload the model used here as it is too heavy for a git repo (~3Gb)
  
### To fine-tuned an xlsr model

  Run python train.py --data_path /path/to/data
  
  The script will download the xlsr-300 model from huggingface and finetune it with your data
  
  The data should be built as follow:
  
   --------------------Whatever corpus name
   
  |
  
  -------------train
  
  |
  
  ---name_1.wav
  
  ---name_1.txt
  
  ---name_2.wav
  
  ---name_2.txt
  
  ...
  
  |
  
  -------------dev
  
  |
  
  --- name_1.wav
  
  --- name_1.txt
  
  --- name_2.wav
  
  --- name_2.txt
  
  ....
