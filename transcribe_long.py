from transformers import Wav2Vec2ForCTC
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio
import re, os
import torch
import soundfile as sf
import IPython
from auditok import split
import argparse
from praatio import tgio
from tqdm import tqdm

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    sr, wav = sf.read(fname)
    return wav, sr

def transcribe(wav_file, repo_name):
    tg = tgio.Textgrid()
    entryList = []
    # repo_name = "/home/getalp/leferrae/post_doc/model_w2v/out/out_60/checkpoint-3700"
    model = Wav2Vec2ForCTC.from_pretrained(repo_name).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(repo_name)
    print("signal exctraction")
    sr, signal = read_audio(wav_file)
    dur = (len(signal)/sr)
    if sr==16000:
        print("Computing VAD")
        region = split(wav_file, eth=70, aw=0.01)
        print(len(signal))

        for i, r in enumerate(tqdm(region)):

            seg = signal[int(r.meta.start*sr):int(r.meta.end*sr)]
            data = [{"input_values" : seg, "input_lenght" : len(seg)}]
            input_dict = processor(data[0]["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
            logits = model(input_dict.input_values.to("cuda")).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            transc = processor.decode(pred_ids)
            interval = tgio.Interval(r.meta.start, r.meta.end, transc)
            entryList.append(interval)

        print("done")
        text = tgio.IntervalTier("Text", entryList, minT=0, maxT=dur)
        tg.addTier(tier=text)
        tg.save(wav_file.replace(".wav", ".TextGrid"))

    else:
        print("Hey hun! Your sample rate is off. sr={}".format(sr))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wav", type=str, default="smp") # wav file path
    parser.add_argument("--model", type=str, default="model/checkpoint-3700") # wav file path
    
    args = parser.parse_args()
    wav_file = args.wav
    model_path = args.model
    transcribe(wav_file, model_path)
if __name__ == "__main__":
    main()