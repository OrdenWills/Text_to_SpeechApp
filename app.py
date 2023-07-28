from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset


import flask
from flask import Flask
from flask import render_template,redirect,url_for,request

def fetch_pretrainedModel():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor,model,vocoder
def fetch_VoiceDataset():
    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return embeddings_dataset

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any-secret-key-you-choose'

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/submit',methods=['GET','POST'])
def submit():
    # get the model
    processor,model,vocoder = fetch_pretrainedModel()
    embeddings_dataset = fetch_VoiceDataset()

    # speaker dict
    speakers = {
    'jean': 7306,
    'matt': 0,
    'john': 1138,
    'ronica': 2271,
    'bill' : 3403,
    'sack':4535,
    'adol':5667,
    'oli':6799,}

    if request.method == 'POST':
        text = request.form['text']
        speaker = request.form['speaker'].lower()
        print(speaker)
        print(text)

    speaker_embeddings = torch.tensor(embeddings_dataset[speakers[speaker]]["xvector"]).unsqueeze(0)
    print('pass speaker')

    inputs = processor(text=text,return_tensors='pt')

    # # generate speech with the model
    speech = model.generate_speech(\
        inputs['input_ids'],
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder)
    sf.write(f'./static/sound/t2speech.mp3',speech.cpu(),samplerate=16_000)
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app = app.run()