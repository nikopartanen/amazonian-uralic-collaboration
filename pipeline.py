# This code is written by Niko Partanen, but primarily reusing different tutorials online.
# For licenses, see the licenses of the packages used here, to my knowledge this doesn't
# replicate as such any complete tutorial but is put together from different snippets
# and through experimenting what works

from pyannote.audio import Pipeline
from pathlib import Path
import torch
import torchaudio
from mikatools import *
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, AutoProcessor
import librosa
import json
import os
import re
import pympi

TOKEN = "" # This needs to be added from HuggingFace 
audio_file = "" # Here goes the path to the audio file
language_code = "por" # Any larger language works here

# Here we load the pipeline for speaker diarization

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",
                                    use_auth_token=TOKEN)

# This is needed to use the GPU

pipeline.to(torch.device("cuda"))

file = Path(audio_file)

# We read the audio file and send it to the diarization pipeline

waveform, sample_rate = torchaudio.load(str(file))
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# We create a directory under temp directory where we keep the files, so it is
# easier to monitor where things break
# The diarization is written into RTTM file, which is one format pyannote uses

Path(f"temp/{file.stem}").mkdir(exist_ok=True)

rttm = f"temp/{file.stem}/{file.with_suffix('.rttm')}"

with open(rttm, 'w') as f:
    diarization.write_rttm(f)

print("Finished diarization")

# Then we read it from the diarization file, not necessary, 
# but doesn't harm to have that file

rttm_file = open(rttm, "r")

lines = rttm_file.readlines()

audio = AudioSegment.from_wav(audio_file)

segments = []

for n, line in enumerate(lines):
    
    i = line.split()

    start = int(float(i[3]) * 1000)
    end = int(float(i[3]) * 1000) + int(float(i[4]) * 1000)

    duration = end - start

    speaker = i[7]

    # Here we throw away the shortest segments as they didn't seem to be
    # very correct or necessary, but this may be need to be adjusted, for
    # example, if there is a new version of pyannote. 
    # Anyway, here we save the segments into their own little WAV files,
    # which may be unnecessary but who cares
  
    if duration > 150:

        segment = audio[start:end]

        segment_file = f"temp/{file.stem}/{file.stem}_{str(n).zfill(4)}_{speaker}_{start}_{end}.wav"

        segment.export(segment_file, format="wav")

        segments.append(segment_file)

print("Starting language detection")

# This doesn't work now, it puts the cache files into a wrong place that
# has to be deleted every now and then

os.environ['TRANSFORMERS_CACHE'] = './cache'

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

# Here we define the language identification pipeline

print("Starting to define LID pipeline")

model_id_lid = "facebook/mms-lid-126"

model_lid = Wav2Vec2ForSequenceClassification.from_pretrained(model_id_lid)
processor_lid = AutoFeatureExtractor.from_pretrained(model_id_lid)

pipe_lid = pipeline(task = "audio-classification", model=model_lid, feature_extractor = processor_lid)

# And here it is done for ASR
# We do it now just for one language, I think one could also do it
# for each detected language, but this seems like loading lots of 
# language models and I don't think the result usually is what we want.
# As we have lots of small endangered languages, which are never
# recognized correctly, then we don't probably want sprurious 
# transcriptions in random languages

print("Starting to define ASR pipe")

model_id_asr = "facebook/mms-1b-all"
processor_asr = AutoProcessor.from_pretrained(model_id_asr)
model_asr = Wav2Vec2ForCTC.from_pretrained(model_id_asr)

# Keep the same model in memory and simply switch out the language adapters by calling load_adapter() for the model and set_target_lang() for the tokenizer
processor_asr.tokenizer.set_target_lang(language_code)
model_asr.load_adapter(language_code)

print("Starting to process segments")

# Here we do the language identification for each audio chunk

for segment in segments:

    language_ids = pipe_lid(str(segment))

    json_dump(language_ids, Path(segment).with_suffix(".json"))

    # And only if the language matches what we wanted we do ASR
  
    if language_ids[0]['label'] == language_code:

        speech, sample_rate = librosa.load(segment)

        inputs = processor_asr(speech, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = model_asr(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor_asr.decode(ids)

        transcription_file = open(Path(segment).with_suffix(".txt"), "w")

        transcription_file.write(transcription)
        
        transcription_file.close()

# In the end we save it all into an ELAN file

print("Starting to create ELAN file")

files = []

for f in Path(f"temp/{file.stem}/").glob("*wav"):

    files.append(f)

files = sorted(files)

pattern = r"(SPEAKER_\d+)_(\d+)_(\d+)"

elan_data = []

for f in files:
    
    # Use re.search to find the pattern in the input string
    match = re.search(pattern, f.stem)

    speaker = match.group(1)  # Whole matched string
    start = match.group(2)  # First number
    end = match.group(3)  # Second number
    
    text = ''

    text_file = Path(f).with_suffix('.txt')

    if text_file.exists():
        
        text = open(text_file, 'r').read()
    
    elan_data.append((speaker, start, end, text))
    
participants = set()

for i in elan_data:
    
    participants.add(i[0])

participants = list(participants)

elan_file = pympi.Elan.Eaf(file_path = None)
elan_file.add_linguistic_type(lingtype='textT', timealignable=True, graphicreferences=False)
elan_file.add_linked_file(file_path = audio_file, mimetype = 'audio/x-wav')

for participant in participants:

    elan_file.add_tier(tier_id=f'text@{participant}', ling='textT')

for i in elan_data:

    elan_file.add_annotation(id_tier = f"text@{i[0]}", start = int(i[1]), end = int(i[2]), value = i[3])


# This tier is created automatically, so we remove it
elan_file.remove_tier(id_tier='default')

elan_file.to_file(file_path = f"{Path(audio_file).with_suffix('.eaf')}")

# Now, for the complete application, this should be sent to the user's email
