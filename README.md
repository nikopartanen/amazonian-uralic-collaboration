# amazonian-uralic-collaboration

This is a web application, running eventually in a Rahti container, that runs an audio processing pipeline and sends an ELAN file into user's email. 

The idea is that one submits a WAV file, and then following steps are performed:

1. Segmentation and speaker diarization using [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.0)
2. Language identification using [MMS](https://huggingface.co/docs/transformers/model_doc/mms)
3. Speech recognition for a language well supported by [MMS](https://huggingface.co/docs/transformers/model_doc/mms)
 - The idea is that the pipeline is used in a low-resource scenario, i.e. while working with endangered Amazonian languages, for which there is no ASR functionality at the moment. There are also situations where one could run ASR for each identified language, although not currently implemented.
4. The result is written into an ELAN file using [pympi](https://github.com/dopefishh/pympi)

This should result in a well segmented ELAN file in which the sections where the local majority language is used are well transcribed, and the researcher can continue working with the segments that cannot be currently recognized. 

Ideally we would be able to offer also fine tuned models for various languages that currently are not supported. 

