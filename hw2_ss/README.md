# Homework 2 (Speaker Separation)

## Task
Implement one of the following solutions
[SpEx+](https://www.isca-speech.org/archive/interspeech_2020/ge20_interspeech.html),
[VoiceFilter](https://arxiv.org/abs/1810.04826).

These are one of the basic and yet powerful solutions of target speaker separation problem clearly illustrating the pipeline of target source separation and able to be trained with limited resources. The original training and validation data was obtained using WSJ dataset but similar results you could obtain on LibriSpeech (1-2 dB lower in SDR metrics).

To do that you are to implement the solution in your existing pipeline, including not only the model but also the mix generation(see the seminar 5 for the details). 

    
* Pay attention to the following:
  * Be careful with audio normalization and use the same normalization procedures in all your experiments to get comparable results
  * Zero padding versus cutting to minimum length when constructing a batch of audios
  * Silence removal is not allowed on validation
  * How to compute inference in the presence of classification module in SpEx+?..
  * You will need to implement your own TCN Block as a pytorch nn.Module, the details are in the paper
  * SDR is not necessarily the best thing reflecting the sound quality, always listen the sound with your ears and check the spectrograms with your own eys to make sure that the model does not go crazy on loudness or digital artefacts.

Note that you **should** describe all your steps in detail. In must be clearly stated in the report how the model is set (the architecture, inference, validation and training scenario), what are the pre-processing and post-processing steps taken, how the mixes are constructed (with all the pre- and post-processing details) and how the dataset is constructed, especially if you implement additional classification module for SpEx+.

Important: you must provide the code and parameters you used for mixing (script and the actual Mixer classes), otherwise we will not be able to grade your solution.

--------------
## Mandatory requirements
In general, the format of the current homework follows that of the first homework assignment (ASR).
You must organize the repository as in the first homework assignment, which is to break up the code into modules and follow the code style.

So, we don't accept homework if any of the following requirements are not satisfied:
* The code should be situated in a public github (or gitlab) repository (private until the deadline)
* All the necessary packages should be mentioned in ./requirements.txt or in an installation guide section of README.md
* You must use W&B for logging losses and synthesed audio. 
* All necessary resources (such as model checkpoints) should be downloadable with a script
  Mention the script (or lines of code) in the `README.md`
* Attach a report. That includes:
  * Description and result of each experiment
  * How to reproduce your model?
  * Attach training logs to show how fast did you network train
  * What worked and what didn't work?
  * What were the major challenges?
  
--------------
## Grade

```
grade = 0.7 * SiSDRGrade + 0.3 * (quality of code and report) + bonus
```

SISDRGrade is computed with respect to the table on our own loudness-normalized to -20LUFS SNR0 mix of similar speech dataset NEW!!!:
| SI-SDR      | PESQ        | Grade |
| ----------- | ----------- |-------|
| <5          | <1.1          |  0    |
| >5          | >1.1        |  3    |
| >7          | >1.2        |  4    |
| >8.5        | >1.2        |  5    |
| >10.5       | >1.5        |  6    |
| >11.5         | >1.8        |  7    |
| >12.5         | >1.8        |  8    |

NEW!!! To measure SI-SDR, use [torchmetrics.audio.SI-SDR](https://torchmetrics.readthedocs.io/en/stable/audio/scale_invariant_signal_distortion_ratio.html) and [torchmetrics.audio.pesq](https://torchmetrics.readthedocs.io/en/stable/audio/perceptual_evaluation_speech_quality.html) with "wb" and 16kHz sampling rate for PESQ.

Similarly to ASR, your test.py must have an argument -t that replaces dataset in config to CustomDirDataset class from your src/datasets/. Previously, this class read two dirs (audio, transcriptions) with Utterance_id.wav and Utterance_id.txt. Now, your class should work with 3 dirs: mix (for mixes), refs (for references) and targets (for targets). Files in each dir are named in the following way ID-mixed.wav, ID-ref.wav, ID-target.wav for mix, ref and target respectively.

Insight1: With original VoiceFilter you could get SI-SDR 7-9 at most and it requires a lot of time to train. There are technical tricks to discover. 

Insight2: On the other hand, SpEx+ would allow you to get up to SI-SDR of 11.7 but just SI-SDR is not enough: you need to make sure your model gets the rid of the sound artefacts. Just few more training epochs should do the trick. 

Insight3: In SpEx+ to go beyound SI-SDR of 11.7 you surely will have to implement classification head and modify your dataset structures so that you could know the id of the target speaker during the training. 

Insight4: Larger batch may be needed for SpEx+ to train smoothly (4-5 of 3sec audio triplets should be enough but see how to exploit your computational infrastructure to allow it in terms of memory).

Bonuses:
+0.7 If SpEx+ is implemented and you used the classification head 

+0.3 for comparison of Mel- and native-scale spectrograms in VoiceFilter

+0.5 for implementing SI-SNR loss in VoiceFilter

+0.5 For providing validation results on the noised dataset of 1000 mixes (mixing WHAM and WHAMR! tracks with the two speakers from LibriSpeech).

+0.5 For trying to use another reference encoder (for instance, wav2vec with modifications) and providing comparison with your original baseline model

+0.5 for measuring the quality of your model in the case of audio stream with segmentation, aggregation and separate chunk procesing.

+1 for measuring WER and CER using one of your ASR ready solutions together with Speech Separation and comparing it to just ASR in case of your test data.

+yourScoreForHydra (if not used before)
