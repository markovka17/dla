![logo5v1](https://user-images.githubusercontent.com/20357655/104316876-2be04600-54ee-11eb-93ed-f9835fde1527.jpg)

# Deep Learning for Audio (DLA)
- Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
- Any technical issues, ideas, bugs in course materials, contribution ideas - add an issue
- The current version of the course is conducted in autumn 2021 at the [CS Faculty](https://cs.hse.ru/en/) of [HSE](https://www.hse.ru/en/)

# Syllabus

- [__week01__](./week01) Introduction to Digital Signal Processing
  - Lecture: Introduction to Course
  - Seminar: Intro in `pytorch`

- [__week02__](./week02) Introduction to Digital Signal Processing
  - Lecture: Signals, Fourier transform, Spectrograms, MelScale, MFCC and etc
  - Seminar: `torchaudio`, spoken digit classification

- [__week03__](./week03) Automatic Speech Recognition (ASR) I
  - Lecture: Metrics, Attention, CTC, LAS, BeamSearch
  - Seminar: Audio augmentations, CTC decoding, CTC BeamSearch
  
- [__week04__](./week04) Automatic Speech Recognition (ASR) II
  - Lecture: RNN-T, LM-fusion, BPE
  - Seminar: W&B tutorial, homework barebones overview
  
- [__week05__](./week05) Speaker verification and identification
  - Lecture: Metric Learning: Cosine, Contrastive, Triplet Losses. Angular Softmax. ArcFace
  - Seminar: ---

- [__week06__](./week06) Key-word spottind (KWS)
  - Lecture: (DNN, CNN, RNN+Attention) based KWS, SVDF, Orthogonality Regularization and other Tricks
  - Seminar: Implementation of CNN+Attention+RNN KWS model

- [__week07__](./week07) Text to Speech (TTS)
  - Lecture: Tacotron, DeepVoice, GST, FastSpeech, AdaSpeech, Attention Tricks
  - Seminar: TTS in `torchaudio`

- [__week08__](./week08) Neural Vocoders
  - Lecture: WaveNet, Parallel WaveGAN
  - Seminar: Implementation of WaveNet

- [__week09__](./week09) Advanced TTS and Vocoders
<!--   - Lecture: Introduction into generative models: GAN, NF -->

- [__week10__](./week10) Voice Conversion

- [__week11__](./week11) Music Generation

- [__week12__](./week12) Speech Enhancement, Denoising

- [__week13__](./week13) Self-supervision in Audio and Speech



<!-- - [__week01__](./week01) Introduction to Digital Signal Processing
  - Lecture: Signals, Fourier transform, Spectrograms, MFCC and etc
  - Seminar: Intro in PyTorch, DevOps, R&D in Deep Learning
  
- [__week02__](./week02) Automatic Speech Recognition I
  - Lecture: Metrics, Attention, LAS, CTC, BeamSearch
  - Seminar: Docker, W&B, Augmentations for Audio

- [__week03__](./week03) Automatic Speech Recognition II
  - Lecture: LM Fusing, RNN Transducer, Schedule Sampling, BPE
  - Seminar: Jasper, QurtzNet, Mixed Precision Training, DDP/DP
  
- [__week05__](./week05) Speaker verification and identification
  - Lecture: Metric Learning: Cosine, Contrastive, Triplet Losses. Angular Softmax. ArcFace
  - Seminar: Generalized End2End Loss for Speaker Verification

- [__week06__](./week06) Text to Speech
  - Lecture: Tacotron, DeepVoice, GST, FastSpeech, Attention Tricks
  - Seminar: Location-Sensitive Attention

- [__week07__](./week07) Neural Vocoders
  - Lecture: Introduction into generative models: AR, GAN, NF. WaveNet, ParallelWaveNet, WaveGlow, WaveFlow, MelGAN, PWG.

- [__week08__](./week08) Voice Conversion
  - Lecture: AutoVC, ConVoice, TTS Skins, StarGAN-VC-1-2, CycleGAN-1-2-3, Blow

- [__week09__](./week09) Music Generation
  - Lecture: VQVAE, Sparse Transformer, MuseNet, JukeBox

- [__week10__](./week10) Speech Enhancement, Denoising and Speaker Diarization
  - Lecture: SEGAN, TF Masking, HiFi Denoising, Speaker Diarization, VAD

- [__week11__](./week11) Self-supervision in Audio and Speech
  - Lecture: Intro to SS Learning. InfoNCE, CPC
 -->
# Homeworks
<!-- - [__DSP__](./week01/homework.ipynb)
  Implementation of basic ops like FFT, Spectrogram and MelScale
   -->
- [__ASR__](./hw1_asr)
  Implementation of ASR model

- [__KWS__](./hw2_kws)
  Implementation of KWS model
  
- [__TTS__](./hw3_tts)
  Implementation of TTS model
  
- [__NV__](./hw4_nv)
  Implementation of Neural Vocoder Model
  
# Contributors & course staff
Course materials and teaching performed by
- [Alexander Markovich](https://t.me/markovka17)
- [Daniil Ivanov](https://t.me/the_longest_id_in_the_world)
