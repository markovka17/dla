# Deep Learning for Audio (DLA)
- Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
- Any technical issues, ideas, bugs in course materials, contribution ideas - add an issue
- The current version of the course is conducted in autumn 2020 at the [CS Faculty](https://cs.hse.ru/en/) of [HSE](https://www.hse.ru/en/)

# Syllabus

- [__week01__](./week01) Introduction to Digital Signal Processing
  - Lecture: Signals, Fourier transform, Spectrograms, MFCC and etc
  - Seminar: Intro in PyTorch, DevOps, R&D in Deep Learning
  
- [__week02__](./week02) Automatic Speech Recognition I
  - Lecture: Metrics, Attention, LAS, CTC, BeamSearch
  - Seminar: Docker, W&B, Augmentations for Audio

- [__week03__](./week03) Automatic Speech Recognition II
  - Lecture: LM Fusing, RNN Transducer, Schedule Sampling, BPE
  - Seminar: Jasper, QurtzNet, Mixed Precision Training, DDP/DP
  
- [__week04__](./week04) Key-word spottind (KWS) and Voice Activity Detection (VAD)
  - Lecture: (DNN, CNN, RNN+Attention) based KWS, SVDF, Orthogonality Regularization and other Tricks
  - Seminar: Speeding Up NNs: Tensor Decomposition, Quantization, Pruning, Distilation and Architecture Design
  
- [__week05__](./week05) Speaker verification and identification
  - Lecture: Metric Learning: Cosine, Contrastive, Triplet Losses. Angular Softmax. ArcFace
  - Seminar: Generalized End2End Loss for Speaker Verification

- [__week06__](./week06) Text to Speech
  - Lecture: Tacotron, DeepVoice, GST, FastSpeech, Attention Tricks
  - Seminar: Location-Sensitive Attention

- [__week07__](./week08) Neural Vocoders
  - Lecture: Introduction into generative models: AR, GAN, NF. WaveNet, ParallelWaveNet, WaveGlow, WaveFlow, MelGAN, PWG.

- [__week08__](./week07) Voice Conversion
  - Lecture: AutoVC, ConVoice, TTS Skins, StarGAN-VC-1-2, CycleGAN-1-2-3, Blow

- [__week09__](./week09) Music Generation
  - Lecture: VQVAE, Sparse Transformer, MuseNet, JukeBox

- [__week10__](./week10) Speech Enhancement, Denoising and Speaker Diarization
  - Lecture: SEGAN, TF Masking, HiFi Denoising, Speaker Diarization, VAD

- [__week11__](./week11) Self-supervision in Audio and Speech
  - Lecture: Intro to SS Learning. InfoNCE, CPC

# Homeworks
- [__DSP__](./week01/homework.ipynb)
  Implementation of basic ops like FFT, Spectrogram and MelScale
  
- [__ASR__](./week03/homework.ipynb)
  Implementation of small ASR model, beam search and LM fusing
  
- [__KWS__](./week04/homework.ipynb)
  Implementation of attention based KWS model, streaming scoring and model distillation
  
- [__TTS__](./week06/homework.ipynb)
  Implementation of TTS model with different tricks

- [__NV__](./week07/homework.ipynb)
  Implementation of Neural Vocoder Model
  
# Contributors & course staff
Course materials and teaching performed by
- [Alexander Markovich](https://t.me/markovka17)
- [Daniil Ivanov](https://t.me/the_longest_id_in_the_world)
