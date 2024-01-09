![logo5v1](https://user-images.githubusercontent.com/20357655/104316876-2be04600-54ee-11eb-93ed-f9835fde1527.jpg)

# Deep Learning for Audio (DLA)

- Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
- Any technical issues, ideas, bugs in course materials, contribution ideas - add an issue
- The current version of the course is conducted in autumn 2023 at the [CS Faculty](https://cs.hse.ru/en/)
  of [HSE](https://www.hse.ru/en/)

# Syllabus

- [__week01__](./week01) Introduction to Course
    - Lecture: Introduction to Course
    - Seminar: Intro in `pytorch`
 
- [__week02__](./week02) Introduction to Digital Signal Processing
    - Lecture: Signals, Fourier Transform, spectrograms, MelScale, MFCC
    - Seminar: DSP in practice, spectrogram creation, training a model for audio MNIST
      
- [__week03__](./week03) Speech Recognition I
    - Lecture: Metrics, datasets, Connectionist Temporal Classification (CTC), Listen Attend and Spell (LAS), Beam Search
    - Seminar: Audio Augmentations, Beam Search, Homework discussion


- [__week04__](./week04) Speech Recognition II
    - Lecture: RNN-T, language model fusion, Byte-Pair Encoding (BPE)
    - Seminar: --
    
- [__week05__](./week05) Source Separation I
  - Lecture: A review of general Source Separation and Denoising, Encoder-Decoder-Separator architectures, Demucs family, DCCRN, FullSubNet+
  - Seminar: Metrics, Dataset of Mixtures and some tech stuff
  
- [__week06__](./week06) Source Separation II
  - Lecture: Speech separation, Blind and Target Separation, Recurrent(TasNet, DPRNN, VoiceFilter) and CNN(ConvTasNet, SpEx+)
  - Seminar: WienerFilter, SincFilter and DEMUCS
  
- [__week07__](./week07) Text to Speech (TTS)
  - Lecture: Tacotron, DeepVoice, GST, FastSpeech, AdaSpeech, Attention Tricks
  - Seminar: FastSpeech I
  
- [__week08__](./week08) Neural Vocoders
  - Lecture: WaveNet, Parallel WaveGAN, WaveGlow, MelGAN, HiFiGAN
  - Seminar: WaveNet


- [__week09__](./week09) Voice Conversion
  - Lecture: Disentanglement & Direct based methods
  - Seminar: TorchScript, HiFi-VC
  
- [__week10__](./week10) Voice Biometry I
  - Lecture: Introduction. CMs for sythesized speech detection (LCNN, RawNet2, AASIST). GNNs
  - Seminar: ASVspoof, Sinc-layer, GNN

- [__week11__](./week11) Voice Biometry II
  - Lecture: CMs for replay attack detection. ASV systems. SASV systems. Streaming
  - Seminar: -

- [__week12__](./week12) Diffusion Models for Audio Generation

  - Lecture, part 1: Introduction to diffusion models from two perspectives: score matching and latent probabilistic models. 
  - Lecture, part2: Diffusion models for audio synthesis and tts. WaveGrad, DiffWave, GradTTS

- __bonus week__ Guest lecture

  - Self-Supervised models in ASR

<!--
- [__week13__](./week13) Music Generation

-->

# Homeworks
- [__ASR__](./hw1_asr) Training speech recognition model
- [__SS__](./hw2_ss) Training speech separation model
- [__TTS__](./hw3_tts) Implementation of TTS model (Part 1, FastSpeech)
- [__NV__](./hw4_nv) Implementation of TTS model (Part 2, Vocoder)
- [__AS__](./hw5_as) Implementation of Anti-spoofing Model

# Resources

* [Lecture recordings on YouTube (in russian)](https://youtube.com/playlist?list=PLYG3WHDP5CWV_DRs9SZ8YiA3agJCX1sIr&si=7eegv6XO75ZCF2Hc)

# Contributors & course staff

Course materials and teaching (in different years) were delivered by:
- [Maxim Kaledin](https://t.me/XuMuK_MK)
- [Alexander Markovich](https://t.me/markovka17)
- [Petr Grinberg](https://t.me/Blinorot)
- [Aibek Alanov](https://t.me/aibrain)
- [Grigory Fedorov](https://t.me/fedorovgv)
- [Ilya Lewin](https://t.me/levensons)
- [Daniil Ivanov(previously)](https://t.me/the_longest_id_in_the_world)
- [Timofey Smirnov(previously)](https://t.me/timothyxp)
- [Alexander Mamaev(previously)](https://t.me/alxmamaev)

