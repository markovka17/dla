# Week 05

- [Lecture slides](https://docs.google.com/presentation/d/12KvRQG5y3FLVM-2O1NQ4qraKTx8KmjNGJHqvoJuiKNI)
- [Recording on YouTube (in Russian)](https://www.youtube.com/watch?v=o-QGlcDRWRw)

### Additional Materials

- **wav2vec 2.0:**

  - [wav2vec 2.0](https://arxiv.org/abs/2006.11477), masked speech representations, contrastive learning and a jointly learned quantizer.
  - [XLS-R](https://arxiv.org/abs/2111.09296), scaling wav2vec 2.0 to multilingual speech representation learning.

- **HuBERT:**

  - [HuBERT](https://arxiv.org/abs/2106.07447), masked prediction of offline cluster assignments with iterative re-clustering.

- **BEST-RQ:**

  - [Self-supervised Learning with Random-projection Quantizer for Speech Recognition](https://arxiv.org/abs/2202.01855), masked prediction with a fixed random projection and codebook.
  - [Google USM](https://arxiv.org/abs/2303.01037), large-scale multilingual pretraining with random-projection quantization.

- **GigaAM:**

  - [GigaAM: Efficient Self-Supervised Learner for Speech Recognition](https://arxiv.org/abs/2506.01192), HuBERT-CTC pretraining with targets derived from a speech recognition model.
  - GigaAM [repository](https://github.com/salute-developers/GigaAM) and GigaAM-v3 [Hugging Face models](https://huggingface.co/ai-sage/GigaAM-v3): SSL, CTC and RNN-T models for Russian speech.

- **LLM-based ASR:**

  - [Qwen3-ASR](https://arxiv.org/abs/2601.21337), speech recognition and language identification built on Qwen3-Omni.

- **Delayed Streams Modeling:**

  - [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753), a decoder-only model over time-aligned audio and text streams with an explicit delay.
  - [Voxtral Realtime](https://arxiv.org/abs/2602.11298), streaming ASR with a causal audio encoder, pretrained LLM and configurable transcription delay.

- **Encoder-free models:**

  - [LLM can Read Spectrogram](https://arxiv.org/abs/2606.10231), Mel-LLM: feeding Mel-spectrogram features into an LLM without a dedicated speech encoder.
  - [dMel](https://arxiv.org/abs/2407.15835), speech tokenization by discretizing Mel-filterbank channels into intensity bins.
