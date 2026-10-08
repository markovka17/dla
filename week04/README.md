# Week 04

- [Lecture slides](https://docs.google.com/presentation/d/1t3GRbg88p6FKHTr4B-MqYTigalEJiMBKt1p1MJJJvY0)
- [Recording on YouTube (in Russian)](https://youtu.be/liboVD4FFvk)

### Practice

- Whisper: greedy decoding, prompting, alignment in cross-attention, language forcing: [Notebook](./seminar04_whisper.ipynb)

### Additional Materials

- **LAS / AED:**

  - original LAS [paper](https://arxiv.org/abs/1508.01211)
  - [Joint CTC-Attention](https://arxiv.org/abs/1609.06773), CTC as an auxiliary loss for an attention-based encoder-decoder.
  - [An analysis of incorporating an external language model into a sequence-to-sequence model](https://arxiv.org/abs/1712.01996), shallow fusion across LM types, decoding units and tasks.

- **Whisper:**

  - Whisper [paper](https://arxiv.org/abs/2212.04356): weakly supervised training on 680k hours, the multitask token
    format and zero-shot robustness.
  - [Whisper prompting guide](https://cookbook.openai.com/examples/whisper_prompting_guide) from the OpenAI cookbook.
  - [Word-level timestamps](https://github.com/openai/whisper/blob/main/whisper/timing.py) in the original implementation: DTW over the cross-attention of `alignment_heads`.

- **RNN-T:**

  - original RNN-T [paper](https://arxiv.org/pdf/1211.3711)
  - [Sequence-to-sequence learning with Transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/), a gentle introduction: the alignment lattice, the loss and greedy decoding.
  - [`torchaudio.functional.rnnt_loss`](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.rnnt_loss.html)
  - GigaAM-v3 RNN-T [huggingface model](https://huggingface.co/ai-sage/GigaAM-v3)

- **Streaming:**

  - [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/abs/1811.06621), RNN-T on Google Pixel.
  - [Stateful Conformer with Cache-based Inference](https://arxiv.org/abs/2312.17279), FastConformer with limited context and activation caching.

- **Decoder-only:**

  - [Seed-ASR](https://arxiv.org/abs/2407.04675), continuous speech representations and context fed into an LLM.
  - [Qwen3-ASR](https://arxiv.org/abs/2601.21337), ASR for 52 languages and dialects on top of Qwen3-Omni.
  - [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
