# Week 03

- [Lecture slides](https://docs.google.com/presentation/d/1jIJHN9T40G5_fPpN2Fp9MOsdSiAcUcy3IHDo05j6VDM)
- [Recording on YouTube (in Russian)](https://youtu.be/ckSlQ4i3Z8U)

### Practice

- Augmentations: [Notebook](./seminar03_1.ipynb)
- CTC Decoding, WER, CER, CTC Beam Search: [Notebook](./seminar03_2.ipynb)


### Additional Materials

- **CTC:**

  - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/), a blog post explaining and visualizing CTC loss.
  - The original CTC Loss paper can be found [here](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

- **Datasets:**

  - [LibriSpeech](https://www.openslr.org/12/) ([paper](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)), ~1000 hours of read English audiobooks.
  - [Common Voice](https://commonvoice.mozilla.org/) ([paper](https://arxiv.org/abs/1912.06670)), crowdsourced read speech in 100+ languages.

- **Encoders:**

  - [Deep Speech 2](https://arxiv.org/abs/1512.02595), an end-to-end CTC model for English and Mandarin.
  - [Conformer](https://arxiv.org/abs/2005.08100), convolutions for local and attention for global dependencies.
  - [Groups, Depthwise, and Depthwise-Separable Convolution](https://www.youtube.com/watch?v=vVaRhZXovbw), a video explaining why depthwise separable convolutions are needed and what their advantages are over regular convolutions.
  - [Fast Conformer](https://arxiv.org/abs/2305.05084), 8x subsampling with depthwise separable convolutions.

- **Language models:**

  - This [tutorial](https://docs.pytorch.org/audio/2.8/tutorials/asr_inference_with_ctc_decoder_tutorial.html) from Torch shows how to use CTC Beam Search with language model support.
  - [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode), CTC beam search with shallow fusion of an n-gram LM.
  - [KenLM](https://github.com/kpu/kenlm), training and querying n-gram LMs.
