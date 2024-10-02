# Week 04

- [Lecture slides](https://docs.google.com/presentation/d/1aa1uS7lm3iuxDM5ZDCBNUaWz1E81zUevxYqZIymxgiA/edit?usp=sharing)
- [Recording on YouTube (in Russian)](TBA)

### Practice & homework

- **Seminar:**
  - CTC and RNN-T Hybrid model: [Notebook](Seminar_04.ipynb)

### Additional Materials

- **General:**
  - depthwise separable convolution explanation with a beautiful visualization [here](https://youtu.be/vVaRhZXovbw?si=4LuymL6WHJJkSMKm) in case of conv2d
  - [comparing end-to-end speech recognition architectures in 2021](https://www.assemblyai.com/blog/a-survey-on-end-to-end-speech-recognition-architectures-in-2021/) a blogpost comparing CTC, LAS and RNN-t models
  - whisper [paper](https://arxiv.org/abs/2212.04356) - a large-scale ASR model with a sophisticated attention-based decoder trained on 680k hours of weakly supervised multilingual and multitask data from openai, released in 2022

- **LAS:**
  - original LAS model [paper](https://arxiv.org/abs/1508.01211)
  - brief [overwiev](https://sh-tsang.medium.com/brief-review-listen-attend-and-spell-a-neural-network-for-large-vocabulary-conversational-106524651804) of the LAS paper from medium
  - see whisper in general section

- **RNN-t:**
  - [sequence-to-sequence learning with transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/) a gentle introduction to RNN-t architecture
  - original RNN-t decoder [paper](https://arxiv.org/pdf/1211.3711)

- **RNN-t optimizations:**
  - fast conformer [paper](https://arxiv.org/abs/2305.05084) - a fast conv2d subsampling with depthwise separable convolutions, 8x time reduction and smaller kernel sizes for convolutions
  - multi-blank transducers [paper](Multi-blank Transducers for Speech Recognition) - add a big blank token in the dictionary and predict it while there is a big pause then we will save computation time
  - token-and-duration transducer [paper](https://arxiv.org/abs/2304.06795) - predict not blank or big blank tokens, but predict all tokens and its duration (nvidia using this tuned decoder in the biggest model - Parakeet-TDT 1.1B)
  - RNN-t with stateless prediction network [paper](https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/) - replace lstm embeddings with embeddings from a simple lookup table (e.g. torch.nn.Embeddings)
  - more about prediction network architectures [here](https://whatsnext.nuance.com/innovation-research/automatic-speech-recognition-on-prediction-network-architecture/)
