# Homework 3 (TTS)

### Task
Implement [FastSpeech](https://arxiv.org/pdf/1905.09263.pdf).

This homework is aimed at implementing a TTS model based on FastSpeech.
Extraction of letter lengths in this case will be based on Wav2Vec2 and [CTC-Segmentation](https://arxiv.org/pdf/2007.09127.pdf).
The model will be trained on the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

The nearby notebook `aligner.ipynb` has many useful things:
* Featurizer
* Dataset and Collator
* Pretrained Vocoder for validation
* Graphem Aligner

Important points(!):
* You are prohibited from using `self-attention` from `pytorch`. So, please, implement it by yourself :)
* Read `FastSpeech` carefully. There are enough nuances in this work. For, example:
  * Prediction of `log`(duration)
  * `Conv1D` instead of `Linear` in `Transormer` block
  * Length Regulator
  * Pre LayerNorm in `Transormer` for to fast convergence
* If you are unfamiliar with the Transformers, read the following papers:
  * [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
  * [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
  * [Transformers without Tears](https://tnq177.github.io/data/transformers_without_tears.pdf)
  * [Transformer Details Not Described in The Paper](https://tunz.kr/post/4)
  * [Training Tips for the Transformer Model](https://arxiv.org/pdf/1804.00247.pdf)


--------------
### Mandatory requirements
In general, the format of the current homework follows that of the first homework assignment (ASR).
You must organize the repository as in the first homework assignment, which is to break up the code into modules and follow the code style.

So, we don't accept homework if any of the following requirements are not satisfied:
* The code should be situated in a public github (or gitlab) repository
* All the necessary packages should be mentioned in ./requirements.txt or in an installation guide section of README.md
* You must use W&B for logging losses and synthesed audio. 
* All necessary resources (such as model checkpoints) should be downloadable with a script
  Mention the script (or lines of code) in the `README.md`
* Attech a report. That includes:
  * Description and result of each experiment
  * How to reproduce your model?
  * Attach training logs to show how fast did you network train
  * What worked and what didn't work?
  * What were the major challenges?

--------------
### Grade
```
grade = 0.5 * MOS + 0.5 * (quality of code and report)
```

To evaluate the MOS, you must add a synthesis of the following sentences to the report:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`
