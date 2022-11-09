# Homework 3 (TTS)

## Task
Implement [FastSpeech2](https://arxiv.org/pdf/2006.04558.pdf).

This homework is aimed at implementing a TTS model based on FastSpeech2.
Extraction of letter lengths in this case will be based on Tacotron2.
The model will be trained on the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

In seminar notebook from week07 you can find FastSpeech1 implementation with some misses.
You can use this implementation as a basis and implement FastSpeech2 based on it.

Important points(!):
* You are prohibited from using `self-attention` or `Transformer` from `pytorch`.
   You can use implementation from seminar, but remember that it is no-efficient and use Post-LN Transformer.
  A more effective implementation can be inspired [here](https://github.com/karpathy/minGPT ) (if you want to wait 10 hours instead of 20)
  
* 
  If you are unfamiliar with the Transformers, read the following papers:
    * [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    * [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
    * [Transformers without Tears](https://tnq177.github.io/data/transformers_without_tears.pdf)
    * [Transformer Details Not Described in The Paper](https://tunz.kr/post/4)
    * [Training Tips for the Transformer Model](https://arxiv.org/pdf/1804.00247.pdf)
    
* Read `FastSpeech` and `FastSpeech2` carefully. There are enough nuances in this work. For, example:
  * Prediction of `log`(duration)
  * `Conv1D` instead of `Linear` in `Transormer` block
  * Length Regulator
  * Pre LayerNorm in `Transormer` for to fast convergence
  * Pitch and Energy normalization

Note that since your work is essentially a reimplementation of the paper, you **should** describe all your steps in detail. For example, how you extract pitch/energy/duration, how you normalize/process them, how you update the base model (from the workshop), and so on. The work done should be visible first in the report, and then in the code.

--------------
## Mandatory requirements
In general, the format of the current homework follows that of the first homework assignment (ASR).
You must organize the repository as in the first homework assignment, which is to break up the code into modules and follow the code style.

So, we don't accept homework if any of the following requirements are not satisfied:
* The code should be situated in a public github (or gitlab) repository
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
grade = 0.5 * MOS + 0.5 * (quality of code and report) + bonus
```
We will consider the normalized MOS for all the received works

To evaluate the MOS, you must add a synthesis in several configurations of the following sentences to the report:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

Configurations:
* usual generated audio
* audio with +20%/-20% for pitch/speed/energy
* audio with +20/-20% for pitch,speed and energy together

To give you a rough idea in advance of how we are going to evaluate the scale is approximately as follows:
   * 5: Words can be heard perfectly, interference can be attributed to vocoder artifacts
   * 4: There is some obvious noise, but all the words are clear and you do not have to listen to them to understand
   * 3: Most words are understandable, but some words are indistinguishable
   * 2: Very noisy audio, but some words are understandable
   * 1: At least it squeaks
   * 0: Vacuum cleaner

Bonus:
* (Up to +1.5) You can use other alignments, for example [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/) 
or find preprocessed ones, don't forget that then you will most likely have to change tokenizer pipeline
* (Up to +2) You can add some more regulators for your variance adaptor and regulate more variances. 
For example you can try extract emotions and add it as regulator
