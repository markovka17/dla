# Homework 4 (NV)

### Task
Implement [HiFiGAN](https://arxiv.org/pdf/2010.05646.pdf).

This homework aims at implementing the second part of the TTS called "(neural) vocoder" based on HiFiGAN. 
The model will be trained on the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

Feel free to  take `Featurizer`, `Dataset` and `Collator` from `aligner.ipynb` of previous homework.

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

The corresponding `MelSpectograms` will be announced later.
