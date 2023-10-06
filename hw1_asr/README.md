# Homework 1 (ASR)


See the [template](https://github.com/WrathOfGrapes/asr_project_template)

### Task

Implement and train a neural-network speech recognition system with CTC loss.


--------------

### Mandatory requirements

We don't accept homework if any of the following requirements are not satisfied:

* The code should be stored in a public github (or gitlab) repository
* All the necessary packages should be mentioned in `./requirements.txt` or be installed in dockerfile
* All necessary resources (such as model checkpoints, LMs, and logs) should be downloadable with a script. Mention the
  script (or lines of code) in the `README.md`
* Unit tests (see template description) should work without errors
* You should implement all functions in `test.py` (for evaluation) so that we can check your assignment
* Basically, your `test.py` and `train.py` scripts should run without issues after running all commands in your
  installation guide.
* You must provide the logs for the training of your final model from the start of the training. We heavly recommend you
  to use W&B Reports feature.
* Attech a brief report. That includes:
    * How to reproduce your model? (_example: train 50 epochs with config `train_1.json` and 50 epochs
      with `train_2.json`_)
    * Attach training logs to show how fast did you network train
    * How did you train your final model?
    * What have you tried?
    * What worked and what didn't work?
    * What were the major challenges?

  Also attach a summary of all bonus tasks you've implemented.

--------------

### Grade

```
grade = 
    min(11, 
      quality_score 
      - (1.0 * days_expired) 
      - implementation_penalty 
      + optional_tasks_score
    )
```

--------------

### Implementation penalties

We also require that you fulfill the following requirements. Not fulfilling them will result in score penalties.

* (Up to `-2.0 points` if missing) Logging. Your W&B logs should include:
    * Text reports with random samples,
      including `target: {target_text}, prediction: {prediction_text}, CER: {cer}, WER: {wer}`
    * Images of your train/valid spectrograms
    * Gradient norm
    * Learning rate
    * Loss
    * Audio records (after augmentation)
* (Up to `-2.0 points` if missing) Implement at least a simple beam search for evaluation
* (Up to `-1.0 points` if missing) Implement at least 4 types of audio augmentations that are relevant for the ASR task

--------------

### Quality score

| Score  | Dataset | CER | WER| Description|
| ------------- | ------------- | ------------- | ------------- | -------------      |
| 1.0 | -- | -- | -- | At least you tried |
| 2.0 | LibriSpeech: test-clean | 50 | -- | Well, it's something |
| 3.0 | LibriSpeech: test-clean | 30 | -- | You can guess the target phrase if you try |
| 4.0 | LibriSpeech: test-clean | 20 | -- | It gets some words right |
| 5.0 | LibriSpeech: test-clean | -- | 40 | More than half of the words are looking fine |
| 6.0 | LibriSpeech: test-clean | -- | 30 | It's quite readable |
| 7.0 | LibriSpeech: test-clean | -- | 20 | Occasional mistakes  |
| 8.0 | LibriSpeech: **test-other** | -- | 30 | Your network can handle somewhat noisy audio. |
| 9.0 | LibriSpeech: **test-other** | -- | 20 | Somewhat suitable for practical applications. |
| 10.0 | LibriSpeech: **test-other** | -- | 10 | Technically better than a human. Well done! |

> ❗ All the results will be sanity-checked on an unannounced dataset. So it's not a good idea to fine-tune on a test set. It will be considered cheating.

--------------

### Optional tasks

* (`+0.5`) Use an external language model for evaluation. The choice of an LM-fusion method is up to you.
  _**Note: implementing this part will yield a very significant quality boost (which will improve your score by a lot).
  We heavily recommend that you implement this part, despite low bonus points amount.**_
* (`+1.0`) BPE instead of characters. You can use SentencePiece, HuggingFace, or YouTokenToMe.
* (up to `+3.0`) Train a LAS/RNN-T model (instead CTC / with CTC). Don't forget to log your attention matrices. You can
  skip beam-search or implement it for an extra *+1.0*
* (`+1.0`) Replace `ConfigParser` class and JSON configs with [Hydra](https://hydra.cc/docs/intro/) and YAML. You should utilize Hydra features and use them instead of methods in `ConfigParser`. The full points are given only if **all** logic of `ConfigParser` is replaced with Hydra.

--------------

### Bonus points / penalties

We can subtract or add up to `1.0 points` for extremely bad or surprisingly clean code structure.

--------------

### Recommended workflow

Recommended archetectures:

* [DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
* [QuartzNet](https://arxiv.org/abs/1910.10261). Note: it is difficult to train without a large
* [Jasper](https://arxiv.org/pdf/1904.03288.pdf)

Training a good NN model is a challenging task that is extremely difficult to debug. We recommend you follow these
steps:

1) Overfit your model on a single batch of examples
2) Train your model in LJ-speech dataset (until you achieve at least 30 WER on a custom test set)
3) Fine-tune your model on Librispeech dataset (until you achieve at least 30 WER on Libirispeech clean test set)
4) Fine-tune your model on a mix of Librispeech and Common Voice datasets (for extra quality on Librispeech test sets)

If you run out of time during one of these steps, you can always submit your somewhat good result and avoid getting
deadline penalties.

Links:

* [Mozilla Common Voice (en)](https://commonvoice.mozilla.org/ru)
* [LibriSpeech](https://www.openslr.org/12)
* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

To save some coding time it is recommended to use [HuggingFace dataset library](https://github.com/huggingface/datasets)
. Look how easy it is:

```
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", split='train-clean-360')
```
