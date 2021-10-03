# Homework 1 (ASR)

### Task
Implement and train a speech recognition system.


--------------
### Mandatory requirements
We don't accept homework if any of the following requirements are not satisfied:
* The code should be situated in a public github (or gitlab) repository
* All the necessary packages should be mentioned in `./requirements.txt` or in an installation 
  guide section of `README.md`
* All necessary resources (such as model checkpoints, LMs, and logs) should be downloadable with a script. 
  Mention the script (or lines of code) in the `README.md`
* You should implement all functions in `test.py` so that we can check your assignment
* Generally, your `test.py` and `train.py` scripts should run without issues after running all commands in your installation guide.
* You must provide the logs for the training of your final model from the start of the training. 
  Either use W&B reports or include your tensorboard directory in the resources of your project.
  


--------------
### Grade
```
grade = 
    quality_score 
    - (1.0 * days_expired) 
    - implementation_penalty 
    + optional_tasks_score
max_grade = max(grade, 10)
```


--------------
### Implementation penalties
We also require that you fulfill the following requirements. Not fulfilling them will result in score penalties.
* (Up to `-2.0 points`) Logging. Your tensorboard/W&B logs should include:
  * Text reports, including `target: {target_text}, prediction: {prediction_text}, CER: {cer}, WER: {wer}`
  * Images of your train/valid spectrograms
  * Gradient norm
  * Learning rate
  * Loss
  * audio records (after augmentation)
* (Up to `-2.0 points`) Implement a simple beam search for evaluation
* (Up to `-1.0 points`) Implement at least 4 types of audio augmentations


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
| 8.0 | LibriSpeech: test-other | -- | 20 | Your network can handle noisy audio. Good job. |
| 9.0 | LibriSpeech: test-other | -- | 10 | Technically better than a human |

__Note__: all the results will be sanity-checked on an unannounced dataset. 
So it's not a good idea to fine-tune on a test set. 
It will be considered cheating.

--------------
### Optional tasks
* (`1.0`) BPE instead of characters. You can use SentencePiece, HuggingFace, or YouTokenToMe.
* (`+1.5`) Use an external language model for evaluation. The choice of an LM-fusion method is up to you.
* (up to `+3.0`) Train a LAS model (instead CTC / with CTC). Don't forget to log your attention matrices. 
  You can skip beam-search or implement it for an extra *+1.0*
* (`+3.0`) Russian ASR. We will use test part of the russian Common Voice dataset for estimating your `quality_score`. 
  Note: this option has a high score value, but russian language is generally more difficult to recognize, 
  so expect your WERs and CERs to be lower on average compared to an English dataset.

--------------
### Bonus points / penalties
We can subtract or add up to 1.0 points for extremely bad or surprisingly clean code structure.

--------------
### Recommended workflow

Training a good NN model is a challenging task that is extremely difficult to debug.
We recommend you follow these steps:
1) Overfit your model on a single batch of examples
2) Train your model in LJ-speech dataset (until you achieve at least 30 WER on LJ-speech test set)
3) Fine-tune your model on Librispeech dataset (until you achieve at least 30 WER on Libirispeech clean test set)
4) Fine-tune your model on a mix of Librispeech and Common Voice datasets (for extra quality on Librispeech test sets)


If you run out of time during one of these steps, you can always submit your somewhat good result and 
  avoid getting deadline penalties.

Links: 
* [Mozilla Common Voice (en)](https://commonvoice.mozilla.org/ru)  
* [LibriSpeech](https://www.openslr.org/12)
* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

To save some coding time it is recommended to use [HuggingFace dataset library](https://github.com/huggingface/datasets). 
Look how easy it is:
```
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", split='train-clean-360')
```
