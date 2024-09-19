# Homework 1 (ASR)

As always, your code **must** be based on the provided [Project Template](https://github.com/Blinorot/pytorch_project_template). Feel free to choose any of the code branches as a starting point (maybe you will find the `main` branch easier than the `ASR` one). However, we strongly recommend the [`ASR` branch](https://github.com/Blinorot/pytorch_project_template/tree/example/asr) for this homework.

### Task

Implement and train a neural-network speech recognition system with a CTC loss.

---

### Mandatory requirements

We **do not** accept the homework if any of the following requirements are not satisfied:

- The code should be stored in a public github (or gitlab) repository and based on the provided template. (Before the deadline, use a private repo. Make it public after the deadline.)
- All the necessary packages should be mentioned in `./requirements.txt` or be installed in a dockerfile.
- All necessary resources (such as model checkpoints, LMs, and logs) should be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
- You should implement all functions in `inference.py` (for evaluation) so that we can check your assignment (see [Testing](#testing) section).
- Basically, your `inference.py` and `train.py` scripts should run without issues after running all commands in your installation guide. Create a clean env and deploy your lib into a separate directory to check if everything works fine given that you follow your stated installation steps.
- You must provide the logs for the training of your final model from the start of the training. We heavily recommend you to use W&B (CometML) Reports feature.
- Attach a brief report that includes:

  - How to reproduce your model? (_example: train 50 epochs with config `train_1.yaml` and 50 epochs with `train_2.yaml`_)
  - Attach training logs to show how fast did you network train
  - How did you train your final model?
  - What have you tried?
  - What worked and what did not work?
  - What were the major challenges?

  Also attach a summary of all bonus tasks you have implemented.

---

### Grade

Your grade will be based on the model performance and code\report quality.

```
grade = quality_score - implementation_penalties - report_penalties
```

---

### Implementation penalties

We also require that you fulfill the following requirements. Not fulfilling them will result in score penalties.

- (Up to `-2.0 points` if missing) Logging. Your W&B (CometML) logs should include:
  - Text reports with random samples, including `target: {target_text}, prediction: {prediction_text}, CER: {cer}, WER: {wer}`
  - Images of your train/valid spectrograms
  - Gradient norm
  - Learning rate
  - Loss
  - Audio records / spectrograms (after augmentation)
    1. Create a separate run showing that all you augmentations indeed work.
    2. Log audio / spectrograms in each of your experiments showing that your augmentations are not too severe (otherwise, it is not possible to predict correct text on your input data).
- (Up to `-2.0 points` if missing) Implement a simple hand-crafted beam search for the evaluation. You must provide a run showing that your beam search works (improves score in comparison to argmax version).
- (Up to `-1.0 points` if missing) Implement at least 4 types of audio augmentations that are relevant for the ASR task

---

### Quality score

| Score | Dataset                     | CER | WER | Description                                                                     |
| ----- | --------------------------- | --- | --- | ------------------------------------------------------------------------------- |
| 1.0   | --                          | --  | --  | At least you tried                                                              |
| 2.0   | LibriSpeech: test-clean     | 50  | --  | Well, it's something                                                            |
| 3.0   | LibriSpeech: test-clean     | 30  | --  | You can guess the target phrase if you try                                      |
| 4.0   | LibriSpeech: test-clean     | 20  | --  | It gets some words right                                                        |
| 5.0   | LibriSpeech: test-clean     | --  | 40  | More than half of the words are looking fine                                    |
| 6.0   | LibriSpeech: test-clean     | --  | 30  | It's quite readable                                                             |
| 7.0   | LibriSpeech: test-clean     | --  | 20  | Occasional mistakes                                                             |
| 8.0   | LibriSpeech: **test-other** | --  | 30  | Your network can handle somewhat noisy audio.                                   |
| 8.5   | LibriSpeech: **test-other** | --  | 25  | Your network can handle somewhat noisy audio but it is still just close enough. |
| 9.0   | LibriSpeech: **test-other** | --  | 20  | Somewhat suitable for practical applications.                                   |
| 9.5   | LibriSpeech: **test-other** | --  | 15  | You are close to human performance.                                             |
| 10.0  | LibriSpeech: **test-other** | --  | 10  | Technically better than a human. Well done!                                     |

> [!IMPORTANT]
> All the results will be sanity-checked on an unannounced dataset. So it's not a good idea to fine-tune on a test set. It will be considered as cheating.

---

### Testing

You **must** add `inference.py` script and a `CustomDirDataset` Dataset class in `src/datasets/` with a proper config in `src/configs/`.

The `CustomDirDataset` should be able to parse any directory of the following format:

```bash
NameOfTheDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav # may be flac or mp3
│   ├── UtteranceID2.wav
│   .
│   .
│   .
│   └── UtteranceIDn.wav
└── transcriptions # ground truth, may not exist
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

It should has an argument for the path to this custom directory that can be changed via `Hydra`-options.

The `inference.py` script must apply the model on the given dataset (custom-one or any other supported in your `src`) and save predicted text in the requested directory. The predicted text id should be the same as the utterance id (so they can be matched together).

Provide a separate script that calculates WER and CER given the path to ground truth and predicted transcriptions.

Mention the lines on how to run inference on your final model in the `README`. Include the lines for the script too.

### Optional tasks

- (`+0.5`) Use an external language model (LM) for evaluation. The choice of an LM-fusion method is up to you.
  _**Note: implementing this part will yield a very significant quality boost (which will improve your score by a lot). We heavily recommend that you implement this part, despite low bonus points amount.**_
- (`+1.0`) BPE instead of characters. You can use SentencePiece, HuggingFace, or YouTokenToMe.
- (up to `+3.0`) Train a LAS model (instead CTC / with CTC). Don't forget to log your attention matrices. You can skip beam-search or implement it for an extra _+1.0_

> [!NOTE]
> If you use LM, you are allowed to take pretrained LM from the internet and use external library for LM-based beam search. However, you still have to write your own hand-crafted beam search (see [implementation penalties](#implementation-penalties)) and validate it. Similarly, you must provide a run showing that your LM works.

---

### Bonus points / penalties

We can subtract or add a certain amount of points for extremely bad or surprisingly clean code structure, non-standard approaches, very good report, etc..

---

### Recommended workflow

Recommended architectures:

- [DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
- [QuartzNet](https://arxiv.org/abs/1910.10261). Note: it is difficult to train without a large batch size and nice GPU.
- [Jasper](https://arxiv.org/pdf/1904.03288.pdf)
- [Conformer](https://arxiv.org/abs/2005.08100)

Training a good NN model is a challenging task that is extremely difficult to debug. We recommend you to follow these steps:

1. Overfit your model on a single batch of examples (a.k.a. One Batch Test).
2. Train your model on Librispeech dataset (until you achieve at least 30 WER on Libirispeech clean test set).
3. Fine-tune your model on a mix of Librispeech and Common Voice datasets (for extra quality on Librispeech other test sets) or add more augmentations.

If you run out of time during one of these steps, you can always submit your somewhat good result and avoid getting deadline penalties.

> [!IMPORTANT]
> We have limited GPU resources that should be used effectively. Therefore, we suggest you to use free Google Colab (8h/day) and Kaggle (30h/week) GPUs for debugging. The whole homework can be solved in Kaggle. Use this [Kaggle dataset](https://www.kaggle.com/datasets/a24998667/librispeech). Also follow the steps below:

Start your assignment by following these steps locally (CPU is enough):

0. Look at the `main` and `example/image_classification` branches of the template. They contain fully-working pipelines that may help you to understand the template code structure and workflow.
1. Fill in all TODOs in the template.
2. Tech writing (logging, Dataset and Dataloader creation, Model init, Trainer init). You can test it without computational overkill.
3. Model testing: verify the inputs and outputs (especially their shapes) are exactly the needed ones (which you wanted to obtain).
4. Run One-Batch training epoch with commented `optim.backward` to test that logging and other infrastructure works fine.
5. Run One-Batch test on full project pipeline with a simple MLP model. Make sure that loss goes close to zero and CER/WER are almost perfect.

Links:

- [Mozilla Common Voice (en)](https://commonvoice.mozilla.org/ru)
- [LibriSpeech](https://www.openslr.org/12)
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

To save some coding time, it is recommended to use [HuggingFace dataset library](https://github.com/huggingface/datasets). Look how easy it is:

```
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", split='train-clean-360')
```
