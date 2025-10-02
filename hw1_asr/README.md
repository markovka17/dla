# Homework 1 (ASR)

> [!IMPORTANT]
> As always, your code **must** be based on the provided [Project Template](https://github.com/Blinorot/pytorch_project_template). Feel free to choose any of the code branches as a starting point (maybe you will find the `main` branch easier than the `ASR` one). However, we strongly recommend the [`ASR` branch](https://github.com/Blinorot/pytorch_project_template/tree/example/asr) for this homework.

To make yourself familiar with the template, we suggest watching the Tutorials from the `README` or/and our seminars on "Creating convenient DL pipelines and clean code" ([Part I](https://github.com/markovka17/dla/tree/2025/week01) and [Part II](https://github.com/markovka17/dla/tree/2025/week03)). As a starting point, you can implement simple Image Classification using the `main` branch and then compare your solution with the template version.

### Task

Implement and train a neural-network speech recognition system (a CTC, RNN-T, or LAS-based variant).

> [!IMPORTANT]
> You **cannot** use implementations available on the internet.

---

### Mandatory requirements

We **do not** accept the homework if any of the following requirements are not satisfied:

- The code should be stored in a public github (or gitlab) repository and based on the provided template. (**Before the deadline, use a private repo. Make it public after the deadline.**)
- All the necessary packages should be mentioned in `./requirements.txt` or be installed in a dockerfile.
- All necessary resources (such as model checkpoints, LMs, and logs) should be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
- You shall implement all functions in `inference.py` (for evaluation) so that we can check your assignment (see [Testing](#testing) section).
- Basically, your `inference.py` and `train.py` scripts should run without issues after running all commands in your installation guide. Create a clean env and deploy your lib into a separate directory to check if everything works fine given that you follow your stated installation steps.
- You shall create a demo notebook (see [Demo](#demo) section).
- You must provide the logs for the training of your final model from the start of the training. We heavily recommend you to use W&B (CometML) Reports feature.
- Attach a brief report that includes:

  - How to reproduce your model? (_example: train 50 epochs with config `train_1.yaml` and 50 epochs with `train_2.yaml`_)
  - Attach training logs to show how fast did you network train
  - How did you train your final model?
  - What have you tried?
  - What worked and what did not work?
  - What were the major challenges?
  - Conduct some analysis on your plots and experiments.

  Also attach a summary of all bonus tasks you have implemented.

> [!NOTE]
> If your version of Comet ML does not support audio panels in Reports, you can use Python Panel. Code for the Python Panel is shown below (carefully look at the normalization for plotting and check if it works for your audio):

<details>

<summary>Audio Panel Code (as Python Comet ML Panel)</summary>

```python
# Comet Python Panels BETA, full documentation available at:
# https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/

# By Petr Grinberg @ https://github.com/markovka17/dla 2024

from comet_ml import API, ui
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os
import numpy as np
import streamlit as st


# Get available metrics
api = API()
metrics = api.get_panel_metrics_names()

exps = api.get_panel_experiments()
all_possible_steps = []
for exp in exps:
    assets = exp.get_asset_list()
    for asset in assets:
        if asset["type"] == "audio":
            step = asset["step"]
            all_possible_steps.append(step)
all_possible_steps = sorted(set(all_possible_steps))

step_option = st.selectbox(
    "Choose a step",
    all_possible_steps,
)

for exp in exps:
    exp_name = exp.name
    assets = exp.get_asset_list()
    for asset in assets:
        if asset["type"] == "audio":
            curl_command = asset["curlDownload"]
            filename = asset["fileName"]
            step = asset["step"]
            if step != step_option:
                continue
            os.system(curl_command)
            sr, wav_array = read(curl_command.split()[-1])
            print(f"Exp: {exp_name}, Step: {step}, Name: {filename}")
            wav_dtype = wav_array.dtype
            max_amplitude = np.iinfo(np.int16).max
            wav_array = wav_array / max_amplitude
            # Visualize the data
            figure, ax = plt.subplots()
            wav_time = np.arange(wav_array.shape[0]) / sr
            ax.plot(wav_time, wav_array)
            ax.set_xlabel("Time (s)")
            ax.grid()
            st.pyplot(figure)
            st.audio(wav_array, format="audio/wav", sample_rate=sr, loop=False)
```

</details>

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
- (Up to `-2.0 points` if missing) Implement a simple hand-crafted beam search for the evaluation (Use the one that corresponds to your type of model: CTC, LAS, or RNN-T). **You must provide a run showing that your beam search works (improves score in comparison to argmax version).**
- (Up to `-1.0 points` if missing) Implement at least 4 types of audio augmentations that are relevant for the ASR task

> [!NOTE]
> One of the homework goals is to get practice in proper project development. Thus, we will look at your Git history, `README`, `requirements.txt`, etc. Provide a comprehensive `README`, do not indicate packages that are not actually needed in the `requirements`, write meaningful commit names, etc. Do not hesitate to use `pre-commit` for code formatting (template version includes `black` and `isort`).

---

### Quality score

> [!NOTE]
> This year, we allow you to do any of the 3 model types. However, currently we provide boundaries only for the CTC. If you want to do RNN-T or LAS, we advise you to wait for the corresponding update to ensure that the models are allowed and the boundaries are set.

Below is the table for **CTC**-based submissions.

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

For **RNN-T** and **LAS** boundaries are provided below (**Implementing RNN-T or LAS gives you +1 point if you use it as a final HW submission**):

| Score | Dataset                     | CER (LAS / RNN-T) | WER (LAS / RNN-T) |
| ----- | --------------------------- | ----------------- | ----------------- |
| 1.0   | --                          | --                | --                |
| 2.0   | LibriSpeech: test-clean     | 50                | --                |
| 3.0   | LibriSpeech: test-clean     | 30                | --                |
| 4.0   | LibriSpeech: test-clean     | 10                | --                |
| 5.0   | LibriSpeech: test-clean     | --                | 30                |
| 6.0   | LibriSpeech: test-clean     | --                | 20                |
| 7.0   | LibriSpeech: test-clean     | --                | 15                |
| 8.0   | LibriSpeech: **test-other** | --                | 25                |
| 8.5   | LibriSpeech: **test-other** | --                | 20                |
| 9.0   | LibriSpeech: **test-other** | --                | 15                |
| 9.5   | LibriSpeech: **test-other** | --                | 12                |
| 10.0  | LibriSpeech: **test-other** | --                | 8                 |

> [!IMPORTANT]
> All the results will be sanity-checked on an unannounced dataset. So it's not a good idea to fine-tune on a test set. It will be considered as cheating.

---

### Testing

You **must** add `inference.py` script and a `CustomDirDataset` Dataset class in `src/datasets/` with a proper config in `src/configs/`.

The `CustomDirDataset` shall be able to parse any directory of the following format:

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

It shall has an argument for the path to this custom directory that can be changed via `Hydra`-options.

The `inference.py` script must apply the model on the given dataset (custom-one or any other supported in your `src`) and save predicted text in the requested directory. The predicted text id should be the same as the utterance id (so they can be matched together).

**Provide a separate script** that calculates WER and CER given the path to ground truth and predicted transcriptions.

Mention the lines on how to run inference on your final model in the `README`. Include the lines for the script too.

### Demo

In addition to providing detailed instructions in the `README`, a great repository shows a demo of the project, i.e., showcases how to use it. This allows end-user to see directly how to apply your code for their needs. The basic version of demo is the inference notebook. You must include such an `ipynb` notebook in your repository.

The structure of the demo:

1. Clones your repository and follows all the installation steps from your `README`.
2. Downloads all the required checkpoints.
3. Shows how to run `inference.py` on a sample dataset and how to calculate metrics using `calc_metrics.py` script.
4. Allows the end-user to enter their own dataset link (e.g. using G-Drive) in the format from [Testing](#testing) section and then runs inference and metrics calculation on it.
5. All of this is accompanied with comments explaining what your cell is doing and/or gives some instructions.

> [!IMPORTANT]
> Be sure to check your demo yourself. If the demo is absent or not working, you will get $0$ for the homework. We will use your demo notebook with our secret dataset link to evaluate your submission. **A user only have to pass the link and run the cells to get metrics on the linked dataset, no other steps should be expected from the user.**

To ensure fair evaluation, use Google Colab as the testing server for the verification of your demo. If your demo code works in a fresh Colab session, then there should not be any problems. If it doesn't work in Colab, we will penalize it.

### Optional tasks

- (`+0.5`) Use an external language model (LM) for evaluation. The choice of an LM-fusion method is up to you.
  _**Note: implementing this part will yield a very significant quality boost (which will improve your score by a lot). We heavily recommend that you implement this part, despite low bonus points amount.**_
- (`+1.0`) BPE instead of characters. You can use SentencePiece, HuggingFace, or YouTokenToMe.

> [!NOTE]
> If you use LM, you are allowed to take pretrained LM from the internet and use external library for LM-based beam search. However, you still have to write your own hand-crafted beam search (see [implementation penalties](#implementation-penalties)) and validate it. Similarly, you **must provide a run showing that your LM works**.

> [!NOTE]
> The same holds for the `BPE` bonus. You need to conduct two experiments with\without BPE and compare them (a.k.a. ablation studies). Correct implementation of BPE should perform at least as good as non-BPE.

---

### Bonus points / penalties

We can subtract or add a certain amount of points for extremely bad or surprisingly clean code structure, non-standard approaches, very good report, etc..

> [!IMPORTANT]
> Use of any LLM for doing the homework is prohibited. We believe that the goal of LLMs is to help doing things faster, but they might harm your development of skills. The course aims to teach you proper skills and avoiding LLMs will allow you to take the most from our materials.

---

### Recommended workflow

While you can choose any architecture you want, we recommend these architectures:

- [DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
- [Conformer](https://arxiv.org/abs/2005.08100)

> [!NOTE]
> If a paper uses a model with a certain loss, nothing prevents you from using the same architecture but with another loss. For example, you can do Conformer-CTC, Conformer-RNN-T (once RNN-T/LAS boundaries are published), or even a hybrid model.

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

To save some coding time, it is recommended to use [HuggingFace dataset library](https://github.com/huggingface/datasets). Look how easy it is:

```
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", split='train-clean-360')
```

For model checkpoint submission, you can also upload it to HuggingFace instead of Google Drive.
