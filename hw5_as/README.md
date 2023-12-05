# Homework 5 (Voice Anti-spoofing)

## Task

Implement and train a Countermeasure (CM) system on the Logical Access partition of the [ASVSpoof 2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336) ([Kaggle Link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)). You may find the [ASVspoof 2019 evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) useful.

## Countermeasure systems

You should choose at least one of the following architectures for your CM system. Grade formula will depend on the chosen model (see description below).

**You cannot use implementations available on the internet**.

### LightCNN (Easy mode)

Implement [LightCNN (LCCN)](https://arxiv.org/abs/1511.02683) following the Speech Technology Center [paper](https://arxiv.org/abs/1904.05576).

**Hints**: 

1. Take training recipe and data preparation scheme from [this paper](https://arxiv.org/abs/2103.11326). Also, read the comparative study and think whether you should use A-Softmax or Cross-Entropy loss function (Explain your choice in the report).

2. Use STFT (FFT in the paper) as front-end. (Do ablation only if you took at least two different front-ends)

3. Put dropout layer as it is done in [this paper](https://ieeexplore.ieee.org/document/9428313).


*Maximum grade*: $6$

In order to achieve $7$, you should conduct experiments with at least two different types of front-ends (CQT, LFCC, etc.) and analyze training\evaluation results in your report. **Note**: it will be enough if you train the second model for a smaller number of epochs (smaller, but reasonable).

#### Report rules

If you read the papers and follow the hints, you should get a working solution with no problems. Do an ablation study on the necessity of the provided hints. Explain why the hint is useful, is it different from what was described in the original paper (or does it provide extra information). For each hint, explain what will change if you do not follow the hint, run the experiment and analyze the result (training curves, metrics, convergence speed, etc.). **Note**: it will be enough if you train the second model for a smaller number of epochs (smaller, but reasonable).

Do not forget about [General mandatory requirements](#general-mandatory-requirements).


### RawNet2 (Normal mode)

Implement [RawNet2](https://arxiv.org/abs/2011.01108). There are three types of sinc filters:

* S1: fixed Mel-scaled
* S2: fixed inverse Mel-scaled
* S3: fixed linear-scaled

You are free to choose any of them.

**Hints**: 
* You have to take the absolute value of sinc-layer output.
* Use 3-layer GRU, do BN and LeakyReLU before.
* Change hyperparameters to ASVspoof2021 style: 
    * Sinc filter length: 1024
    * `Conv(3,1,128) -> Conv(3,1,20)`, `Conv(3,1,512) -> Conv(3,1,128)`.
* Use weighted cross-entropy: $1.0$ for spoof class, $9.0$ for bona-fide.
* Set $0.0001$ weight decay in optimizer.
* Take Sinc-layer from the seminar. Note the difference between SincNet And RawNet2. Set `min_low_hz, min_band_hz` to zero. (For ablation: try non-zero, explain the difference with SincNet)


*Maximum grade*: $8$

In order to achieve $10$, you should conduct experiments with at least two different types of sinc filters and analyze training\evaluation results in your report. **Note**: it will be enough if you train the second model for a smaller number of epochs (smaller, but reasonable).

#### Report rules

If you read the paper and follow the hints, you should get a working solution with no problems. Do an ablation study on the necessity of the provided hints. Explain why the hint is useful, is it different from what was described in the original paper (or does it provide extra information). For each hint, explain what will change if you do not follow the hint, run the experiment and analyze the result (training curves, metrics, convergence speed, etc.). **Note**: it will be enough if you train the second model for a smaller number of epochs (smaller, but reasonable).

Do not forget about [General mandatory requirements](#general-mandatory-requirements).

### AASIST (Hard mode)

Implement [AASIST](https://arxiv.org/abs/2110.01200). You **must** use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) or [Deep Graph Library](https://docs.dgl.ai/index.html) for the implementation of GAT, HS-GAL parts of the model.

**Hints**:
* Check documentations and tutorials. You may find materials from [HSE GNN course](https://github.com/elvarid/gnn_course_hse) useful.
* Check source code for package implementations of GNN-layers.
* Add learnable positional encoding to spectral graph before applying graph module.
* Apply $0.2$ dropout to input for each graph-layer. Apply $0.2$ dropout to graphs before readout. Apply $0.5$ dropout to the last hidden layer.
* Apply GraphPool after the first HS-GAL and residual connection (not Graph Pool) after the second one.
* Take Sinc-layer from the seminar. Note the difference between SincNet And RawNet2. Set `min_low_hz, min_band_hz` to zero.

* In Graph Attention, use Softmax with temperature: $100$ for HS-GAL, $2$ for GAT.

* In Graph Attention, do not forget about Tanh activation in Softmax (see lecture).

*Maximum grade*: $10$

#### Report rules:

Task is hard enough on its own, so here we expect a standard report like in previous homeworks: see [General mandatory requirements](#general-mandatory-requirements). Hints ablation is not mandatory.

## Testing

You should provide `test.py` for your final solution. 

* Run it on any synthesized speech utterance that you generated in HW-3 or HW-4 (If you did not do these homeworks, ask a friend for the final utterances). 

* Choose any utterance you like from the internet and run your model on it. 

* Run you model on three utterances, provided in this [kaggle dataset](https://www.kaggle.com/datasets/blinorot/dla-hw5-test-audio) by teachers. 

Provide probabilitiess of audio beeing spoofed, obtained from your model. Listen the utterances and say if you think they are real or spoofed. Compare your thoughts with model predictions and describe. Does your model work good on these utterances? Explain.

## General mandatory requirements

In general, the format of the current homework follows that of the first homework assignment (ASR). You must organize the repository as in the first homework assignment, which is to break up the code into modules and follow the code style.

So, we don't accept homework if any of the following requirements are not satisfied:

* The code should be situated in a public github (or gitlab) repository.
* All the necessary packages should be mentioned in `./requirements.txt` or in an installation guide section of `README.md`.
* You must use W&B for logging losses and performance metrics.
* All necessary resources (such as model checkpoints) should be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
* Attach a report. That includes:
    - Description and result of each experiment.
    - How to reproduce your model?
    - Attach training logs to show how fast did you network train.
    - What worked and what didn't work?
    - What were the major challenges?
    - Comparison of different front-ends (for LCNN and RawNet2, if you did it).

## Grade

Your mark ($M$) will depend on your model performance ($P$) and the quality of your code and report ($Q$): 

* For LCNN and RawNet2: 
    $$M = 0.3P + 0.7 Q$$
    where $P, Q \in [0, \text{Maximum grade for the chosen CM}]$

* For AASIST:
    $$M = 0.8P + 0.2Q$$
    Solution in native PyTorch without GNN libraries does not count.

Perfomance metrics (must be calculated on the evalutation set of the LA partition of the ASVspoof 2019 Dataset):

* Equal Error Rate (EER).

[Code](./calculate_eer.py) for metrics is provided in this repo (see `compute_eer` function). You should provide metrics in your report and achieve the following performance:

* EER for $P = \text{Maximum grade for the chosen CM}$ ($P=0$ otherwise):
    * $< 5\\%$ for STFT-LCNN.
    * $< 5.3\\%$ for LFCC-LCNN.
    * $< 5.8\\%$ for S1-RawNet2.
    * $< 5.5\\%$ for S2-RawNet2.
    * $< 5\\%$ for S3-RawNet2.

* Detailed EER for AASIST:

    |   EER      | Grade ($P$)|
    | ----------- |-------|
    | $< 5\%$ | 4 |
    | $< 4\%$ | 6 |
    | $< 3\%$ | 8 |
    | $< 2.5\%$ | 10 |

**Note**: requested EERs are much higher than the ones described in the paper to save your time.

**Note 2**: for the ablation experiments (LCNN and RawNet) you do not have to get EER lower than written above. This threshold is only for the primary system, meaning that is mandatory to get below threshold only in one of your experiments (one per model).

**Bonus**: You can get up to $2$ bonus points if you solve tasks for at least two different CMs.

**Extra**: Hydra Bonus if not used before.


Time to achieve full $P$ grade in Kaggle for the teacher's solution (may vary a bit if the random seed is bad):

| Model | Time (h) |
|---|---|
| LCNN | 4.5-5 |
| RawNet2 | 5-6 |
| AASIST | 23-27 |