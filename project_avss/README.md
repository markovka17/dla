# Project (Audio-Visual Source Separation)

> [!IMPORTANT]
> As always, your code **must** be based on the provided [Project Template](https://github.com/Blinorot/pytorch_project_template).

## Task

Implement and train an audio-only or audio-visual Source Separation system on the dataset provided by the teachers (see our official channel). You cannot use external datasets without teachers' permission, except for noise/rir augmentations.

> [!IMPORTANT]
> You **cannot** use implementations available on the internet.

## Project Outcomes

Each student is assigned with a team of 2-3 people that should work together on the project. To find your group, see our official channel. The submission consists of a code repository that reproduces the solution (including a demo notebook) and a report that explains the team's system and experiments.

### Code

As always, your code **must** be based on the provided [Project Template](https://github.com/Blinorot/pytorch_project_template). Feel free to choose any of the code branches as a starting point (maybe you will find the `main` branch easier than the `ASR` one).

The code must follow the following rules:

- The code should be stored in a public github (or gitlab) repository (**one per team**) and based on the provided template. (**Before the deadline, use a private repo. Make it public after the deadline.**)
- All the necessary packages should be mentioned in `./requirements.txt`, `environment.yaml`, dockerfile or in an installation guide section of the `README.md`.
- You must use `W&B`/`Comet ML` for logging losses, objects (like audio), and performance metrics.
- You shall create a demo notebook (see [Demo](#demo) section).
- You shall implement all functions in `inference.py` (for evaluation) so that we can check your assignment (see [Testing](#testing) section).
- All necessary resources (such as model checkpoints) should be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
- Your solution should be reproducible with a script. Create a script-file or mention the required lines in the `README.md`.

> [!IMPORTANT]
> This is a group project. So the whole team **must work in a single repository**. Create branches, pull-requests, merge branches, split the responsibilities, etc. We will look at the commits history and investigate how you managed team code development.

> [!NOTE]
> Due to restrictions of the free `Comet ML` subscription, we allow each student to have separate logs. If you use logs in your report, download them from WandB/Comet and create plots manually. **Self-crafted figures will also improve the quality and design of your report.**

During you research, you will (and have to) try different models, different training schemes, etc. This is when `Hydra`-based configuration and purpose-based separation of configs should help a lot. Do not create litter commits just to update the config each time you want to try a new configuration, use command-line `Hydra` options instead (See [Seminar with Q&A and project template discussion](https://github.com/markovka17/dla/tree/2025/week03)).

#### Testing

You shall add `inference.py` script and a `CustomDirDataset` Dataset class in `src/datasets/` with a proper config in `src/configs/`.

The `CustomDirDataset` should be able to parse any directory with mixed speech of the following format:

```bash
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```

It should have an argument for the path to this custom directory that can be changed via `Hydra`-options.

The `inference.py` script must apply the model on the given dataset (custom-one or any other supported in your `src`) and save separated utterances in the requested directory. The separated utterance output name should be the same as the mix name (so they can be matched together, see example for how ground-truth utterances are located). Note that ground-truth may not be given and you should handle this case.

Provide a separate script that calculates all required metrics given the path to ground truth and predicted `s1` and `s2` utterances.

Mention the lines on how to run inference on your final model in the `README`. Include the lines for the script too.

**Clearly indicate the default behavior where your predictions are saved in the README**.

#### Demo

In addition to providing detailed instructions in the `README`, a great repository shows a demo of the project, i.e., showcases how to use it. This allows end-user to see directly how to apply your code for their needs. The basic version of demo is the inference notebook. You must include such an `ipynb` notebook in your repository.

The structure of the demo:

1. Clones your repository and follows all the installation steps from your `README`.
2. Downloads all the required checkpoints.
3. Shows how to run `inference.py` on a sample dataset and how to calculate metrics using `calc_metrics.py` script.
4. Allows the end-user to enter their own dataset link (e.g. using G-Drive) in the format from [Testing](#testing) section and then runs inference and metrics calculation on it.
5. All of this is accompanied with comments explaining what your cell is doing and/or gives some instructions.

> [!IMPORTANT]
> Be sure to check your demo yourself. If the demo is absent or not working, the whole team will get $0$ for the homework. We will use your demo notebook with our secret dataset link to evaluate your submission. **A user only have to pass the link and run the cells to get metrics on the linked dataset, no other steps should be expected from the user.**

To ensure fair evaluation, use Google Colab as the testing server for the verification of your demo. If your demo code works in a fresh Colab session, then there should not be any problems. If it doesn't work in Colab, we will penalize it.

> [!IMPORTANT]
> This time, the secret dataset uses Yandex Disk. Make sure you can support downloading from Yandex, e.g. using [Yadisk tool](https://github.com/Yegorov/yadisk).

### Report

Each team must provide an article-style report with all the required sections: Abstract, Introduction\Related Work, Methodology\Experimental Setup, Results\Discussion, Conclusion, and References. The page limit is 4 pages (4.5 if your report is in Russian) and non-limited for the bibliography.

Look at [these guidelines](https://github.com/Blinorot/pytorch_project_template/blob/report/HowToWriteAPaper.pdf) to understand what should be written in each section. **Use the files in guidelines repository as a `LaTeX` template. Other templates will not be accepted. You cannot change the formatting of the template (spacing, margins, etc.)**

> [!IMPORTANT]
> At the end of the report, include a Contributions Section (not counted in the page limit), explaining the contributions of each team member.

## Hints

The provided dataset is audio-visual: each mix utterance `s1id_s2id.wav` has corresponding `npz` files `s1id.npz` and `s2id.npz` for a mouth-region of the corresponding speaker from the mix. Therefore, your model performance will increase significantly if you utilize video information.

You are **allowed** to use any pre-trained video feature extractors from [this Lip-reading repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) (**Only from this**). The given `npz` dataset files are compatible with these networks.

As the project starts before we discuss audio-visual models in the lectures, we advise you to start with audio-only models. This will also be helpful for the report to prove your design choices.

To start your research, we suggest reading some surveys (in general, this is a good way to quickly start working in a field that is new for you). Search yourself using Google Scholar or use the suggestions below:

<details>

<summary>List of surveys</summary>

- [Wang, DeLiang, and Jitong Chen. "Supervised speech separation based on deep learning: An overview." IEEE/ACM transactions on audio, speech, and language processing 26.10 (2018): 1702-1726.](https://arxiv.org/pdf/1708.07524)

- [Zhu, Hao, et al. "Deep audio-visual learning: A survey." International Journal of Automation and Computing 18.3 (2021): 351-376.](https://arxiv.org/abs/2001.04758)
- [Michelsanti, Daniel, et al. "An overview of deep-learning-based audio-visual speech enhancement and separation." IEEE/ACM Transactions on Audio, Speech, and Language Processing 29 (2021): 1368-1396](https://arxiv.org/abs/2008.09586)

</details>

<details>

<summary>Papers discussed in the speech separation lecture:</summary>

- [Luo, Yi, and Nima Mesgarani. "Tasnet: time-domain audio separation network for real-time, single-channel speech separation." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.](https://ieeexplore.ieee.org/document/8462116)

- [Luo, Yi, Zhuo Chen, and Nima Mesgarani. "Speaker-independent speech separation with deep attractor network." IEEE/ACM Transactions on Audio, Speech, and Language Processing 26.4 (2018): 787-796.](https://dl.acm.org/doi/10.1109/TASLP.2018.2795749)

- [Wang, Quan, et al. "Voicefilter: Targeted voice separation by speaker-conditioned spectrogram masking." arXiv preprint arXiv:1810.04826 (2018).](https://arxiv.org/abs/1810.04826)

- [Luo, Yi, and Nima Mesgarani. "Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation." IEEE/ACM transactions on audio, speech, and language processing 27.8 (2019): 1256-1266.](https://arxiv.org/pdf/1809.07454.pdf)

- [Ge, Meng, et al. "Spex+: A complete time domain speaker extraction network." arXiv preprint arXiv:2005.04686 (2020).](https://arxiv.org/pdf/2005.04686.pdf)

- [Luo, Yi, Zhuo Chen, and Takuya Yoshioka. "Dual-path rnn: efficient long sequence modeling for time-domain single-channel speech separation." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.](https://ieeexplore.ieee.org/document/9054266)

- [Wang, Quan, et al. "VoiceFilter-Lite: Streaming targeted voice separation for on-device speech recognition." arXiv preprint arXiv:2009.04323 (2020).](https://arxiv.org/pdf/2009.04323)

- [Rikhye, Rajeev, et al. "Multi-user VoiceFilter-Lite via attentive speaker embedding." 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021.](https://arxiv.org/pdf/2107.01201.pdf)

</details>

<details>

<summary>Papers discussed in the audio-visual lecture:</summary>

- [Pegg, Samuel, Kai Li, and Xiaolin Hu. "RTFS-Net: Recurrent time-frequency modelling for efficient audio-visual speech separation." arXiv preprint arXiv:2309.17189 (2023).](https://arxiv.org/abs/2309.17189)

- [Li, Kai, et al. "An audio-visual speech separation model inspired by cortico-thalamo-cortical circuits." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).](https://arxiv.org/abs/2212.10744)

</details>

> [!IMPORTANT]
> You can try to implement any of the suggested models, search for the literature with other systems, modify them, build something on top of them, or design your own new architectures. The goal of this project is to get practice in doing research, so explore the literature, conduct experiments, justify your choices. Note that some of the models are extremely computationally expensive, so some adjustments to the architectures will be required. You can also use gradient accumulation or mixed-precision training techniques (see more info [here](https://github.com/LauzHack/deep-learning-bootcamp/tree/main/day06)).

## Grade

For the project, you will get two grades: Code and Report.

**Code** grade $\in [0, 10]$. Depends on the quality of your code and repository (works as penalty, if not good enough), how you handled team code development (works as penalty, if not good enough), and the best model performance.

Model performance grade is based on the competition score with other teams and baselines.

| SI-SNRi | Grade for Passing Baseline |
| ------- | -------------------------- |
| $\ge 5$ | $4$                        |
| $\ge 9$ | $7$                        |

- If you achieved $< 5$ SI-SNRi, your performance grade is $0$.

- If you achieved SI-SNRi between $5$ and $9$ your grade is scaled between the baseline grades according to your place in the competition across teams with the same SI-SNRi range.

- If you achieved SI-SNRi $\ge 9$ your grade is scaled between the baseline grade and $10$ according to your place in the competition across teams with the same SI-SNRi range.

**Also indicate other relevant metrics, like SDRi, PESQ, and STOI**.

**Report** grade $\in [0, 10]$. Depends on the quality and clarity of your report, how well you followed the guidelines and article-style. We will also take into account the experiments you conducted, their discussion, etc.

**Bonus**. Apart from model performance competition, we will have a model speed (grade $\in [0, 0.6]$) / resource consumption (grade $\in [0, 0.4]$) competition within the solutions that outperformed the highest baseline. If you want to participate in the bonus, include all relevant metrics, such as FLOPs (MACs), memory required to process 1 batch with a single mix and video, the number of model parameters, size of the saved model on disk, time for 1 step, etc. (see [guidelines](https://github.com/Blinorot/pytorch_project_template/blob/report/HowToWriteAPaper.pdf)). **Note**: To ensure proper comparison between different teams, you **must** calculate inference time for a single batch consisting of 1 mix audio and video **using Colab** (so the hardware and system packages are the same for all teams). In addition, provide a script that calculates all metrics for this bonus for your final solution. Include it in the `demo` notebook too.

**Report Bonus**: we may add report bonus for interesting approaches that go beyond implementation of an existing paper.

> [!IMPORTANT]
> In this project, you can choose any architecture you like. This can be one of the models from lectures or any model you found in the literature. We expect you to try different setups. Remember that using open-source code with these architectures is forbidden. **Do not forget about proving your design choices by conducting ablation studies and citing literature**.
