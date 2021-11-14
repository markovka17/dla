# Homework 2 (KWS)

### Task
Implement streaming KWS, speed up and compress the model 10 times.

This homework don't aimed to implement fancy architecture and write clean and well structured code.
On the contrary, you will be able to write down all the code in notebooks, but with a lot of comments and graphs :).
The purpose of the work is to explore various ways to accelerate NNs.

So, you need to take tht **base model** from `seminar.ipynb` and to do the following things:
  1) [Streaming] 
      This model works differently during training and inferance. During training you have some
      fixed input and you know that it has a keyword (or not). During the inferance, you read the T frames
      and make a prediction on them. And the `next step` is to read the `T + 1` frame's,
      run the neural network just for it, and make a prediction based on it and the `T - 1` of the previous frames.
     
      When you implement streaming mode note that you need to:
      1. define `max_window_length` and restrict you buffer with this parameter.
      Don't forget to refresh the buffer by dropping the first frame and adding `T + 1` frame.
      2. share `hidden state` of GRU between steps
      3. implement streaming KWS as separate `class`

      To demonstrate the work in streaming mode, take two random audio tracks of 10-20 seconds and glue them together so that your
      keyword will be between them. Run the model through this glued track and draw how the probability of your keyword changing over time.
      The most suituble way to visualize is notebook.
     
  2) [Speed up & Compression] 
     You need to speed up and compress **base model** 10 times.
     You are free in the methods of achieving the goal, but to make it easier, you can start with:
     1. Dark Knowledge Distillation.
        Design the smaller model with same topology and distill it with teacher (**base model**).
     2. Knowledge Distillation of attention mechanism.
        Design the smaller attention mechanism and distill it by optimizing KL divergence between propability vectors of teacher and student
     3. Quantization.
        Quantize weights of model to lower precision (fp16/qint8). Pay attention to QAT and official PyTorch tutorial about Quantization.
     4. Structured Pruning.
        Pay attention to official PyTorch tutorial about Pruning.
        
     
     Important points(!):
     1. The quality of the **base model** is `~0.0002`, so your optimized model should not have a quality less than `~0.0002 * 0.9`.
     2. You should try at least 6 settings of model optimization (speed up and/or compression) and visualize it in two ways: Metric-FLOPs and Metric-Memory.
        Either combine it together. You may get inspiration from https://arxiv.org/pdf/1905.11946.pdf.

        Examples of settings:
          1. Dark Knowledge Distillation
          2. Dark Knowledge Distillation + fp16
          3. Attention Distillation + Dark Knowledge Distillation + qint8
          4. QAT + fp16 + Pruning
          5. and so on

        Note that, for example, the temperature change during distillation is not a new setting:)    

--------------
### Mandatory requirements
We don't accept homework if any of the following requirements are not satisfied:
* The code and report should be situated in a public github (or gitlab) repository
* All necessary resources (such as model checkpoints, and logs) should be downloadable with a script. 
  Mention the script (or lines of code) in the `README.md`
* Attech a report. That includes:
  * Description and result of each experiment
  * How to reproduce your model?
  * Attach training logs to show how fast did you network train
  * What worked and what didn't work?
  * What were the major challenges?

### Time estimation
One of the way to easly estimate running time
```python
import time


class Timer:

    def __init__(self, name: str, verbose=False):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} | Elapsed time : {self.t:.2f}")
```

### FLOPs/MACs estimation
```python
from thop import profile  # !pip install thop

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Conv1d(1, 1, 3, bias=False)
    
    def forward(self, x):
        return self.model(x)
        
profile(Model(), (torch.randn(1, 1, 4), ))  # -> (6.0 MACs, 3.0 parameters)
```

### Memory estimation
```python
def get_size_in_megabytes(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = next(model.parameters()).element_size()
    return (num_params * param_size) / (2 ** 20)
```
   

--------------
### Grade
```
grade = 4 * (`compression rate` / 10) + 4 * (`speed up rate` / 10) 
  + 2 * `streaming`
  - days_expired - report_penalty (up to 3.0 points)
```
