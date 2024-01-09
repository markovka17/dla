from datetime import datetime
import numpy as np
import wandb


class WanDBWriter:
    def __init__(self, config):
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if not hasattr(config, 'wandb_project'):
            raise ValueError("please specify project name for wandb")

        wandb.init(
            project=getattr(config, 'wandb_project'),
            config=config
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{self.mode}/{scalar_name}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()
