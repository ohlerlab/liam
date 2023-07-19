"""
This software is distributed under the BSD 3-Clause License included below. A copy of the license can be found in the
LICENSE file distributed with this program.

BSD 3-Clause License

Copyright (c) 2021, Pia Rautenstrauch
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


This file incorporates work covered by the following copyright and permission notices:

GitHub: YosefLab/scvi-tools:
BSD 3-Clause License

Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from inspect import getfullargspec
from typing import Optional, Union

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi._compat import Literal
from scvi.module.base import BaseModuleClass
import pytorch_lightning as pl


def _compute_kl_weight(
    epoch: int,
    step: int,
    n_epochs_kl_warmup: Optional[int],
    n_steps_kl_warmup: Optional[int],
) -> float:
    epoch_criterion = n_epochs_kl_warmup is not None
    step_criterion = n_steps_kl_warmup is not None
    if epoch_criterion:
        kl_weight = min(1.0, epoch / n_epochs_kl_warmup)
    elif step_criterion:
        kl_weight = min(1.0, step / n_steps_kl_warmup)
    else:
        kl_weight = 1.0
    return max(kl_weight, 1e-3)


class TrainingPlanLiam(pl.LightningModule):
    """
    Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.
    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/model_user_guide`.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    lr
        Learning rate used for optimization.
    weight_decay
        Weight decay used in optimizatoin.
    eps
        eps used for optimization.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`).
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        optimizer: Literal["Adam", "AdamW"] = "Adam",
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        **loss_kwargs,
    ):
        super(TrainingPlanLiam, self).__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs

        self._n_obs_training = None

        # automatic handling of kl weight
        self._loss_args = getfullargspec(self.module.loss)[0]
        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

    @property
    def n_obs_training(self):
        """
        Number of observations in the training set.

        This will update the loss kwargs for loss rescaling.
        """
        return self._n_obs_training

    @n_obs_training.setter
    def n_obs_training(self, n_obs: int):
        if "n_obs" in self._loss_args:
            self.loss_kwargs.update({"n_obs": n_obs})
        self._n_obs_training = n_obs

    def forward(self, *args, **kwargs):
        """Passthrough to `model.forward()`."""
        return self.module(*args, **kwargs)

    # Test custom loss recording
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        # pytorch lightning automatically backprops on "loss"
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        # lightning wants non loss keys detached
        return {
            "loss": scvi_loss.loss,
            "reconstruction_loss_sum": reconstruction_loss.sum().detach(),
            "kl_local_sum": scvi_loss.kl_local.sum().detach(),
            "kl_global": scvi_loss.kl_global.detach(),
            "n_obs": reconstruction_loss.shape[0],
            "reconst_loss_peaks_sum": scvi_loss.reconst_loss_peaks.sum().detach(),
            "adversarial_loss_sum": scvi_loss.adversarial_loss.sum().detach(),
            "kl_divergence_l_sum": scvi_loss.kl_divergence_l.sum().detach(),
            "kl_divergence_z_sum": scvi_loss.kl_divergence_z.sum().detach(),
            "kl_divergence_d_sum": scvi_loss.kl_divergence_d.sum().detach(),
        }

    def training_epoch_end(self, outputs):
        (
            n_obs,
            elbo,
            rec_loss,
            kl_local,
            rec_loss_peaks,
            adversarial_loss,
            kl_div_l,
            kl_div_z,
            kl_div_d,
        ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            rec_loss_peaks += tensors["reconst_loss_peaks_sum"]
            adversarial_loss += tensors["adversarial_loss_sum"]
            kl_div_l += tensors["kl_divergence_l_sum"]
            kl_div_z += tensors["kl_divergence_z_sum"]
            kl_div_d += tensors["kl_divergence_d_sum"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_train", elbo / n_obs)
        self.log("reconstruction_loss_train", rec_loss / n_obs)
        self.log("kl_local_train", kl_local / n_obs)
        self.log("kl_global_train", kl_global)
        self.log("reconstruction_loss_peaks_train", rec_loss_peaks / n_obs)
        self.log("adversarial_loss_train", adversarial_loss / n_obs)
        self.log("kl_div_l_loss_train", kl_div_l / n_obs)
        self.log("kl_div_z_loss_train", kl_div_z / n_obs)
        self.log("kl_div_d_loss_train", kl_div_d / n_obs)

    def validation_step(self, batch, batch_idx):
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
            "reconst_loss_peaks_sum": scvi_loss.reconst_loss_peaks.sum(),
            "adversarial_loss_sum": scvi_loss.adversarial_loss.sum(),
            "kl_divergence_l_sum": scvi_loss.kl_divergence_l.sum(),
            "kl_divergence_z_sum": scvi_loss.kl_divergence_z.sum(),
            "kl_divergence_d_sum": scvi_loss.kl_divergence_d.sum(),
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        (
            n_obs,
            elbo,
            rec_loss,
            kl_local,
            rec_loss_peaks,
            adversarial_loss,
            kl_div_l,
            kl_div_z,
            kl_div_d,
        ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            rec_loss_peaks += tensors["reconst_loss_peaks_sum"]
            adversarial_loss += tensors["adversarial_loss_sum"]
            kl_div_l += tensors["kl_divergence_l_sum"]
            kl_div_z += tensors["kl_divergence_z_sum"]
            kl_div_d += tensors["kl_divergence_d_sum"]

        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", kl_global)
        self.log("reconstruction_loss_peaks_validation", rec_loss_peaks / n_obs)
        self.log("adversarial_loss_validation", adversarial_loss / n_obs)
        self.log("kl_div_l_loss_validation", kl_div_l / n_obs)
        self.log("kl_div_z_loss_validation", kl_div_z / n_obs)
        self.log("kl_div_d_loss_validation", kl_div_d / n_obs)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        if self.optimizer_name == "Adam":
            optim_cls = torch.optim.Adam
        elif self.optimizer_name == "AdamW":
            optim_cls = torch.optim.AdamW
        else:
            raise ValueError("Optimizer not understood.")
        optimizer = optim_cls(
            params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": scheduler,
                    "monitor": self.lr_scheduler_metric,
                },
            )
        return config

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        return _compute_kl_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_kl_warmup,
            self.n_steps_kl_warmup,
        )
