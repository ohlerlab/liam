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

GitHub: YosefLab/scvi-tools-skeleton:
BSD 3-Clause License

Copyright (c) 2021, Yosef Lab
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


GitHub:  BIMSBbioinfo/bavaria:
BAVARIA - Batch-adversarial variational auto-encoder
Copyright (c) 2020, Wolfgang Kopp.
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

For the code originating from bavaria an exception was granted by the author Wolfgang Kopp to redistribute the
negative binomial loss function in this project under a BSD 3-Clause License.


GitHub: fungtion/DANN:
MIT License

Copyright (c) 2019 fungtion

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import numpy as np
import torch
from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial

from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, one_hot
from scvi.nn._base_components import FCLayers
from scvi._compat import Literal

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import Optional, Iterable


import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class LiamVAE(BaseModuleClass):
    """
    VAE for Liam.

    Parameters
    ----------
    n_mod1
        Number of input features for mod1.
    n_mod2
        Number of input features for mod2.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    adversarial_training
        If `True` an adversarial training strategy will be employed using the batch as adversarial target - under
        development.
    conditional_training
        If `True` a conditional VAE model is trained.
    use_mod1_libsize
        If `True` an neural network will estimate a sample specific latent library factor for the mod1 data.
    use_mod2_libsize
        If `True` an neural network will estimate a sample specific latent library factor for the mod2 data.
    dispersion_mod1
        One of the following:
        * ``'constant'`` - dispersion parameter is the same across all batches
        * ``'batch'`` - dispersion can differ between different batches
    dispersion_mod2
        One of the following:
        * ``'constant'`` - dispersion parameter is the same across all batches
        * ``'batch'`` - dispersion can differ between different batches
    mod1_only
        If `True` only the mod1 module of the model will be trained.
    mod2_only
        If `True` only the mod2 module of the model will be trained. Data still needs to be in the obsm field of the AnnData object.
    dropout_rate
        Dropout rate for neural networks
    use_batch_norm_encoder
        If `True` batch normalization is applied to all encoder layers except the last layer.
    use_batch_norm_decoder
        If `True` batch normalization is applied to all decoder layers except the last layer.
    use_layer_norm_encoder
        If `True` layer normalization is applied to all encoder layers except the last layer.
    use_layer_norm_decoder
        If `True` layer normalization is applied to all decoder layers except the last layer.
    library_log_means
        1 x n_batch array of means of the log library sizes for the mod1 data. Parameterizes prior on library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes for the mod1 data. Parameterizes prior on library size.
    library_d_log_means
        1 x n_batch array of means of the log library sizes for the mod1 data. Parameterizes prior on library size.
    library_d_log_vars
        1 x n_batch array of variances of the log library sizes for the mod1 data. Parameterizes prior on library size.
    no_cond_decoder
        If `True` no batch labels are fed to the decoder. Default: `False`, for development purposes only.
    """

    def __init__(
        self,
        n_mod1: int,
        n_mod2: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        adversarial_training: bool = False,
        conditional_training: bool = False,
        use_mod1_libsize: bool = True,
        use_mod2_libsize: bool = True,
        dispersion_mod1: Literal["contant", "batch"] = "batch",
        dispersion_mod2: Literal["constant", "batch"] = "batch",
        mod1_only: bool = False,
        mod2_only: bool = False,
        use_batch_norm_encoder: bool = False,
        use_batch_norm_decoder: bool = False,
        use_layer_norm_encoder: bool = True,
        use_layer_norm_decoder: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        library_d_log_means: Optional[np.ndarray] = None,
        library_d_log_vars: Optional[np.ndarray] = None,
        factor_adversarial_loss: float = 1.0,
        no_cond_decoder: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.latent_distribution = "ln"
        self.dropout_rate = dropout_rate
        self.n_mod1 = n_mod1
        self.n_mod2 = n_mod2
        self.n_hidden = n_hidden
        self.n_cat_list = [self.n_batch]

        if library_log_means is not None:
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )
        if library_d_log_means is not None:
            self.register_buffer(
                "library_d_log_means", torch.from_numpy(library_d_log_means).float()
            )
            self.register_buffer(
                "library_d_log_vars", torch.from_numpy(library_d_log_vars).float()
            )

        logger.info("Number of batches: {}.".format(self.n_cat_list))

        self.conditional_training = conditional_training
        self.use_batch_norm_encoder = use_batch_norm_encoder
        self.use_batch_norm_decoder = use_batch_norm_decoder
        self.use_layer_norm_encoder = use_layer_norm_encoder
        self.use_layer_norm_decoder = use_layer_norm_decoder
        self.dispersion_mod1 = dispersion_mod1
        self.dispersion_mod2 = dispersion_mod2
        self.use_mod1_libsize = use_mod1_libsize
        self.use_mod2_libsize = use_mod2_libsize

        # Only use adversarial training if more than one batch is registered and adversarial training selected
        if n_batch != 1 and adversarial_training:
            self.adversarial_training = adversarial_training
        elif adversarial_training:
            logger.info(
                "Adversarial training disabled as only a single batch is present."
            )
            self.adversarial_training = False
        else:
            self.adversarial_training = adversarial_training

        # Only needed for FCLayers with more than 1 layer
        self.deeply_inject_covariates_encoder = (
            True if self.conditional_training else False
        )
        self.deeply_inject_covariates_decoder = (
            True if (self.conditional_training | self.adversarial_training) else False
        )

        logger.info("Adversarial training: {}.".format(self.adversarial_training))

        logger.info("Conditional training: {}.".format(self.conditional_training))


        self.mod1_only = mod1_only
        logger.info("Mod1 only training: {}.".format(self.mod1_only))

        self.mod2_only = mod2_only
        logger.info("Mod2 only training: {}.".format(self.mod2_only))

        # Setup the parameters of your generative model, as well as your inference model
        if self.dispersion_mod1 == "constant":
            self.r_mod1 = torch.nn.Parameter(torch.randn(1))
        # Same scalar dispersion across all features per batch
        elif self.dispersion_mod1 == "batch":
            self.r_mod1 = torch.nn.Parameter(torch.randn(1, self.n_batch))
        else:
            raise ValueError(
                "Dispersion for mod1 must be one of ['constant', 'batch'],"
                " but input was {}".format(self.dispersion_mod1)
            )

        # Same scalar dispersion across all features
        if self.dispersion_mod2 == "constant":
            self.r_mod2 = torch.nn.Parameter(torch.randn(1))
        # Same scalar dispersion across all features per batch
        elif self.dispersion_mod2 == "batch":
            self.r_mod2 = torch.nn.Parameter(torch.randn(1, self.n_batch))
        else:
            raise ValueError(
                "Dispersion for mod2 must be one of ['constant', 'batch'],"
                " but input was {}".format(self.dispersion_mod2)
            )

        self.factor_adversarial_loss = factor_adversarial_loss

        if adversarial_training:
            logger.info("Adversarial loss * {}.".format(factor_adversarial_loss))

        # For development purpose
        self.no_cond_decoder = no_cond_decoder

        # Encoder networks creating latent space representation
        self.mod1_encoder = FCLayers(
            n_in=self.n_mod1,
            n_out=self.n_hidden,
            n_cat_list=self.n_cat_list if self.conditional_training else None,
            n_layers=1,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            bias=True,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        self.mod2_encoder = FCLayers(
            n_in=self.n_mod2,
            n_out=self.n_hidden,
            n_cat_list=self.n_cat_list if self.conditional_training else None,
            n_layers=1,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            bias=True,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        self.z_encoder = Encoder(
            self.n_hidden + self.n_hidden,
            self.n_latent,
            n_cat_list=self.n_cat_list if self.conditional_training else None,
            n_layers=1,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            bias=True,
        )

        self.z_encoder_one_mod = Encoder(
            self.n_hidden,
            self.n_latent,
            n_cat_list=self.n_cat_list if self.conditional_training else None,
            n_layers=1,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            bias=True,
        )

        # l encoder goes from n_input-dimensional data to 1-d library size (for mod1 features)
        self.l_encoder = Encoder(
            self.n_mod1,
            1,
            n_layers=1,
            n_cat_list=self.n_cat_list,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        # d encoder goes from n_input-dimensional data to 1-d library size (for mod2 features)
        self.d_encoder = Encoder(
            self.n_mod2,
            1,
            n_layers=1,
            n_cat_list=self.n_cat_list,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            inject_covariates=self.deeply_inject_covariates_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        # Decoder go from n_latent-dimensional space to n_input-d data
        self.mod1_decoder = DecoderFLLA(
            self.n_latent,
            self.n_mod1,
            use_libsize=self.use_mod1_libsize,
            n_cat_list=None if self.no_cond_decoder else self.n_cat_list,
            n_layers=1,
            n_hidden=self.n_hidden,
            inject_covariates=None if self.no_cond_decoder else self.deeply_inject_covariates_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
        )

        self.mod2_decoder = DecoderFLLA(
            self.n_latent,
            self.n_mod2,
            use_libsize=self.use_mod2_libsize,
            n_cat_list=None if self.no_cond_decoder else self.n_cat_list,
            n_layers=1,
            n_hidden=self.n_hidden,
            inject_covariates=None if self.no_cond_decoder else self.deeply_inject_covariates_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
        )

        # Experimental domain classifier for batch adversarial training
        self.domain_classifier_z = DecoderFLLA(
            self.n_latent,
            sum(self.n_cat_list),
            n_cat_list=None,
            n_layers=1,
            n_hidden=32,
            inject_covariates=False,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
        )

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        if self.mod2_only:
            x = None
        else:
            x = tensors[_CONSTANTS.X_KEY]

        if self.mod1_only:
            y = None
        else:
            y = tensors["mod2_counts"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        input_dict = dict(x=x, y=y, batch_index=batch_index)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        mod1_library = inference_outputs["mod1_library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        mod2_library = inference_outputs["mod2_library"]

        input_dict = {
            "z": z,
            "mod1_library": mod1_library,
            "mod2_library": mod2_library,
            "batch_index": batch_index,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, y, batch_index, n_samples=None):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        if self.mod2_only:
            ql_m, ql_v, mod1_library = None, None, None
        else:
            # log the input to the variational distribution for numerical stability
            x_ = torch.log(1 + x)

            # get variational parameters via the encoder networks
            init_mod1 = (
                self.mod1_encoder(x_.view(-1, self.n_mod1), batch_index)
                if self.conditional_training
                else self.mod1_encoder(x_.view(-1, self.n_mod1))
            )
            # Note that the non logged input is used here, I haven't tried logged input so far
            if self.use_mod1_libsize:
                ql_m, ql_v, mod1_library = self.l_encoder(
                    x, batch_index
                )
            else:
                ql_m, ql_v, mod1_library = None, None, None

        if self.mod1_only:
            qz_m, qz_v, z = (
                self.z_encoder_one_mod(init_mod1.view(-1, self.n_hidden), batch_index)
                if self.conditional_training
                else self.z_encoder_one_mod(init_mod1.view(-1, self.n_hidden))
            )
            qd_m, qd_v, mod2_library = None, None, None
        else:
            init_mod2 = (
                self.mod2_encoder(y, batch_index)
                if self.conditional_training
                else self.mod2_encoder(y)
            )

            if self.mod2_only:
                qz_m, qz_v, z = (
                    self.z_encoder_one_mod(
                        init_mod2.view(-1, self.n_hidden), batch_index
                    )
                    if self.conditional_training
                    else self.z_encoder_one_mod(init_mod2.view(-1, self.n_hidden))
                )
            else:
                qz_m, qz_v, z = (
                    self.z_encoder(
                        torch.cat((init_mod1, init_mod2), 1).view(
                            -1, self.n_hidden + self.n_hidden
                        ),
                        batch_index,
                    )
                    if self.conditional_training
                    else self.z_encoder(
                        torch.cat((init_mod1, init_mod2), 1).view(
                            -1, self.n_hidden + self.n_hidden
                        )
                    )
                )

            if self.use_mod2_libsize:
                qd_m, qd_v, mod2_library = self.d_encoder(y, batch_index)
            else:
                qd_m, qd_v, mod2_library = None, None, None

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            ql_m=ql_m,
            ql_v=ql_v,
            mod1_library=mod1_library,
            qd_m=qd_m,
            qd_v=qd_v,
            mod2_library=mod2_library,
        )

        return outputs

    @auto_move_data
    def generative(self, z, mod1_library, mod2_library, batch_index):
        """Runs the generative model."""
        if self.mod2_only:
            logits_mod1, r_mod1 = None, None
        else:
            if self.use_mod1_libsize:
                logits_mod1 = self.mod1_decoder(
                    z,
                    batch_index,
                    library=(
                            torch.exp(torch.clamp(mod1_library, min=None, max=15)) + 1e-4
                    ),
                )
            else:
                logits_mod1 = self.mod1_decoder(z, batch_index, library=False)
            
            r_mod1 = torch.clamp(torch.nn.Softplus()(self.r_mod1), min=1e-10, max=1e5)

            if self.dispersion_mod1 == "batch":
                r_mod1 = torch.squeeze(F.linear(one_hot(batch_index, self.n_batch), r_mod1))

        if self.mod1_only:
            logits_mod2, r_mod2 = None, None
        else:
            if self.use_mod2_libsize:
                logits_mod2 = self.mod2_decoder(
                    z,
                    batch_index,
                    library=(
                            torch.exp(torch.clamp(mod2_library, min=None, max=15)) + 1e-4
                    ),
                )
            else:
                logits_mod2 = self.mod2_decoder(z, batch_index, library=False)

            r_mod2 = torch.clamp(torch.nn.Softplus()(self.r_mod2), min=1e-10, max=1e5)

            if self.dispersion_mod2 == "batch":
                r_mod2 = torch.squeeze(F.linear(one_hot(batch_index, self.n_batch), r_mod2))

        if self.adversarial_training:
            reverse_z = ReverseLayerF.apply(z, 1)
            adversarial = self.domain_classifier_z(reverse_z)
        else:
            adversarial = False

        return dict(
            logits_mod1=logits_mod1,
            r_mod1=r_mod1,
            logits_mod2=logits_mod2,
            r_mod2=r_mod2,
            adversarial=adversarial,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        if self.mod2_only:
            x = None
            local_l_mean = None
            local_l_var = None
        else:
            x = tensors[_CONSTANTS.X_KEY]
            (
                local_l_mean,
                local_l_var,
            ) = self._compute_local_library_params(batch_index)

        if self.mod1_only:
            y = None
            local_d_mean = None
            local_d_var = None

        else:
            y = tensors["mod2_counts"]
            (
                local_d_mean,
                local_d_var,
            ) = self._compute_local_library_params_mod2(batch_index)

        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        qd_m = inference_outputs["qd_m"]
        qd_v = inference_outputs["qd_v"]
        logits_mod1 = generative_outputs["logits_mod1"]
        r_mod1 = generative_outputs["r_mod1"]
        logits_mod2 = generative_outputs["logits_mod2"]
        r_mod2 = generative_outputs["r_mod2"]
        adversarial = generative_outputs["adversarial"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        # Create mask that masks modalities missing in any given cell
        if (not self.mod1_only) & (not self.mod2_only):
            mask_mod1 = x.sum(axis=1) > 0
            mask_mod2 = y.sum(axis=1) > 0
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        if self.mod2_only:
            kl_divergence_l = torch.zeros(batch_index.flatten().shape).to(
                kl_divergence_z.device
            )
        else:
            if self.use_mod1_libsize:
                # For kl divergence to work, the empirical mean and variance can't be 0
                # as we mask the respective loss later we set this to the smallest possible positive value here
                # https://github.com/pytorch/pytorch/issues/74459
                eps = torch.finfo(local_l_mean.dtype).eps
    
                # For gene expression (modality 1)
                local_l_mean = local_l_mean.clamp(min=eps)
                local_l_var = local_l_var.clamp(min=eps)
    
                kl_divergence_l = kl(
                    Normal(ql_m, torch.sqrt(ql_v)),
                    Normal(local_l_mean, torch.sqrt(local_l_var)),
                ).sum(dim=1)
            else:
                kl_divergence_l = torch.zeros(batch_index.flatten().shape).to(
                    kl_divergence_l.device
                )

        if self.mod1_only:
            kl_divergence_d = torch.zeros(batch_index.flatten().shape).to(
                kl_divergence_l.device
            )
        else:
            if self.use_mod2_libsize:
                # For kl divergence to work, the empirical mean and variance can't be 0
                # as we mask the respective loss later we set this to the smallest possible positive value here
                # https://github.com/pytorch/pytorch/issues/74459
                eps = torch.finfo(local_d_mean.dtype).eps

                # For chromatin accessibility (modality 2)
                local_d_mean = local_d_mean.clamp(min=eps)
                local_d_var = local_d_var.clamp(min=eps)

                kl_divergence_d = kl(
                    Normal(qd_m, torch.sqrt(qd_v)),
                    Normal(local_d_mean, torch.sqrt(local_d_var)),
                ).sum(dim=1)
            else:
                kl_divergence_d = torch.zeros(batch_index.flatten().shape).to(
                    kl_divergence_l.device
                )

        if self.mod2_only:
            reconst_loss_mod1 = torch.zeros(batch_index.flatten().shape).to(
                kl_divergence_z.device
            )
        else:
            reconst_loss_mod1 = -self.negative_multinomial_likelihood(x, logits_mod1, r_mod1)
            
        torch.autograd.set_detect_anomaly(True)

        if self.mod1_only:
            reconst_loss_mod2 = torch.zeros(batch_index.flatten().shape).to(
                kl_divergence_z.device
            )
        else:
            reconst_loss_mod2 = -self.negative_multinomial_likelihood(y, logits_mod2, r_mod2)
        kl_local_for_warmup = kl_divergence_z
        # use mask to set contribution of missing modalities to this loss to 0
        if (not self.mod1_only) & (not self.mod2_only):
            kl_local_no_warmup = kl_divergence_l * mask_mod1 + kl_divergence_d * mask_mod2
        else:
            kl_local_no_warmup = kl_divergence_l + kl_divergence_d
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        if self.adversarial_training:
            adversarial_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                adversarial, batch_index.flatten().long()
            )
        else:
            adversarial_loss = torch.zeros(batch_index.flatten().shape).to(
                kl_divergence_z.device
            )

        # use mask to set contribution of missing modalities to respective losses to 0
        if (not self.mod1_only) & (not self.mod2_only):
            loss = torch.mean(
                reconst_loss_mod1 * mask_mod1 + reconst_loss_mod2 * mask_mod2 + weighted_kl_local + self.factor_adversarial_loss * adversarial_loss
            )
        else:
            loss = torch.mean(
                reconst_loss_mod1 + reconst_loss_mod2 + weighted_kl_local + self.factor_adversarial_loss * adversarial_loss
            )


        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_d=kl_divergence_d,
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(
            loss,
            reconst_loss_mod1,
            kl_local,
            kl_global,
            reconst_loss_mod2=reconst_loss_mod2,
            adversarial_loss=adversarial_loss,
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_d=kl_divergence_d,
        )

    def negative_multinomial_likelihood(self, targets, logits, r):
        # Loss function from BAVARIA [Kopp2021]_.
        # https://github.com/BIMSBbioinfo/bavaria/blob/0c4bf5485d7fec89abd63fbf96c896647521cf53/src/bavaria/layers.py#L12
        # line 11

        # "targets * log(p)"
        likeli = torch.sum(self.xlogy(targets, self.softmax1p(logits) + 1e-10), dim=-1)
        # "r * log(1-p)"
        likeli += torch.sum(self.xlogy(r, self.softmax1p0(logits) + 1e-10), dim=-1)
        # "lgamma(r + x)"
        likeli += torch.lgamma(r + torch.sum(targets, dim=-1))
        # "lgamma(r)"
        likeli -= torch.sum(torch.lgamma(r), dim=-1)
        return likeli

    @staticmethod
    def softmax1p(x):
        # Loss function from BAVARIA [Kopp2021]_.
        # https://github.com/BIMSBbioinfo/bavaria/blob/0c4bf5485d7fec89abd63fbf96c896647521cf53/src/bavaria/layers.py#L48
        # line 48
        # clamp to avoid +-inf values arising in mosaic integration scenario
        xmax = torch.clamp(torch.max(x, dim=-1, keepdim=True)[0], min=-88.7228, max=88.7228)
        x = x - xmax
        sp = torch.exp(x) / (
            torch.exp(-xmax) + torch.sum(torch.exp(x), dim=-1, keepdim=True)
        )
        return sp

    @staticmethod
    def softmax1p0(x):
        # Loss function from BAVARIA [Kopp2021]_.
        # https://github.com/BIMSBbioinfo/bavaria/blob/0c4bf5485d7fec89abd63fbf96c896647521cf53/src/bavaria/layers.py#L48
        # line 57
        # clamp to avoid +-inf values arising in mosaic integration scenario
        xmax = torch.clamp(torch.max(x, dim=-1, keepdim=True)[0], min=-88.7228, max=88.7228)
        x = x - xmax
        sp = torch.squeeze(torch.exp(-xmax)) / (
            torch.squeeze(torch.exp(-xmax)) + torch.sum(torch.exp(x), dim=-1)
        )
        return sp

    # More recent version of torch implements this, could be replaced
    def xlogy(self, x, y):
        z = torch.zeros(())
        if x.get_device() == 0:
            z = z.to("cuda:0")
        return x * torch.where(x == 0.0, z, torch.log(y))

    def _compute_local_library_params_mod2(self, batch_index):
        """
        Modified from scvi.module._vae function _compute_local_library_params

        Computes local library parameters for the mod2 data.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_d_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_d_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_d_log_vars
        )
        return local_library_log_means, local_library_log_vars

    def _compute_local_library_params(self, batch_index):
        """
        From scvi.moduel._vae

        Computes local library parameters.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars
class DecoderSCVINB(torch.nn.Module):
    """
    Modifies scvi.nn._base_components.DecoderSCVI to only return parameters for the negative binomial distribution
    and not ZINB.

    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output),
            torch.nn.Softmax(dim=-1),
        )

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the NB distribution of expression

        Parameters
        ----------
        dispersion
            One of the following
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the NB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = None  # self.px_dropout_decoder(px)
        # Clamp for numerical stability
        px_rate = (torch.exp(torch.clamp(library, min=None, max=15)) + 1e-4) * px_scale
        px_r = None
        return px_scale, px_r, px_rate, px_dropout


class DecoderFLLA(torch.nn.Module):
    """
    FLLA (*F*inal *l*ayer *l*inear *a*ctivation)

    Decoder returning logits from a final linear activation layer.

    When used for data modeled as negative multinomially distributed, use_modX_libsize can be used to let the learned library
    size inform the final layer output.

    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    use_libsize
        If `True` inferred library size factor will be fed to the penultimate decoder layer.
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        use_libsize: bool = False,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        if use_libsize:
            self.decoder_output = torch.nn.Sequential(
                torch.nn.Linear(n_hidden + 1, n_output, bias=True),
            )
        else:
            self.decoder_output = torch.nn.Sequential(
                torch.nn.Linear(n_hidden, n_output, bias=True),
            )

    def forward(self, z: torch.Tensor, *cat_list: int, library=False):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the negative multinomal distribution of the modeled modality features

        Parameters
        ----------
       z
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :py:class:`torch.Tensor`
            Logits for each feature of the modeled modality data.

        """
        # The decoder returns the logits that serve as input for the NM distribution or, e.g., a classifier
        if torch.is_tensor(library):
            x_hat = self.decoder_output(
                torch.cat(
                    (self.decoder(z, *cat_list, library), library), dim=1
                )
            )
        # Normal decoder with last layer being a linear activation function
        else:
            x_hat = self.decoder_output(self.decoder(z, *cat_list, library))

        return x_hat


class ReverseLayerF(torch.autograd.Function):
    """
    Gradient reversal layer as introduced in [Ganin2016]_.

    Implementation from GitHub: fungtion/DANN
    Specifically: https://github.com/fungtion/DANN/blob/476147f70bb818a63bb3461a6ecc12f97f7ab15e/models/functions.py
    """
    #  @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
