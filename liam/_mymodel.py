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

"""


import logging
import warnings
from typing import List, Optional, Union, Sequence, Tuple

import numpy as np
import torch

from anndata import AnnData
from scipy.sparse import csr_matrix, vstack
from scvi import _CONSTANTS
from scvi._compat import Literal
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils._docstrings import setup_anndata_dsp
from scvi.data._anndata import _setup_anndata
from scvi.data import register_tensor_from_anndata, get_from_registry
from scvi.data._utils import _check_nonnegative_integers
from scvi.model._utils import _init_library_size

from scvi.dataloaders import DataSplitter
from scvi.train import TrainRunner
from scvi.train._callbacks import SaveBestState

from ._mymodule import LiamVAE
from ._trainingplans import TrainingPlanLiam


logger = logging.getLogger(__name__)


class Liam(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Liam (**L**\ everaging **i**\ nformation **a**\ cross **m**\ odalities) is a model for integrating multimodal
    data from the same single cell. It simultaneously performs vertical (deriving a joint
    low-dimensional embedding informed by both modalities) and horizontal integration (batch integration).
    Liam can also integrate paired with unimodal datasets (mosaic integration).

    By default, liam utilizes a tunable batch adversarial training strategy known as batch adversarial VAE (BAVAE).
    You can enable this with the parameter :attr:`adversarial_training=True` (default setting).

    The adversarial training strategy is tunable via :attr:`factor_adversarial_loss=1.0` (default).

    Liam can also be run as a conditional VAE (CVAE) by setting :attr:`conditional_training=True` and disabling
    adversarial training (:attr:`adversarial_training=False`). Alternatively, liam can be run without batch correction
    by setting both :attr:`adversarial_training=False` and :attr:`conditional_training=False`.

    In this experimental version liam employs a negative multinomial loss as introduced by [Kopp2021]_ for both modalities.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~liam.Liam.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    adversarial_training
        If `True` an adversarial training strategy will be employed using the batch as adversarial target - under
        development.
    conditional_training
        If `True` a conditional VAE model is trained.
    use_mod1_libsize
        If `True` a neural network will estimate a sample specific latent library factor for the mod1 data.
    use_mod2_libsize
        If `True` a neural network will estimate a sample specific latent library factor for the mod2 data.
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
        If `True` only the mod2 module of the model will be trained.
        Data still needs to be in the obsm field of the AnnData object.
    factor_adversarial_loss
        Factor with which the adversarial loss is multiplied with.
    no_cond_decoder
        If `True` no batch labels are fed to the decoder. Default: `False`, for development purposes only.
    **model_kwargs
        Keyword args for :class:`~liam.Liam`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> liam.Liam.setup_anndata(adata, batch_key="batch", mod2_obsm_key="mod2_data")
    >>> vae = liam.Liam(adata)
    >>> vae.train()
    >>> adata.obsm["X_liam"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 20,
        adversarial_training: bool = True,
        conditional_training: bool = False,
        use_mod1_libsize: bool = True,
        use_mod2_libsize: bool = True,
        dispersion_mod1: Literal["constant", "batch"] = "batch",
        dispersion_mod2: Literal["constant", "batch"] = "batch",
        mod1_only: bool = False,
        mod2_only: bool = False,
        factor_adversarial_loss: float = 1.0,
        no_cond_decoder: bool = False,
        **model_kwargs,
    ):
        super(Liam, self).__init__(adata)

        n_batch = self.summary_stats["n_batch"]

        if not adversarial_training and not conditional_training and n_batch != 1:
            logger.warning("You have disabled adversarial and conditional training but the number of registered batches "
                           "is not zero. This batch information will be used in many parts of this model. This is "
                           "option is only enabled/advisable for development purpose. If you want to train a VAE "
                           "without batch correction repeat setting up your AnnData object with batch_key=None and "
                           "rerun your model.")

        if mod2_only and mod1_only:
            logger.error("You can choose either mod1_only or mod2_only, not both.")

        # Setup library size factor priors
        elif mod2_only:
            library_d_log_means, library_d_log_vars = _init_library_size_liam(
                adata, n_batch, "mod2_counts"
            )
            library_log_means, library_log_vars = None, None

        elif mod1_only:
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)
            library_d_log_means, library_d_log_vars = None, None

        else:
            library_d_log_means, library_d_log_vars = _init_library_size_liam(
                adata, n_batch, "mod2_counts"
            )
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)

        self.module = LiamVAE(
            n_mod1=0 if mod2_only else self.summary_stats["n_vars"],
            n_mod2=0
            if mod1_only
            else get_from_registry(self.adata, "mod2_counts").shape[1],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_batch=self.summary_stats["n_batch"],
            adversarial_training=adversarial_training,
            conditional_training=conditional_training,
            use_mod1_libsize=use_mod1_libsize,
            use_mod2_libsize=use_mod2_libsize,
            dispersion_mod1=dispersion_mod1,
            dispersion_mod2=dispersion_mod2,
            mod1_only=mod1_only,
            mod2_only=mod2_only,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            library_d_log_means=library_d_log_means,
            library_d_log_vars=library_d_log_vars,
            factor_adversarial_loss=factor_adversarial_loss,
            no_cond_decoder=no_cond_decoder,
            **model_kwargs,
        )
        self._model_summary_string = "Liam model with the following parameters: " \
                                     "mod1_only: {}," \
                                     "mod2_only: {}, " \
                                     "adversarial_training: {}, " \
                                     "conditional_training: {}, " \
                                     "use_mod1_libsize: {}, " \
                                     "use_mod2_libsize: {}, " \
                                     "dispersion_mod1: {}, " \
                                     "dispersion_mod2: {}, " \
                                     "n_hidden: {}, " \
                                     "n_latent: {}, " \
                                     "n_batch: {}, " \
                                     "factor_adversarial_loss: {}.".format(mod1_only,
                                                            mod2_only,
                                                            adversarial_training,
                                                            conditional_training,
                                                            use_mod1_libsize,
                                                            use_mod2_libsize,
                                                            dispersion_mod1,
                                                            dispersion_mod2,
                                                            n_hidden,
                                                            n_latent,
                                                            n_batch,
                                                            factor_adversarial_loss)
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @staticmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        copy: bool = False,
        mod1_only: bool = False,
        mod2_obsm_key: Optional[str] = None,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_layer)s
        %(param_copy)s
        mod1_only:
            If `True` the model will use only the mod1 data saved in adata.X and disable the mod2 part of the model.
        mod2_obsm_key
            Key in `adata.obsm` for mod2 data, required for all models except mod1 only models.
        -------
        %(returns)s
        """

        # Setup of AnnData object analogous to scvi for gene expression and covariates
        logger.info(
            "Setting up anndata object using scVI for gene expression related variables."
        )

        if copy:
            adata = _setup_anndata(
                adata,
                batch_key=batch_key,
                layer=layer,
                copy=copy,
            )

            if not mod1_only and mod2_obsm_key is None:
                logger.error(
                    "You must provide a key to the mod2 data stored in adata.obsm."
                )

            if not mod1_only:
                # Setup of AnnData object for mod2 specific features
                logger.info(
                    "Additionally setting up variables for mod2 related variables."
                )

                if mod2_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for mod2.".format(
                            mod2_obsm_key
                        )
                    )

                if not _check_nonnegative_integers(adata.obsm[mod2_obsm_key]):
                    warnings.warn(
                        "adata.obsm[{}] does not contain count data. Are you sure this is what you want?".format(
                            mod2_obsm_key
                        )
                    )
                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=mod2_obsm_key,
                        registry_key="mod2_counts",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} mod2 features".format(
                        get_from_registry(adata, "mod2_counts").shape[1]
                    )
                )

            return adata

        else:
            _setup_anndata(
                adata,
                batch_key=batch_key,
                layer=layer,
                copy=copy,
            )

            if not mod1_only and mod2_obsm_key is None:
                logger.error(
                    "You must provide a key to the mod2 data stored in adata.obsm."
                )

            if not mod1_only:
                # Setup of AnnData object for mod2 specific features
                logger.info(
                    "Additionally setting up variables for mod2 related variables."
                )

                if mod2_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for mod2.".format(
                            mod2_obsm_key
                        )
                    )

                if not _check_nonnegative_integers(adata.obsm[mod2_obsm_key]):
                    warnings.warn(
                        "adata.obsm[{}] does not contain count data. Are you sure this is what you want?".format(
                            mod2_obsm_key
                        )
                    )
                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=mod2_obsm_key,
                        registry_key="mod2_counts",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} mod2 features".format(
                        get_from_registry(adata, "mod2_counts").shape[1]
                    )
                )

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.95,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = True,
        save_best: bool = True,
        early_stopping_patience: int = 10,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Modifies train method from class UnsupervisedTrainingMixin() of scvi.model.base._training_mixin.py class to use
        custom trainingsplan TrainingPlanLiam.

        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        save_best
            Save the best model state with respect to the validation loss (default), or use the final
            state in the training procedure.
        early_stopping_patience
            How many epochs to wait for improvement before early stopping/
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        print("Using Liam train() method.")
        print("trainer_kwargs", trainer_kwargs)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        training_plan = TrainingPlanLiam(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        if save_best:
            if "callbacks" not in trainer_kwargs.keys():
                trainer_kwargs["callbacks"] = []
            trainer_kwargs["callbacks"].append(SaveBestState(monitor="validation_loss"))

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping_monitor="validation_loss",
            early_stopping_patience=early_stopping_patience,
            **trainer_kwargs,
        )
        return runner()

    @staticmethod
    def get_elbo():
        """Not implemented."""
        return "Not implemented."

    @staticmethod
    def get_marginal_ll():
        """Not implemented."""
        return "Not implemented."

    @staticmethod
    def get_reconstruction_error():
        """Not implemented."""
        return "Not implemented."


def _init_library_size_liam(
    adata: AnnData, n_batch: dict, registry_key: Literal["mod2_counts"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modifies scvi.model._utils._init_library_size to return library size of the mod2 data.

    Computes and returns library size.
    Parameters
    ----------
    adata
        AnnData object setup with `liam`.
    n_batch
        Number of batches.
    Returns
    -------
    type
        Tuple of two 1 x n_batch ``np.ndarray`` containing the means and variances
        of library size in each batch in adata.
        If a certain batch is not present in the adata, the mean defaults to 0,
        and the variance defaults to 1. These defaults are arbitrary placeholders which
        should not be used in any downstream computation.
    """
    data = get_from_registry(adata, registry_key)
    batch_indices = get_from_registry(adata, _CONSTANTS.BATCH_KEY)

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[
            idx_batch.nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`"
            )

        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)
