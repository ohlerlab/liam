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

from ._mymodule import LiamVAE, LiamVAE_ADT
from ._trainingplans import TrainingPlanLiam


logger = logging.getLogger(__name__)


class Liam(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Liam (**L**\ everaging **i**\ nformation **a**\ cross **m**\ odalities) - model for vertical (derives a joint
    low-dimensional embedding informed by both modalities) and horizontal (batch integration) integration of multimodal
    scRNA-seq and scATAC-seq data from the same single cell.

    Can be run as a conditional VAE (CVAE, :attr:`conditional_training=True`), or a
    batch adversarial VAE (BAVAE, :attr:`adversarial_training=True`) - under development,
    and without batch correction (set :attr:`adversarial_training=False` & :attr:`conditional_training=False` (default)).

    Employs a negative multionomial loss for the scATAC-seq reconstruction introduced by [Kopp2021]_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~liam_NeurIPS2021_challenge_reproducibility.Liam.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    adversarial_training
        If `True` an adversarial training strategy will be employed using the batch as adversarial target - under
        development.
    conditional_training
        If `True` a conditional VAE model is trained.
    use_atac_libsize
        If `True` a neural network will estimate a sample specific latent library factor for the scATAC-seq data.
    dispersion_gex
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    dispersion_atac
        One of the following:
        * ``'constant'`` - dispersion parameter is the same across all batches
        * ``'batch'`` - dispersion can differ between different batches
    rna_only
        If `True` only the scRNA-seq module of the model will be trained.
    atac_only
        If `True` only the scATAC-seq module of the model will be trained.
        Data still needs to be in the obsm field of the AnnData object.
    factor_adversarial_loss
        Factor with which the adversarial loss is multiplied with.
    no_cond_decoder
        If `True` no batch labels are fed to the decoder. Default: `False`, for development purposes only.
    **model_kwargs
        Keyword args for :class:`~liam_NeurIPS2021_challenge_reproducibility.Liam`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> liam_NeurIPS2021_challenge_reproducibility.Liam.setup_anndata(adata, batch_key="batch", chrom_acc_obsm_key="Peaks")
    >>> vae = liam_NeurIPS2021_challenge_reproducibility.Liam(adata)
    >>> vae.train()
    >>> adata.obsm["X_liam"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        adversarial_training: bool = False,
        conditional_training: bool = False,
        use_atac_libsize: bool = True,
        dispersion_gex: Literal["gene", "gene-batch"] = "gene-batch",
        dispersion_atac: Literal["constant", "batch"] = "batch",
        rna_only: bool = False,
        atac_only: bool = False,
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

        if atac_only and rna_only:
            logger.error("You can choose either rna_only or atac_only, not both.")

        # Setup library size factor priors
        elif atac_only:
            library_d_log_means, library_d_log_vars = _init_library_size_liam(
                adata, n_batch, "chromatin_accessibility"
            )
            library_log_means, library_log_vars = None, None

        elif rna_only:
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)
            library_d_log_means, library_d_log_vars = None, None

        else:
            library_d_log_means, library_d_log_vars = _init_library_size_liam(
                adata, n_batch, "chromatin_accessibility"
            )
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)

        self.module = LiamVAE(
            n_genes=0 if atac_only else self.summary_stats["n_vars"],
            n_peaks=0
            if rna_only
            else get_from_registry(self.adata, "chromatin_accessibility").shape[1],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_batch=self.summary_stats["n_batch"],
            adversarial_training=adversarial_training,
            conditional_training=conditional_training,
            use_atac_libsize=use_atac_libsize,
            dispersion_gex=dispersion_gex,
            dispersion_atac=dispersion_atac,
            rna_only=rna_only,
            atac_only=atac_only,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            library_d_log_means=library_d_log_means,
            library_d_log_vars=library_d_log_vars,
            factor_adversarial_loss=factor_adversarial_loss,
            no_cond_decoder=no_cond_decoder,
            **model_kwargs,
        )
        self._model_summary_string = "Liam model with the following parameters: " \
                                     "rna_only: {}," \
                                     "atac_only: {}, " \
                                     "adversarial_training: {}, " \
                                     "conditional_training: {}, " \
                                     "use_atac_libsize: {}, " \
                                     "dispersion_gex: {}, " \
                                     "dispersion_atac: {}, " \
                                     "n_hidden: {}, " \
                                     "n_latent: {}, " \
                                     "n_batch: {}, " \
                                     "factor_adversarial_loss: {}.".format(rna_only,
                                                            atac_only,
                                                            adversarial_training,
                                                            conditional_training,
                                                            use_atac_libsize,
                                                            dispersion_gex,
                                                            dispersion_atac,
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
        rna_only: bool = False,
        chrom_acc_obsm_key: Optional[str] = None,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_layer)s
        %(param_copy)s
        rna_only:
            If `True` the model will use only the scRNA-seq data saved in adata.X and disable the scATAC part of the model.
        chrom_acc_obsm_key
            Key in `adata.obsm` for chromatin accessibility data, required for all models except RNA only models.
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

            if not rna_only and chrom_acc_obsm_key is None:
                logger.error(
                    "You must provide a key to the chromatin accessibility data stored in adata.obsm."
                )

            if not rna_only:
                # Setup of AnnData object for chromatin accessibility specific features
                logger.info(
                    "Additionally setting up variables for chromatin accessibility related variables."
                )

                if chrom_acc_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for chromatin accessibility.".format(
                            chrom_acc_obsm_key
                        )
                    )

                if not _check_nonnegative_integers(adata.obsm[chrom_acc_obsm_key]):
                    warnings.warn(
                        "adata.obsm[{}] does not contain count data. Are you sure this is what you want?".format(
                            chrom_acc_obsm_key
                        )
                    )
                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=chrom_acc_obsm_key,
                        registry_key="chromatin_accessibility",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} chromatin features".format(
                        get_from_registry(adata, "chromatin_accessibility").shape[1]
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

            if not rna_only and chrom_acc_obsm_key is None:
                logger.error(
                    "You must provide a key to the chromatin accessibility data stored in adata.obsm."
                )

            if not rna_only:
                # Setup of AnnData object for chromatin accessibility specific features
                logger.info(
                    "Additionally setting up variables for chromatin accessibility related variables."
                )

                if chrom_acc_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for chromatin accessibility.".format(
                            chrom_acc_obsm_key
                        )
                    )

                if not _check_nonnegative_integers(adata.obsm[chrom_acc_obsm_key]):
                    warnings.warn(
                        "adata.obsm[{}] does not contain count data. Are you sure this is what you want?".format(
                            chrom_acc_obsm_key
                        )
                    )
                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=chrom_acc_obsm_key,
                        registry_key="chromatin_accessibility",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} chromatin features".format(
                        get_from_registry(adata, "chromatin_accessibility").shape[1]
                    )
                )

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = True,
        save_best: bool = True,
        early_stopping_patience: int = 50,
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
    adata: AnnData, n_batch: dict, registry_key: Literal["chromatin_accessibility", "CLR_ADT_counts"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modifies scvi.model._utils._init_library_size to return library size of the scATAC-seq or ADT data.

    Computes and returns library size.
    Parameters
    ----------
    adata
        AnnData object setup with `liam_NeurIPS2021_challenge_reproducibility`.
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

class Liam_ADT(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Liam_ADT (**L**\ everaging **i**\ nformation **a**\ cross **m**\ odalities - **a**\ ntibody-**d**\ erived
    **t**\ ag) - model for vertical (derives a joint low-dimensional embedding informed by both modalities) and horizontal
    (batch integration) integration of multimodal scRNA-seq and ADT data from the same single cell.

    Can be run as a conditional VAE (CVAE, :attr:`conditional_training=True`), or a
    batch adversarial VAE (BAVAE, :attr:`adversarial_training=True`) - under development,
    and without batch correction (set :attr:`adversarial_training=False` & :attr:`conditional_training=False` (default)).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~liam_NeurIPS2021_challenge_reproducibility.Liam.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    adversarial_training
        If `True` an adversarial training strategy will be employed using the batch as adversarial target - under
        development.
    conditional_training
        If `True` a conditional VAE model is trained.
    dispersion_gex
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    dispersion_ADT
        One of the following:
        * ``'ADT'`` - dispersion parameter of NB is constant per ADT across cells
        * ``'ADT-batch'`` - dispersion can differ between different batches
    rna_only
        If `True` only the scRNA-seq module of the model will be trained.
    ADT_only
        If `True` only the scATAC-seq module of the model will be trained. Data still needs to be in the obsm field of the AnnData object.
    factor_adversarial_loss
        Factor with which the adversarial loss is multiplied with.
    no_cond_decoder
        If `True` no batch labels are fed to the decoder. Default: `False`, for development purposes only.
    **model_kwargs
        Keyword args for :class:`~liam_NeurIPS2021_challenge_reproducibility.Liam`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> liam_NeurIPS2021_challenge_reproducibility.Liam_ADT.setup_anndata(adata, batch_key="batch", ADT_obsm_key="ADT")
    >>> vae = liam_NeurIPS2021_challenge_reproducibility.Liam_ADT(adata)
    >>> vae.train()
    >>> adata.obsm["X_liam_ADT"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        adversarial_training: bool = False,
        conditional_training: bool = False,
        dispersion_gex: Literal["gene", "gene-batch"] = "gene-batch",
        dispersion_ADT: Literal["ADT", "ADT-batch"] = "ADT-batch",
        rna_only: bool = False,
        ADT_only: bool = False,
        factor_adversarial_loss: float = 1.0,
        no_cond_decoder: bool = False,
        **model_kwargs,
    ):
        super(Liam_ADT, self).__init__(adata)

        n_batch = self.summary_stats["n_batch"]

        if not adversarial_training and not conditional_training and n_batch != 1:
            logger.warning(
                "You have disabled adversarial and conditional training but the number of registered batches "
                "is not zero. This batch information will be used in many parts of this model. This is "
                "option is only enabled/advisable for development purpose. If you want to train a VAE "
                "without batch correction repeat setting up your AnnData object with batch_key=None and "
                "rerun your model.")

        if rna_only and ADT_only:
            logger.error("You can choose either rna_only or ADT_only, not both.")

        if rna_only:
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)
            library_d_log_means, library_d_log_vars = None, None
        elif ADT_only:
            library_log_means, library_log_vars = None, None
            library_d_log_means, library_d_log_vars = _init_library_size_liam(adata, n_batch, "CLR_ADT_counts")
        else:
            library_log_means, library_log_vars = _init_library_size(adata, n_batch)
            library_d_log_means, library_d_log_vars = _init_library_size_liam(adata, n_batch, "CLR_ADT_counts")

        self.module = LiamVAE_ADT(
            n_genes=0 if ADT_only else self.summary_stats["n_vars"],
            n_ADT=0
            if rna_only
            else get_from_registry(self.adata, "CLR_ADT_counts").shape[1],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_batch=self.summary_stats["n_batch"],
            adversarial_training=adversarial_training,
            conditional_training=conditional_training,
            dispersion_gex=dispersion_gex,
            dispersion_ADT=dispersion_ADT,
            rna_only=rna_only,
            ADT_only=ADT_only,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            library_d_log_means=library_d_log_means,
            library_d_log_vars=library_d_log_vars,
            factor_adversarial_loss=factor_adversarial_loss,
            no_cond_decoder=no_cond_decoder,
            **model_kwargs,
        )
        self._model_summary_string = "Liam_ADT model with the following parameters: " \
                                     "rna_only: {}," \
                                     "ADT_only: {}, " \
                                     "adversarial_training: {}, " \
                                     "conditional_training: {}, " \
                                     "dispersion_gex: {}, " \
                                     "dispersion_ADT: {}, " \
                                     "n_hidden: {}, " \
                                     "n_latent: {}, " \
                                     "n_batch: {}, " \
                                     "factor_adversarial_loss: {}.".format(rna_only,
                                                            ADT_only,
                                                            adversarial_training,
                                                            conditional_training,
                                                            dispersion_gex,
                                                            dispersion_ADT,
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
        rna_only: bool = False,
        ADT_obsm_key: Optional[str] = None,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_layer)s
        %(param_copy)s
        rna_only:
            If `True` the model will use only the scRNA-seq data saved in adata.X and disable the scATAC part of the model.
        ADT_obsm_key
            Key in `adata.obsm` for CLR transformed ADT data, required for all models except RNA only models.
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

            if not rna_only and ADT_obsm_key is None:
                logger.error(
                    "You must provide a key to the CLR transformed ADT counts accessibility stored in adata.obsm."
                )

            if not rna_only:
                # Setup of AnnData object for chromatin accessibility specific features
                logger.info(
                    "Additionally setting up variables for chromatin accessibility related variables."
                )

                if ADT_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for CLR transformed ADT counts.".format(
                            ADT_obsm_key
                        )
                    )

                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=ADT_obsm_key,
                        registry_key="CLR_ADT_counts",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} CLR transformed ADT features".format(
                        get_from_registry(adata, "CLR_ADT_counts").shape[1]
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

            if not rna_only and ADT_obsm_key is None:
                logger.error(
                    "You must provide a key to the CLR transformed ADT count data stored in adata.obsm."
                )

            if not rna_only:
                # Setup of AnnData object for chromatin accessibility specific features
                logger.info(
                    "Additionally setting up variables for CLR transfomred ADT count related variables."
                )

                if ADT_obsm_key not in adata.obsm.keys():
                    raise KeyError(
                        "Can't find {} in adata.obsm for CLR transformed ADT counts.".format(
                            ADT_obsm_key
                        )
                    )

                else:
                    register_tensor_from_anndata(
                        adata=adata,
                        adata_attr_name="obsm",
                        adata_key_name=ADT_obsm_key,
                        registry_key="CLR_ADT_counts",
                        is_categorical=False,
                    )

                logger.info(
                    "Additionally, successfully registered {} CLR transformed ADT features".format(
                        get_from_registry(adata, "CLR_ADT_counts").shape[1]
                    )
                )

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = True,
        save_best: bool = True,
        early_stopping_patience: int = 50,
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
