# Copyright (c) Facebook, Inc. and its affiliates.

import os
import warnings

from mmf.common.registry import registry
from mmf.datasets.builders.propaganda.dataset import (
    PropagandaTask3FeaturesDataset,
    PropagandaTask3Dataset,
)
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.configuration import get_mmf_env
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path


@registry.register_builder("propaganda")
class PropagandaBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="propaganda",
        dataset_class=PropagandaTask3Dataset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = PropagandaTask3Dataset

    @classmethod
    def config_path(self):
        return "configs/datasets/propaganda/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        config = config

        if config.use_features:
            self.dataset_class = PropagandaTask3FeaturesDataset

        self.dataset = super().load(config, dataset_type, *args, **kwargs)

        return self.dataset

    def build(self, config, *args, **kwargs):
        # self.data_folder = os.path.join(
            # get_mmf_root(), config.data_dir
        # )
        super().build(config, *args, **kwargs)

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(self.dataset_name + "_num_final_outputs", 2)
