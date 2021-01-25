from typing import List, Optional, Union

import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import LightningDataModule, LightningModule

from utilities import all_elements_identical


class MaskRCNNTrainer(pl.Trainer):
    def on_fit_start(self):
        """Functionality of overridden method with added check, whether the train_dataloader,
        val_dataloaders, datamodule have the same map_label_to_class_name property. If yes, then
        the common map_label_to_class_name is stored in the model, if its number of classes matches
        that of the model.
        """
        super().on_fit_start()

        class_mapping = self._check_class_mappings(
            self.datamodule, self.train_dataloader, self.val_dataloaders
        )
        self.model.map_label_to_class_name = class_mapping

        num_classes_data = len(class_mapping) if class_mapping is not None else None
        num_classes_model = getattr(self.model, "num_classes", None)

        if num_classes_model is not None and num_classes_data is not None:
            assert (
                num_classes_model == num_classes_data
            ), "Model and data have unequal numbers of classes."

    def tune(
        self,
        model: LightningModule,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloaders: Optional[
            Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]
        ] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        """Functionality of overridden method with added check, whether the supplied
        train_dataloader, val_dataloaders and/or datamodule have the same map_label_to_class_name
        property as the model.
        """
        self._check_class_mappings(datamodule, train_dataloader, val_dataloaders, model=model)

        super().tune(model, train_dataloader, val_dataloaders, datamodule)

    def test(
        self,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[
            Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]
        ] = None,
        ckpt_path: Optional[str] = "best",
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
        """Functionality of overridden method with added check, whether the supplied
        test_dataloaders have the same map_label_to_class_name property as the model.
        """
        self._check_class_mappings(datamodule, test_dataloaders=test_dataloaders, model=model)

        return super().test(model, test_dataloaders, ckpt_path, verbose, datamodule)

    @staticmethod
    def _check_class_mappings(
        datamodule: Optional[LightningDataModule] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloaders: Optional[
            Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]
        ] = None,
        test_dataloaders: Optional[
            Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]
        ] = None,
        model: Optional[LightningModule] = None,
    ) -> dict:
        """Checks if the map_label_to_class_name properties of all supplied dataloaders, the
            datamodule and the model are identical.

        :param datamodule: pytorch lightning data module
        :param train_dataloader: pytorch dataloader
        :param val_dataloaders: list of pytorch dataloaders
        :param test_dataloaders: pytorch dataloader
        :param model: pytorch lightning model
        :return: Common map_label_to_class_name property.
        """
        class_mappings = []
        if val_dataloaders is not None:
            class_mappings += [
                getattr(dl.dataset, "map_label_to_class_name", None) for dl in val_dataloaders
            ]

        if train_dataloader is not None:
            class_mappings.append(
                getattr(train_dataloader.dataset, "map_label_to_class_name", None)
            )

        if test_dataloaders is not None:
            class_mappings += [
                getattr(dl.dataset, "map_label_to_class_name", None) for dl in test_dataloaders
            ]

        if datamodule is not None:
            class_mappings.append(getattr(datamodule, "map_label_to_class_name", None))

        if model is not None:
            class_mappings.append(getattr(model, "map_label_to_class_name", None))

        error_message = (
            "All datasets must have identical map_label_to_class_name properties."
            if model is None
            else "All datasets and the model must have identical map_label_to_class_name properties."
        )

        assert all_elements_identical(class_mappings), error_message

        return class_mappings[0]
