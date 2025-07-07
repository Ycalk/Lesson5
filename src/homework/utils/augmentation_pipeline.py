from dataclasses import dataclass
from torch import nn
import torch
from enum import StrEnum
import torchvision.transforms.functional as F
from PIL import Image


class InputType(StrEnum):
    IMAGE = "image"
    TENSOR = "tensor"


@dataclass(frozen=True)
class Augmentation:
    """Класс для представления аугментации."""

    name: str
    augmentation: nn.Module
    input_type: InputType = InputType.IMAGE

    def cast(self, input: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
        """Приводит входные данные к нужному типу для применения аугментации.

        Args:
            input (Image.Image | torch.Tensor): Входные данные, которые нужно привести к нужному типу.

        Returns:
            Image.Image | torch.Tensor: Входные данные в нужном формате.
        """
        if self.input_type == InputType.IMAGE and isinstance(input, torch.Tensor):
            return F.to_pil_image(input)
        elif self.input_type == InputType.TENSOR and isinstance(input, Image.Image):
            return F.to_tensor(input)
        return input


class AugmentationPipeline:
    """Класс для создания пайплайна аугментаций."""

    def __init__(self) -> None:
        self._augmentations: list[Augmentation] = []

    def add_augmentation(self, augmentation: Augmentation) -> None:
        """Добавляет аугментацию в пайплайн.

        Args:
            augmentation (Augmentation): Аугментация, которую нужно добавить в пайплайн.
        """
        self._augmentations.append(augmentation)

    def get_augmentations(self) -> list[Augmentation]:
        """Возвращает список аугментаций в пайплайне.

        Returns:
            list[Augmentation]: Список аугментаций.
        """
        return self._augmentations

    def apply(self, image: torch.Tensor | Image.Image) -> torch.Tensor:
        """Применяет все аугментации к изображению.

        Args:
            image (torch.Tensor | Image.Image): Изображение, к которому нужно применить аугментации.

        Returns:
            torch.Tensor: Изображение после применения всех аугментаций.
        """
        for augmentation in self._augmentations:
            try:
                image = augmentation.augmentation(augmentation.cast(image))
            except Exception as e:
                raise RuntimeError(
                    f"Ошибка при применении аугментации '{augmentation.name}': {e}"
                ) from e

        return image if isinstance(image, torch.Tensor) else F.to_pil_image(image)
