import torch
from torch import nn
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import random
from PIL import ImageFilter


class RandomGaussianBlur(nn.Module):
    """Применяет случайное размытие Гаусса к изображению с заданной вероятностью и радиусом."""

    def __init__(
        self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        """Инициализация класса RandomGaussianBlur.

        Args:
            p (float, optional): Вероятность применения размытия. Defaults to 0.5.
            radius_min (float, optional): Минимальный радиус размытия. Defaults to 0.1.
            radius_max (float, optional): Максимальный радиус размытия. Defaults to 2.0.
        """
        super().__init__()
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        img = F.to_pil_image(tensor)
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))


class RandomPerspectiveTransform(nn.Module):
    """Применяет случайное перспективное искажение к изображению с заданной вероятностью и масштабом искажения."""

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        """Инициализация класса RandomPerspectiveTransform.

        Args:
            distortion_scale (float, optional): Масштаб искажения. Defaults to 0.5.
            p (float, optional): Вероятность применения искажения. Defaults to 0.5.
        """
        super().__init__()
        self.distortion_scale = distortion_scale
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        _, height, width = F.get_dimensions(tensor)
        start_points, endpoints = transforms.RandomPerspective.get_params(
            width, height, self.distortion_scale
        )
        return F.perspective(tensor, start_points, endpoints)


class RandomBrightnessContrast(nn.Module):
    """Применяет случайную коррекцию яркости и контрастности к изображению с заданной вероятностью."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        """Инициализация класса RandomBrightnessContrast.

        Args:
            brightness (float, optional): Яркость, на которую будет изменено изображение. Defaults to 0.2.
            contrast (float, optional): Контрастность, на которую будет изменено изображение. Defaults to 0.2.
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        img = F.adjust_brightness(
            tensor, 1 + random.uniform(-self.brightness, self.brightness)
        )
        return F.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast))
