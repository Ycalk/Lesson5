import torch
from torch import nn
import random
import numpy as np
import cv2
from PIL import Image, ImageOps


class AddGaussianNoise(nn.Module):
    """Добавляет гауссов шум к изображению."""

    def __init__(self, mean: float = 0, std: float = 0.1):
        """Инициализация класса AddGaussianNoise.

        Args:
            mean (int, optional): Среднее значение шума. Defaults to 0.
            std (float, optional): Стандартное отклонение шума. Defaults to 0.1.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class RandomErasingCustom(nn.Module):
    """Случайно затирает прямоугольную область изображения."""

    def __init__(self, p: float = 0.5, scale: tuple[float, float] = (0.02, 0.2)):
        """Инициализация класса RandomErasingCustom.

        Args:
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
            scale (tuple[float, float], optional): Диапазон масштабов затираемой области. Defaults to (0.02, 0.2).
        """
        super().__init__()
        self.p = p
        self.scale = scale

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        _, h, w = tensor.shape
        area = h * w
        erase_area = random.uniform(*self.scale) * area
        erase_w = int(np.sqrt(erase_area))
        erase_h = int(erase_area // erase_w)
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        tensor[:, y : y + erase_h, x : x + erase_w] = 0

        return tensor


class CutOut(nn.Module):
    """Вырезает случайную прямоугольную область из изображения."""

    def __init__(self, p: float = 0.5, size: tuple[int, int] = (16, 16)):
        """Инициализация класса CutOut.

        Args:
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
            size (tuple[int, int], optional): Размер вырезаемой области (высота, ширина). Defaults to (16, 16).
        """
        super().__init__()
        self.p = p
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        _, h, w = tensor.shape
        cut_h, cut_w = self.size
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)
        tensor[:, y : y + cut_h, x : x + cut_w] = 0

        return tensor


class Solarize(nn.Module):
    """Инвертирует пиксели выше порога."""

    def __init__(self, threshold: float = 128):
        """Инициализация класса Solarize.

        Args:
            threshold (float, optional): Порог для инвертирования пикселей. Defaults to 128.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        img_np = tensor.numpy()
        mask = img_np > self.threshold / 255.0
        img_np[mask] = 1.0 - img_np[mask]

        return torch.from_numpy(img_np)


class Posterize(nn.Module):
    """Уменьшает количество бит на канал."""

    def __init__(self, bits: int = 4):
        """Инициализация класса Posterize.

        Args:
            bits (int, optional): Количество бит на канал. Defaults to 4.
        """
        super().__init__()
        self.bits = bits

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        img_np = tensor.numpy()
        factor = 2 ** (8 - self.bits)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = (img_np // factor) * factor
        return torch.from_numpy(img_np.astype(np.float32) / 255.0)


class AutoContrast(nn.Module):
    """Автоматически улучшает контраст изображения."""

    def __init__(self, p: float = 0.5):
        """ Инициализация класса AutoContrast.

        Args:
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
        """
        super().__init__()
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        img_np = tensor.numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = ImageOps.autocontrast(img_pil)
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        return torch.from_numpy(img_np.transpose(2, 0, 1))


class ElasticTransform(nn.Module):
    """Эластичная деформация изображения."""

    def __init__(self, p: float = 0.5, alpha: int = 1, sigma: int = 50):
        """ Инициализация класса ElasticTransform.

        Args:
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
            alpha (int, optional): Параметр, определяющий степень деформации. Defaults to 1.
            sigma (int, optional): Стандартное отклонение гауссовского фильтра для сглаживания смещений. Defaults to 50.
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        img_np = tensor.numpy().transpose(1, 2, 0)
        h, w = img_np.shape[:2]

        # Создаем случайные смещения
        dx = np.random.randn(h, w) * self.alpha
        dy = np.random.randn(h, w) * self.alpha

        # Сглаживаем смещения
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        # Применяем деформацию
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x + dx
        y = y + dy

        # Нормализуем координаты
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        # Применяем трансформацию
        img_deformed = cv2.remap(
            img_np, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR
        )
        return torch.from_numpy(img_deformed.transpose(2, 0, 1))


class MixUp(nn.Module):
    """Смешивает два изображения."""

    def __init__(self, second_image: torch.Tensor, p: float = 0.5, alpha: float = 0.2):
        """ Инициализация класса MixUp.

        Args:
            second_image (torch.Tensor): Второе изображение для смешивания.
            p (float, optional): Вероятность применения аугментации. Defaults to 0.5.
            alpha (float, optional): Параметр для бета-распределения, определяющий степень смешивания. Defaults to 0.2.
        """
        super().__init__()
        self.second_image = second_image
        self.p = p
        self.alpha = alpha

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * tensor + (1 - lam) * self.second_image
