import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import math


def show_images(
    images: torch.Tensor | list[Image.Image | torch.Tensor],
    labels: list[str] | None = None,
    images_in_row: int = 8,
    title: str | None = None,
    size: int = 128,
):
    """Показывает изображения в виде сетки.

    Args:
        images (torch.Tensor): Изображения в формате тензора.
        labels (list[str] | None, optional): Список меток для изображений. Defaults to None.
        number_rows (int, optional): Количество изображений в строке. Defaults to 8.
        title (str | None, optional): Заголовок для отображения. Defaults to None.
        size (int, optional): Размер изображений после изменения размера. Defaults to 128.
    """

    if isinstance(images, list):
        # Преобразуем список изображений PIL в тензор
        input_images: list[torch.Tensor] = []
        for img in images:
            if not isinstance(img, (Image.Image, torch.Tensor)):
                raise ValueError(
                    f"Все элементы списка должны быть изображениями PIL или тензорами. Получено: {type(img)}"
                )
            elif isinstance(img, Image.Image):
                # Преобразуем PIL изображение в тензор
                input_images.append(F.to_tensor(img))
            else:
                input_images.append(img)

        tensor_images = torch.stack(input_images)  # type: ignore

    elif isinstance(images, torch.Tensor):
        tensor_images = images
    else:
        raise ValueError(
            f"Не верный формат входных данных. Ожидается список изображений PIL или тензоров. Получено: {type(images)}"
        )

    # Увеличиваем изображения
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized: list[torch.Tensor] = [
        resize_transform(img) for img in tensor_images
    ]

    # Создаем сетку изображений
    rows = math.ceil(tensor_images.size(0) / images_in_row)
    fig, axes = plt.subplots(
        rows, images_in_row, figsize=(images_in_row * size / 64, rows * size / 64)
    )
    axes = np.array(axes).reshape(-1)

    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis("off")
        if labels is not None:
            axes[i].set_title(labels[i])

    # Убираем лишние пустые ячейки
    for j in range(len(images_resized), len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
