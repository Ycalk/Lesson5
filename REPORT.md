# Домашнее задание к уроку 5: Аугментации и работа с изображениями

## Задание 1: Стандартные аугментации torchvision

### Гароу

![Гароу](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%93%D0%B0%D1%80%D0%BE%D1%83.png)

### Генос

![Генос](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%93%D0%B5%D0%BD%D0%BE%D1%81.png)

### Сайтама

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%A1%D0%B0%D0%B9%D1%82%D0%B0%D0%BC%D0%B0.png)

### Соник

![Соник](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%A1%D0%BE%D0%BD%D0%B8%D0%BA.png)

### Татсумаки

![Татсумаки](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%A2%D0%B0%D1%82%D1%81%D1%83%D0%BC%D0%B0%D0%BA%D0%B8.png)

### Фубики

![Фубики](https://github.com/Ycalk/Lesson5/raw/main/results/standard_augmentations/%D0%A4%D1%83%D0%B1%D0%B8%D0%BA%D0%B8.png)

---

## Задание 2: Кастомные аугментации

### Гароу

#### Кастомные аугментации

![Гароу](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%93%D0%B0%D1%80%D0%BE%D1%83.png)

#### Экстра аугментации

![Гароу](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%93%D0%B0%D1%80%D0%BE%D1%83.png)

### Генос

#### Кастомные аугментации

![Генос](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%93%D0%B5%D0%BD%D0%BE%D1%81.png)

#### Экстра аугментации

![Генос](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%93%D0%B5%D0%BD%D0%BE%D1%81.png)

### Сайтама

#### Кастомные аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%A1%D0%B0%D0%B9%D1%82%D0%B0%D0%BC%D0%B0.png)

#### Экстра аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%A1%D0%B0%D0%B9%D1%82%D0%B0%D0%BC%D0%B0.png)

### Соник

#### Кастомные аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%A1%D0%BE%D0%BD%D0%B8%D0%BA.png)

#### Экстра аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%A1%D0%BE%D0%BD%D0%B8%D0%BA.png)

### Татсумаки

#### Кастомные аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%A2%D0%B0%D1%82%D1%81%D1%83%D0%BC%D0%B0%D0%BA%D0%B8.png)

#### Экстра аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%A2%D0%B0%D1%82%D1%81%D1%83%D0%BC%D0%B0%D0%BA%D0%B8.png)

### Фубики

#### Кастомные аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/custom_augmentations/%D0%A4%D1%83%D0%B1%D0%B8%D0%BA%D0%B8.png)

#### Экстра аугментации

![Сайтама](https://github.com/Ycalk/Lesson5/raw/main/results/extra_augmentations/%D0%A4%D1%83%D0%B1%D0%B8%D0%BA%D0%B8.png)

### Сравнение

Кастомные аугментации в основном не много искажают изображение: меняют персепективу, положение, не много изменяют цветы. В то время как экстра аугментации (которые были на лекции) довольно сильно меняют изображение, особенно если применять их все.

---

## Задание 3: Анализ датасета

### Анализ датасета train

#### Сравнение классов ввиде таблицы

| Класс     | Кол-во изображений | Ширина (мин.) | Ширина (макс.) | Ширина (сред.) | Высота (мин.) | Высота (макс.) | Высота (сред.) |
|-----------|--------------------|---------------|----------------|----------------|---------------|----------------|----------------|
| Гароу     | 30                 | 246           | 735            | 538.43         | 246           | 889            | 514.63         |
| Генос     | 30                 | 270           | 720            | 550.17         | 363           | 1070           | 673.90         |
| Сайтама   | 30                 | 375           | 736            | 559.33         | 366           | 1308           | 669.27         |
| Соник     | 30                 | 210           | 604            | 525.87         | 240           | 1076           | 609.93         |
| Татсумаки | 30                 | 267           | 736            | 527.20         | 267           | 1308           | 635.83         |
| Фубуки    | 30                 | 286           | 736            | 532.37         | 353           | 1104           | 637.77         |
| —         | —                  | —             | —              | —              | —             | —              | —              |
| **Итог**  | **180**            | **210**       | **736**        | **538.89**     | **240**       | **1308**       | **623.56**     |

#### Анализ ввиде диаграмм

![Количество изображений в каждом классе](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/train_sample/%D0%9A%D0%BE%D0%BB%D0%B8%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%BE%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%B2%20%D0%BA%D0%B0%D0%B6%D0%B4%D0%BE%D0%BC%20%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B5.png)

![Распределение размеров изображений](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/train_sample/%D0%A0%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BE%D0%B2%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9.png)

![Гистограммы распределения размеров изображений](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/train_sample/%D0%93%D0%B8%D1%81%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B%20%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F%20%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BE%D0%B2%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9.png)

### Анализ датасета test

#### Сравнение классов ввиде таблицы

| Класс     | Кол-во изображений | Ширина (мин.) | Ширина (макс.) | Ширина (сред.) | Высота (мин.) | Высота (макс.) | Высота (сред.) |
|-----------|---------------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Гароу     | 100                 | 225            | 736            | 533.11         | 274            | 1002           | 567.82         |
| Генос     | 100                 | 236            | 736            | 546.30         | 240            | 1082           | 640.35         |
| Сайтама   | 100                 | 300            | 736            | 549.31         | 305            | 1308           | 655.14         |
| Соник     | 100                 | 225            | 735            | 531.78         | 235            | 1002           | 584.36         |
| Татсумаки | 100                 | 250            | 736            | 560.78         | 233            | 1308           | 654.07         |
| Фубуки    | 100                 | 220            | 736            | 564.73         | 220            | 1308           | 684.16         |
| —         | —                   | —              | —              | —              | —              | —              | —              |
| **Итог**  | **600**             | **220**        | **736**        | **547.67**     | **220**        | **1308**       | **630.98**     |

#### Анализ ввиде диаграмм

![Количество изображений в каждом классе](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/test_sample/%D0%9A%D0%BE%D0%BB%D0%B8%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%BE%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%B2%20%D0%BA%D0%B0%D0%B6%D0%B4%D0%BE%D0%BC%20%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B5.png)

![Распределение размеров изображений](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/test_sample/%D0%A0%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BE%D0%B2%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9.png)

![Гистограммы распределения размеров изображений](https://github.com/Ycalk/Lesson5/raw/main/results/dataset_analysis/test_sample/%D0%93%D0%B8%D1%81%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B%20%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F%20%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BE%D0%B2%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9.png)

---

## Задание 4: Pipeline аугментации

### Применение Light конфигурации

![Light конфигурация](https://github.com/Ycalk/Lesson5/raw/main/results/augmentation_pipeline/Light%20%D0%BA%D0%BE%D0%BD%D1%84%D0%B8%D0%B3%D1%83%D1%80%D0%B0%D1%86%D0%B8%D1%8F.png)

### Применение Medium конфигурации

![Medium конфигурация](https://github.com/Ycalk/Lesson5/raw/main/results/augmentation_pipeline/Medium%20%D0%BA%D0%BE%D0%BD%D1%84%D0%B8%D0%B3%D1%83%D1%80%D0%B0%D1%86%D0%B8%D1%8F.png)

### Применение Heavy конфигурации

![Medium конфигурация](https://github.com/Ycalk/Lesson5/raw/main/results/augmentation_pipeline/Heavy%20%D0%BA%D0%BE%D0%BD%D1%84%D0%B8%D0%B3%D1%83%D1%80%D0%B0%D1%86%D0%B8%D1%8F.png)

### Результаты

Результаты применений конфигураций находятся по пути results/augmentation_pipeline и название конфигурации (light, medium, heavy)

---

## Задание 6: Дообучение предобученных моделей

### График обучения

![Medium конфигурация](https://github.com/Ycalk/Lesson5/raw/main/results/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%20ResNet-18.png)

#### Анализ по метрика

- Train Loss. Снижается стабильно и выходит на низкие значение, значит модель успешно обучается на тренировочном наборе.
- Test Loss. Очень нестабильный: резкие скачки и падения. Иногда растет, несмотря на снижение Train Loss, что является признаком переобучения или нестабильности на валидации. Возможно модель слишком большая для объема данных (всего 180 картинок на обучение).
- Train Accuracy. Быстро растет и выходит на высокое значение: ```0.95```–```0.97```.
- Test Accuracy. Ниже Train Accuracy (```0.7```–```0.8```), заметно колеблется от эпохи к эпохе. Это также указывает на переобучение: модель запоминает тренировку, но плохо обобщает.

Модель хорошо учится на тренировочном датасете, но обобщающая способность ограничены. Об этом говорит скачущая кривая Test Loss и заметная разница между Train и Test Accuracy. Чтобы это исправить стоит попробовать обучить модель на большем объеме данных.
