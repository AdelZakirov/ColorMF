# ColorMF и опубликованные результаты на ImageNet Val50k

Проверено 5 октября 2026 года. Все строки относятся к **полной validation
выборке ImageNet: 50 000 изображений**, с одной колоризацией на изображение
в нашем запуске. Числа остальных моделей взяты из опубликованных таблиц;
эти модели здесь не запускались. Совпадение датасета не означает совпадения
препроцессинга и реализации метрик, поэтому таблица служит ориентиром.

## Сравнение

PSNR — dB, выше лучше. FID и ΔCF — ниже лучше. CF показывает красочность;
его увеличение само по себе не доказывает улучшение качества.

| Модель / публикация | FID ↓ | PSNR ↑ | CF | ΔCF ↓ | Источник чисел именно для Val50k |
| --- | ---: | ---: | ---: | ---: | --- |
| **ColorMF, EMA 500, seed 1** | **4.0860** | **22.3275** | **26.1857** | **12.6107** | [Наш отчёт](results/imagenet_val_original_size_50000/report.json), исходное разрешение |
| MultiColor, ACM MM 2024 | 0.42 | 24.58 | 38.89 | 0.20 | [Table 2, столбцы ImageNet (val50k)](https://arxiv.org/html/2408.04172) |
| DDColor-large, ICCV 2023 | 0.96 | 23.74 | 38.65 | 0.44 | [Table 1, столбцы ImageNet (val50k)](https://arxiv.org/html/2212.11613) |
| DDColor-tiny, ICCV 2023 | 1.23 | 23.63 | 37.72 | 1.37 | [DDColor Table 1](https://arxiv.org/html/2212.11613) |
| BigColor, ECCV 2022 | 1.24 | 21.24 | 40.01 | 0.92 | [Переоценка в DDColor Table 1](https://arxiv.org/html/2212.11613) |
| ColorFormer, ECCV 2022 | 1.71 | 23.00 | 39.76 | 0.67 | [Переоценка в DDColor Table 1](https://arxiv.org/html/2212.11613) |
| GCPColor / Wu et al., ICCV 2021 | 3.62 | 21.81 | 35.13 | 3.96 | [Переоценка в DDColor Table 1](https://arxiv.org/html/2212.11613) |
| CT², ECCV 2022 | 4.95 | 22.93 | 39.96 | 0.87 | [Переоценка в DDColor Table 1](https://arxiv.org/html/2212.11613) |
| ColTran, ICLR 2021 | 6.14 | 22.30 | 35.50 | 3.59 | [Переоценка в DDColor Table 1](https://arxiv.org/html/2212.11613) |

DDColor сообщает, что переоценивал предыдущие методы с их официальным кодом
и весами (§4.2). MultiColor использует опубликованные baseline-числа
DDColor/ColorFormer (§5.2); повторяющиеся строки не являются независимым
подтверждением результата. [DDColor](https://arxiv.org/html/2212.11613),
[MultiColor](https://arxiv.org/html/2408.04172).

## Какие протоколы различаются

- **Наш запуск:** сначала L из исходного RGB, затем L → 256×256 → предсказание
  ab → ab обратно в исходное разрешение → RGB с исходным L. Метрики считаются
  на исходном размере без внешнего resize/crop. Все 50 000 изображений
  включены; FID использует 50 000 GT и 50 000 predictions.
- **DDColor:** публичный [validation config](https://github.com/piddnad/DDColor/blob/master/options/train/train_ddcolor.yml)
  задаёт `gt_size: 256`; [LabDataset](https://github.com/piddnad/DDColor/blob/master/basicsr/data/lab_dataset.py)
  сначала делает `cv2.resize` RGB до этого размера, затем извлекает LAB.
  Это растяжение изображения, а не center crop. Этот config содержит список
  val5k: он подтверждает публичный validation-пайплайн, но сам по себе не
  устанавливает все настройки отдельного опубликованного Val50k запуска.
- **MultiColor:** §5.1 указывает resize всех изображений до 256×256.
  [Описание экспериментов](https://arxiv.org/html/2408.04172).
- **FID:** у нас TF-compatible `pytorch-fid` InceptionV3, pool3/2048.
  Публичный [FID DDColor](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/custom_fid.py)
  использует torchvision Inception и другое преобразование входа;
  [ColorModel](https://github.com/piddnad/DDColor/blob/master/basicsr/models/color_model.py)
  загружает `inception_v3_google-1a9a5a14.pth`. Значения FID нельзя считать
  напрямую взаимозаменяемыми. Точный backend собственного MultiColor по
  проверенному тексту не установлен.

**ΔCF в таблице** для ColorMF — `abs(mean(CF_pred) - mean(CF_GT))`, поле
`delta_mean_colorfulness`, как в конвенции DDColor/MultiColor.
Наш `mean(abs(CF_pred_i - CF_GT_i))` равен **18.0566** и является другой
агрегацией. Средний CF наших GT — **38.7964**.
У опубликованных Val50k строк DDColor/MultiColor CF и ΔCF соответствуют
среднему GT CF около 39.09 (с учётом округления); он отличается от нашего GT.
Это дополнительно ограничивает прямую сопоставимость CF.

SSIM, LPIPS–GT и ΔE00 отсутствуют в выбранных опубликованных Val50k таблицах.
Наши значения: **SSIM 0.903537**, **LPIPS AlexNet v0.1 0.181936**,
**ΔE00 11.646028**. Числа этих метрик из Val5k или других сетов сюда не перенесены.

## Зафиксированный запуск ColorMF

Epoch **239**, global step **1050000**, EMA **500**, seed **1**, noise **0.25**,
один шаг семплирования, float32 CUDA. Все 50k сгенерированы заново одним
зафиксированным checkpoint; предыдущие 5k от epoch 234 не смешивались с ними.

- [Протокол, происхождение весов и результаты](results/imagenet_val_original_size_50000/README.md).
- [Машиночитаемый отчёт](results/imagenet_val_original_size_50000/report.json).
- [Проверка покрытия и файлов](results/imagenet_val_original_size_50000/verification.json).
- PNG: `/mnt/IMAGING/HUB/DATASETS/general_datasets/imagenet/imagenet1k/source/val_set_full_cmf/original_size_50000`.

Слабое место текущего запуска — недостаточная красочность: CF **26.19**
против **38.80** у его собственных GT. Это согласуется с наблюдением о блеклых
результатах. PSNR численно ниже DDColor-large/MultiColor, но размер изображений
при оценке различается. Для строгого сравнения нужно оценить baseline-выходы
и ColorMF при одинаковом препроцессинге и одним evaluator.

## Почему другие строки не включены

В проверенной ранее [таблице работ](paper_comparison.md) Palette генерирует
ctest10k, хотя использует Val50k как FID reference; это не 50k колоризаций
validation-сета. SeAda, L-CAD и UniColor имеют Val5k результаты, а Imagination
и CtrlColor — ctest10k. Сравнение с публикациями на Val5k сохранено
[отдельно](paper_comparison_val5k.md). Эта таблица содержит только подтверждённые
Val50k строки; она не претендует на исчерпывающий обзор всех публикаций.
