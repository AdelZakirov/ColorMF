# ImageNet: размер и геометрия при evaluation

Проверено 6 октября 2026 года по статьям и официальному коду. Размер входа
модели, размер изображения для метрик и внутренний resize Inception — разные
этапы. Описание training augmentation не подтверждает preprocessing evaluation.

| Работа | Что подтверждено для evaluation | Ограничение |
| --- | --- | --- |
| DDColor | Публичный validation использует `gt_size: 256` и `cv2.resize(img_gt, (256, 256))`, затем RGB -> LAB: растяжение всего изображения без square crop. [Config](https://github.com/piddnad/DDColor/blob/master/options/train/train_ddcolor.yml), [LabDataset](https://github.com/piddnad/DDColor/blob/master/basicsr/data/lab_dataset.py). | Config выбирает val5k; он не документирует все настройки отдельного опубликованного Val50k запуска. |
| MultiColor | В §5.1 сказано, что все изображения resized до 256x256. [Статья](https://arxiv.org/html/2408.04172). | Square/center crop в этом описании не указан; конкретный interpolation и собственный FID backend из текста не установлены. |
| L-CAD | §4.4: первые 5k ImageNet val, test images center cropped и resized до 256x256. [Статья, стр. 9](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf). | Текст не фиксирует interpolation и точную последовательность операций на уровне пикселей. |
| ColTran | Официальный non-training preprocessing берёт максимальный центральный квадрат с размером `min(H, W)` и делает area resize до 256x256. [Код](https://github.com/google-research/google-research/blob/master/coltran/datasets.py). | Training использует случайное положение максимального квадрата; это отдельный режим. |
| Palette | Все входы/выходы задач имеют размер 256x256. Для training colorization указан случайный максимальный square crop -> resize, со ссылкой на ColTran. [Статья, §4 и Appendix B](https://arxiv.org/html/2111.05826). | Этот training-фрагмент сам по себе не доказывает точное положение crop при evaluation. Palette также использует разные evaluation subsets/reference sets; их нельзя смешивать с полным val50k. |
| UniColor | §5.1 явно указывает метрики на изображениях 256x256 (кроме Contextual Loss 96x96). [Статья](https://arxiv.org/html/2209.11223). | Training resize 294x294 + random crop 256x256 описан отдельно. Из него нельзя вывести точный test crop. ImageNet subset — 5 случайных изображений на класс, а не первые 5k. |
| CtrlColor | Inference: пропорциональный resize короткой стороны до 512, затем возврат в исходный размер и замена L. [Статья, §4.1](https://arxiv.org/html/2402.10855). | Это подтверждённый inference pipeline; он не устанавливает отсутствие отдельного resize/crop в evaluator метрик. |
| SeAda | В §3.2 условный вход имеет размер 512x512. [Статья](https://www.ijcai.org/proceedings/2025/0106.pdf). | Точные test crop/resize и размер для FID/PSNR в проверенном тексте не установлены. |
| Imagination | Приведён inference timing для 512x512. [Статья, §4.4](https://arxiv.org/html/2404.05661). | Timing не доказывает, что все опубликованные метрики считались на 512x512 или original-size. |

## Протокол для ColorMF

Фиксированный протокол реализован в `python -m eval.imagenet_val5k`: первые
5,000 по номерам `ILSVRC2012_val_00000001…00005000` → максимальный центральный
квадрат по короткой стороне (нечётные поля округляются вниз) → один resize
uint8 RGB до 256×256 через `cv2.INTER_LINEAR`, без аугментаций и flip.
Подготовленные RGB PNG одновременно служат источником L для модели и GT для
всех метрик. Манифест фиксирует выбор, crop и контрольные суммы; GT FID moments
и оба варианта colorfulness сохраняются заранее. Команды — в `eval/README.md`.

Для сравнения собственных чекпоинтов подходит один зафиксированный набор GT
256x256: максимальный квадрат (желательно центральный), затем resize одним
выбранным методом. Если квадрат выбирался случайно, сохранить выбор/seed либо
использовать уже подготовленные неизменные файлы для всех запусков. Predictions
нужно сравнивать с этими же GT; случайные новые crops при каждом evaluation
изменяют задачу и reference distribution.

Возвращать cropped prediction в исходный прямоугольник ради метрик не нужно:
потерянный при crop контекст не восстанавливается масштабированием. Подготовленные
RGB GT 256x256 можно передать в `python -m eval` без `--resize`. Флаг `--resize`
растягивает оба набора и не реализует crop.

Для сопоставления с DDColor/MultiColor нужен отдельный протокол на исходных RGB:
resize всего изображения до 256x256 без crop, согласованный decoder/interpolation,
преобразование в LAB в том же месте pipeline и одинаковая реализация метрик для
ColorMF и baseline. Уже cropped файлы не позволяют восстановить этот протокол.
Сопоставление с опубликованными числами дополнительно требует совпадения subset,
FID weights/normalization, числа samples и агрегации. В частности, наш
TensorFlow-compatible `pytorch-fid` и публичный [FID DDColor](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/custom_fid.py)
различаются.

Для текущих cropped 256x256 GT нужно один раз создать отдельный FID cache,
например `imagenet_val50k_crop256_gt.npz`. Старый
`results/imagenet_val_original_size_50000/fid_real_statistics.npz` описывает
оригинальные изображения и для этого набора не подходит. Аналогично отдельные
кэши нужны для stretch256, original-size и каждого subset.
