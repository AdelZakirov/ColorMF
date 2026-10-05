# Работы для сравнения ColorMF

Проверено 2 октября 2026 года. Целевой эксперимент ColorMF: автоматическая
колоризация ImageNet validation, RGB 256×256. Числа ниже — опубликованные
результаты, а не наши запуски моделей. Источник и выборка указаны для каждой
строки; таблица служит подборкой ориентиров, а не общим рейтингом FID.

Для основной экспериментальной таблицы рекомендую **DDColor, MultiColor,
SeAda, UniColor, L-CAD в scarce/automatic режиме, CtrlColor без подсказок и
Palette**. DDColor и MultiColor особенно полезны для полного val50k; SeAda
и CtrlColor добавляют публикации 2025 года; UniColor и Palette дают
стохастические базовые модели. **VVFM** полезен для сравнения методики оценки:
его таблица включает все семь наших метрик и best-of-5.

## Единая таблица опубликованных метрик

PSNR в dB. `—` означает отсутствие числа в выбранной таблице/режиме;
`н/д` — метрика заявлена, но числовую таблицу не удалось проверить.
CF измеряет насыщенность, поэтому его рост сам по себе не означает улучшения.
LPIPS в основной колонке — расстояние до GT, а не разнообразие результатов.
«Статья» ведёт на оригинальную работу каждой модели; «Источник чисел» —
на таблицу, откуда взяты значения, включая переоценку модели в другой работе.

| Работа, публикация | Эксперимент источника | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS–GT ↓ | CF | ΔCF ↓ | ΔE00 ↓ | Многовариантная оценка | Источник чисел | Статья |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| **SeAda, IJCAI 2025** | ImageNet val5k | 3.36 | 24.53 | — | — | 36.98 | 1.23 | — | Визуальные варианты; числовой diversity/best-K не приведён | [Table 1](https://www.ijcai.org/proceedings/2025/0106.pdf) | [Статья](https://www.ijcai.org/proceedings/2025/106) |
| **CtrlColor, IJCV 2025; preprint 2024** | ImageNet ctest10k, unconditional; числа из preprint | 4.2915 | — | — | — | 44.9256 | — | — | Без пользовательских подсказок | [Table 1](https://arxiv.org/html/2402.10855); [публикация 2025](https://link.springer.com/article/10.1007/s11263-025-02549-6) | [Статья](https://link.springer.com/article/10.1007/s11263-025-02549-6) |
| **SemanticColorizer, Displays 2025** | ImageNet val; открытый текст обсуждает val50k и val5k | н/д | — | — | — | — | н/д | — | Не установлено по доступному тексту | [Страница издателя](https://www.sciencedirect.com/science/article/pii/S0141938225002094) | [Статья](https://www.sciencedirect.com/science/article/pii/S0141938225002094) |
| **MultiColor, ACM MM 2024** | ImageNet val50k | 0.42 | 24.58 | — | — | 38.89 | 0.20 | — | Числовой diversity/best-K не приведён | [Table 2](https://arxiv.org/html/2408.04172) | [Статья](https://arxiv.org/abs/2408.04172) |
| MultiColor, ACM MM 2024 | ImageNet val5k | 2.17 | 24.69 | — | — | 38.24 | 0.03 | — | Тот же метод | [Table 2](https://arxiv.org/html/2408.04172) | [Статья](https://arxiv.org/abs/2408.04172) |
| **Imagination, CVPR 2024** | ImageNet ctest10k | 3.62 | 23.8 | 0.884 | 0.207 | 48.8 | — | — | Несколько синтезированных референсов; это не oracle best-K по GT | [Table 3](https://arxiv.org/pdf/2404.05661) | [Статья](https://arxiv.org/abs/2404.05661) |
| **VVFM, SIGGRAPH 2024** | COCO val, один случайный вариант | 7.92 | 20.90 | 0.930 | 0.201 | 34.43 | 13.01 | 10.25 | Single sample | [Table 1](https://assets.studios.disneyresearch.com/app/uploads/2024/07/Versatile-Vision-Foundation-Model-for-Image-and-Video-Colorization-Paper.pdf) | [Статья](https://studios.disneyresearch.com/2024/07/28/versatile-vision-foundation-model-for-image-and-video-colorization/) |
| VVFM, SIGGRAPH 2024 | COCO val, best-of-5 | 7.29 | 23.11 | 0.942 | 0.162 | 32.19 | 9.44 | 8.53 | Один вариант выбран голосованием семи метрик | [Table 1, §4.1](https://assets.studios.disneyresearch.com/app/uploads/2024/07/Versatile-Vision-Foundation-Model-for-Image-and-Video-Colorization-Paper.pdf) | [Статья](https://studios.disneyresearch.com/2024/07/28/versatile-vision-foundation-model-for-image-and-video-colorization/) |
| **L-CAD, NeurIPS 2023** | ImageNet, scarce description | 4.36 | 24.47 | 0.92 | 0.16 | 34.04 | 3.68 | — | Общий промпт, например «a colorful image» | [Table 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) | [Статья](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) |
| **DDColor-large, ICCV 2023** | ImageNet val50k | 0.96 | 23.74 | — | — | 38.65 | 0.44 | — | Один результат | [Table 1](https://arxiv.org/html/2212.11613) | [Статья](https://arxiv.org/abs/2212.11613) |
| **UniColor, TOG 2022** | ImageNet: 5k случайных val, 5 на класс; unconditional | 9.46 | — | — | 0.1945 | 39.01 | — | — | Стохастическая авторегрессия | [Tables 1–2, §5](https://arxiv.org/html/2209.11223) | [Статья](https://arxiv.org/abs/2209.11223) |
| **Palette, SIGGRAPH 2022** | ImageNet ctest10k; FID относительно val50k | 3.4 | — | — | — | — | — | — | LPIPS-diversity **0.15** для L2, **0.09** для L1; пары соседних вариантов | [Tables 6, C.1](https://arxiv.org/html/2111.05826) | [Статья](https://arxiv.org/abs/2111.05826) |
| ColorFormer, ECCV 2022 | ImageNet, числа из сравнительной таблицы L-CAD | 4.64 | 23.14 | 0.89 | 0.18 | 37.95 | 0.23 | — | — | [L-CAD Table 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) | [Статья](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760020.pdf) |
| CT², ECCV 2022 | ImageNet, числа из сравнительной таблицы L-CAD | 5.51 | 23.50 | 0.92 | 0.19 | 38.48 | 2.17 | — | — | [L-CAD Table 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) | [Статья](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670001.pdf) |
| BigColor, ECCV 2022 | ImageNet val50k, числа из DDColor | 1.24 | 21.24 | — | — | 40.01 | 0.92 | — | Возможны разные варианты | [DDColor Table 1](https://arxiv.org/html/2212.11613) | [Статья](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670343.pdf) |
| GCPColor, ICCV 2021 | ImageNet val50k, числа из DDColor | 3.62 | 21.81 | — | — | 35.13 | 3.96 | — | Генеративный prior | [DDColor Table 1](https://arxiv.org/html/2212.11613) | [Статья](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Towards_Vivid_and_Diverse_Image_Colorization_With_Generative_Color_Prior_ICCV_2021_paper.html) |

Для L-CAD/CT²/ColorFormer здесь сохранена целиком одна исходная строка из
L-CAD Table 3: её CF/ΔCF не заменены числами из DDColor или SeAda. В частности,
CT² имеет ΔCF 2.17 в L-CAD и 0.27 в DDColor. Два значения относятся к
разным опубликованным измерениям; это не ошибка переноса.

## Какие работы брать прежде всего

1. **DDColor + MultiColor** — основные ориентиры для val50k. MultiColor
   явно использует 256×256 и развивает декодирование цветов в нескольких
   пространствах. У DDColor есть открытые веса/код; для MultiColor официальный
   код/веса в проверенных источниках не найдены.
2. **SeAda + CtrlColor** — приоритетные недавние diffusion-модели.
   SeAda получает условие из grayscale через captioner/semantic adapter,
   не требуя описания цвета от пользователя. У
   [CtrlColor есть открытые код и веса](https://github.com/ZhexinLiang/Control-Color).
   У SeAda официальный код/веса в проверенных источниках не найдены.
3. **UniColor + L-CAD + Palette** — полезны для стохастического ColorMF.
   UniColor брать в unconditional-режиме, L-CAD — с общим scarce-промптом,
   Palette — как классическую conditional diffusion модель. У
   [UniColor есть открытая реализация](https://github.com/luckyhzt/unicolor).
4. **Imagination** — сильная работа CVPR 2024 с автоматической генерацией
   референсов. Полезна для качества/разнообразия, но использует более сложный
   многокомпонентный inference. В статье обсуждается 512×512 inference;
   утверждать, что все её опубликованные метрики посчитаны на 256×256, нельзя.
5. **VVFM** — наиболее близкий найденный источник для *состава метрик*,
   включая CIEDE2000 и best-of-5. В таблице выше оставлены явно помеченные
   COCO-числа: не выдаём их за ImageNet. Для ImageNet в основном тексте
   есть обсуждение результатов, но числовая Table 1 относится к COCO.
6. **SemanticColorizer** — подходящий свежий кандидат по задаче,
   но числовая таблица и официальный код по доступным источникам не проверены.
   Пока не использовать как подтверждённую числовую baseline-строку.

## Соответствие текущему evaluator

- **FID:** наш backend — `pytorch-fid`, TF-compatible InceptionV3 pool3/2048.
  [Код UniColor](https://github.com/luckyhzt/unicolor/blob/dfd77f2f1498c1944133dfcaf0a873e1a3871189/sample/unsorted_codes/metrics.ipynb)
  использует этот backend. У
  [DDColor](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/custom_fid.py)
  другой Inception/normalization. MultiColor наследует часть чисел DDColor,
  поэтому его таблица сама по себе не подтверждает совместимость FID с нами.
  Для SeAda, Imagination и VVFM точный backend по проверенным источникам
  не установлен. Название FID в статье не доказывает совпадение реализации.
- **PSNR/SSIM:** RGB PSNR и Gaussian SSIM 11×11/sigma=1.5 проверены на
  20 синтетических парах 256×256 относительно официальных функций DDColor;
  максимальное абсолютное расхождение соответственно `1.62e-7` и `1.12e-7`.
  SSIM также сверён с `pytorch-msssim`: `4.18e-7`.
- **LPIPS:** используется официальный v0.1, по умолчанию AlexNet,
  RGB `[-1,1]`. Это совпадает с выбранным backend в коде UniColor.
  Реальные pretrained LPIPS/Inception в технических тестах не запускались;
  проверены адаптеры, порядок пар и агрегация с substitute networks.
- **CF:** вариант `absolute` совпадает с официальной функцией DDColor
  на проверенных парах с ошибкой не более `8.72e-7`.
- **ΔCF:** `delta_colorfulness = mean_i(abs(CF(pred_i)-CF(gt_i)))`;
  `delta_mean_colorfulness = abs(mean_i(CF(pred_i))-mean_i(CF(gt_i)))`.
  DDColor/MultiColor публикуют очень малую разницу средних; для этой
  конвенции использовать второе поле. VVFM описывает per-sample разницу;
  для неё использовать первое. В L-CAD точная конвенция агрегации по
  доступному описанию не установлена.
- **ΔE00:** физический CIELAB D65/2°, CIEDE2000, среднее по пикселям.
  Проверены все 34 [эталонные пары Sharma](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/):
  максимальная ошибка `4.95e-5`, в пределах округления эталона.
- **Diversity:** наш LPIPS усредняет *все* неупорядоченные пары вариантов;
  Palette усредняет *соседние* пары. Это одна идея оценки, но разные
  политики выбора пар; опубликованные 0.15/0.09 не являются точными
  целевыми значениями для нашего all-pairs режима.
- **Best-of-K:** наш evaluator независимо минимизирует LPIPS и ΔE00
  по цельным изображениям. VVFM выбирает один общий вариант голосованием
  PSNR/MSE/SSIM/MS-SSIM/UQI/ERGAS/VIF и затем считает все метрики на нём.
  Наши oracle minima нельзя подписывать «VVFM best-of-5 protocol».

Проверка FID-формулы/streaming statistics относительно `pytorch-fid`
на синтетических признаках дала максимальное расхождение `5.69e-13`.
Все 37 тестов `eval/tests` прошли. Новых ошибок в вычислении метрик не найдено;
различия выше — различия опубликованных определений и режимов оценки.

## Дополнительно просмотренные недавние работы

- [GoLoColor, ICASSP 2025](https://tianaiyue.github.io/): интересен как
  global/local semantic diffusion baseline; полный текст с проверенной
  числовой таблицей в доступных источниках не найден.
- [Color-Turbo, preprint 2025](https://arxiv.org/html/2503.14974): полезен
  для сравнения быстрого diffusion inference. Однако ImageNet-эксперимент
  подаёт BLIP-описания **цветного GT**, поэтому опубликованные результаты
  не включены в automatic grayscale-only таблицу. Репозиторий
  [авторов](https://github.com/lyf1212/Color-Turbo) пока содержит TODO
  публикации кода модели и evaluation protocol.
- [VGG19+CLAHE, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-40292-1):
  подтверждённая свежая публикация с ImageNet, но основная проверенная
  числовая оценка — PSNR/SSIM; PSNR усредняется по каналам, что отличается
  от нашего PSNR по общей RGB MSE. В тексте расходятся описания количества
  тестовых изображений и размеров. Приоритет ниже основных baseline-моделей.
- [Residual Attention U-Net, Electronics 2026](https://www.mdpi.com/2079-9292/15/7/1462):
  заявлены summer2winter/NCData/COCO-Stuff, поэтому для основной ImageNet
  таблицы не выбран.
- [Color and Frequency Correction, preprint 2025](https://arxiv.org/pdf/2510.23399):
  не установлена ImageNet validation выборка, а часть экспериментов
  получает средние цвета GT как дополнительный вход; низкий приоритет
  для automatic grayscale-only сравнения.
- [LGA-Net, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_LGA-Net_Learning_Local_and_Global_Affinities_for_Sparse_Scribble_based_ICCV_2025_paper.pdf)
  требует scribbles; MangaNinja/ColorizeDiffusion и подобные новые работы
  решают колоризацию line art или reference-based задачу. Их не включаем
  в основной набор автоматической колоризации фотографий.

Свежесть сама по себе не делает работу лучшим baseline. Из проверенных
публикаций 2025 года приоритет для основного сравнения — SeAda и CtrlColor;
для полного ImageNet val50k особенно важен MultiColor 2024.
