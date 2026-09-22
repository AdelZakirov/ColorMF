#!/usr/bin/env python3
"""Render measured v2 results without loading networks or changing metrics."""
import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


LOSS_NAMES = (
    'huber_ab', 'gradient_ab', 'lpips_vgg',
    'convnext_perceptual', 'dinov3_perceptual',
)
ERROR_TYPES = (
    'wrong_hue', 'low_saturation', 'color_bleeding', 'subtle_color_error',
    'semantically_wrong_color', 'plausible_alternative_colorization',
)


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(root):
    metadata = json.loads((root / 'summary.json').read_text())
    if metadata.get('protocol_version') != 2:
        raise ValueError('Only revised v2 results are supported')
    rows = read(root / 'per_sample_metrics.csv')
    paired = read(root / 'semantic_paired.csv')
    semantic = read(root / 'semantic_summary.csv')
    sensitivity = read(root / 'sensitivity_by_type.csv')
    monotonicity = read(root / 'within_image_monotonicity.csv')
    cosine = [r for r in read(root / 'gradient_cosine_similarity.csv') if r['scope'] == 'all']
    nonfinite = sum(not math.isfinite(float(v)) for r in rows for k, v in r.items()
                    if k not in ('sample_id', 'error_type'))
    mismatch = max(abs(float(r['severity_difference'])) for r in paired)
    if nonfinite or mismatch > 1e-4:
        raise ValueError(f'Invalid results: nonfinite={nonfinite}, matching error={mismatch}')
    lines = [
        '# Исправленный эксперимент по colorization losses', '',
        '**Рекомендация «ConvNeXt first, DINO second» отозвана.** '
        'Huber + Gradient остаётся разумным baseline, но оптимальность комбинации не установлена.', '',
        f"Полный пересчёт: {metadata['sample_count']} исходника, {len(rows)} оценок, "
        f"{len(metadata['severity_levels'])} severity, устройство {metadata['device']}. "
        f'Все значения и нормы градиентов конечны. Максимальное расхождение severity '
        f'в semantic/plausible паре: {mismatch:.2g} Lab units.', '',
        '## Sensitivity по типам ошибок', '',
        'Доля среднего loss данного типа в сумме средних этого же loss по шести типам. '
        'Столбцы нормированы независимо: они сравнивают профиль одного loss, но не '
        'абсолютную силу разных loss.', '',
        '| Тип ошибки | Huber(ab) | Gradient(ab) | LPIPS/VGG | ConvNeXt | DINOv3 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    sensitivity_by_key = {(r['error_type'], r['loss']): r for r in sensitivity}
    for error_type in ERROR_TYPES:
        values = [float(sensitivity_by_key[(error_type, loss)]['loss_share_across_types'])
                  for loss in LOSS_NAMES]
        lines.append('| ' + error_type + ' | ' + ' | '.join(f'{value:.1%}' for value in values) + ' |')
    lines += ['',
        'Для wrong_hue наибольшая относительная доля у LPIPS (31.7%), затем Huber (26.3%) '
        'и DINOv3 (25.1%). Для color_bleeding наиболее выраженный профиль дают '
        'Gradient(ab) (10.5%) и spatial ConvNeXt (10.3%); Huber почти не выделяет bleeding '
        '(1.8%). Это относительная sensitivity на данном наборе, а не меж-loss рейтинг.', '',
        '## Рост loss с severity', '',
        'Средний within-image Spearman по четырём уровням; в скобках доля строго '
        'монотонных кривых. Эта проверка не смешивает разные изображения.', '',
        '| Тип ошибки | Huber(ab) | Gradient(ab) | LPIPS/VGG | ConvNeXt | DINOv3 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    monotonic_by_key = {}
    for error_type in ERROR_TYPES:
        for loss in LOSS_NAMES:
            values = [float(r['spearman']) for r in monotonicity
                      if r['error_type'] == error_type and r['loss'] == loss]
            monotonic_by_key[(error_type, loss)] = (
                statistics.fmean(values),
                sum(value > 0.999 for value in values) / len(values),
            )
    for error_type in ERROR_TYPES:
        values = [monotonic_by_key[(error_type, loss)] for loss in LOSS_NAMES]
        lines.append('| ' + error_type + ' | ' + ' | '.join(
            f'{mean:.3f} ({perfect:.0%})' for mean, perfect in values) + ' |')
    lines += ['',
        'Huber наиболее стабильно растёт с фактическим ab-отклонением. LPIPS и DINOv3 '
        'тоже в основном монотонны. ConvNeXt менее стабилен; Gradient(ab) хорошо '
        'отслеживает величину bleeding, но норма его градиента почти постоянна из-за L1.', '',
        '## Что показывают парные результаты', '',
        'Доля пар, в которых loss для semantic wrong **выше**, чем для plausible alternative, '
        'при одинаковом среднем расстоянии ab внутри одного исходника. '
        '95% интервалы: bootstrap по исходникам, 2000 повторов, seed=0. '
        'Четыре severity одного исходника не считаются независимыми наблюдениями.', '',
        '| Loss | Semantic > plausible | 95% CI |', '|---|---:|---:|',
    ]
    for row in semantic:
        lines.append(f"| {row['loss']} | {float(row['semantic_win_fraction']):.1%} | "
                     f"{float(row['ci_low']):.1%}–{float(row['ci_high']):.1%} |")
    lines += ['',
        'Таблица описывает парное предпочтение каждого loss на данном наборе кандидатов. '
        'Это не доказательство семантической инверсии: совпадение среднего ab не контролирует '
        'площадь изменений, текстуру, локальные границы и достоверность меток Qwen. '
        'После ослабления corruption её цвет может стать правдоподобным. '
        'Нужна человеческая проверка именно matched-примеров и объектные маски.', '',
        'В старом CSV средняя severity была 35.72 против 13.43 (2.66×), '
        'а pooled ConvNeXt давал semantic больший loss в 89.29% пар. '
        'Сравнение старого и нового процентов не является отдельной абляцией matching: '
        'одновременно изменились признаки, preprocessing и gamut handling.', '',
        '## Масштаб градиента', '',
        'Median gradient RMS по physical ab, усреднённый по типам ошибок. Абсолютный '
        'масштаб зависит от формулы и reduction; перед смешиванием нужны коэффициенты '
        'или gradient balancing.', '',
        '| Loss | Median gradient RMS |', '|---|---:|',
    ]
    for loss in LOSS_NAMES:
        medians = [float(r['median_gradient_rms']) for r in sensitivity if r['loss'] == loss]
        lines.append(f'| {loss} | {statistics.fmean(medians):.3e} |')
    lines += ['', '## Матрица gradient cosine similarity', '',
        'Средний cosine по 3696 изображениям; диагональ равна 1 по определению.', '',
        '| Loss | Huber | Gradient | LPIPS | ConvNeXt | DINOv3 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    cosine_by_pair = {
        frozenset((row['loss_a'], row['loss_b'])): float(row['mean_cosine'])
        for row in cosine
    }
    short_names = ('Huber', 'Gradient', 'LPIPS', 'ConvNeXt', 'DINOv3')
    for row_index, loss_a in enumerate(LOSS_NAMES):
        values = []
        for loss_b in LOSS_NAMES:
            value = 1.0 if loss_a == loss_b else cosine_by_pair[frozenset((loss_a, loss_b))]
            values.append(f'{value:.4f}')
        lines.append('| ' + short_names[row_index] + ' | ' + ' | '.join(values) + ' |')
    lines += ['',
        'Cosine остался низким и в доступном модели пространстве ab. '
        'Значит RGB-параметризация не была единственным объяснением. '
        'Низкий cosine показывает разные направления, но не полезность их комбинации. '
        'Для нормализованного ab нормы умножаются на 127.5; cosine не меняется. '
        'Это градиенты изображений, а не параметров модели. Самые похожие пары — '
        'Huber–LPIPS (0.119) и Huber–Gradient (0.117), но даже это слабое совпадение. '
        'ConvNeXt почти ортогонален всем остальным (0.001–0.004): это новый сигнал по '
        'направлению, но его полезность из одной ортогональности не следует.', '',
        '## Вывод для обучения pMF', '',
        '1. **Базовый набор: Huber(ab) + Gradient(ab).** Huber надёжно задаёт глобальную '
        'точность chroma и единственный чаще предпочитает semantic-wrong plausible-варианту '
        '(61.0%). Gradient — самый прямой и дешёвый сигнал для bleeding; его sensitivity '
        'к bleeding в 5.7 раза выше Huber по доле профиля.', '',
        '2. **Первый perceptual-кандидат для абляции: LPIPS/VGG.** Он сильнее всех выделяет '
        'wrong_hue и почти не дублирует Gradient (cosine 0.0245). Но semantic win rate '
        '40.9% не показывает семантического преимущества над pixel loss.', '',
        '3. **Spatial ConvNeXt — отдельная bleeding-абляция, не default.** По профилю он '
        'реагирует на bleeding почти как Gradient, но даёт почти ортогональный градиент. '
        'Его severity-кривые менее стабильны и semantic win rate всего 23.7%.', '',
        '4. **DINOv3 пока не добавлять в основной рецепт.** Сигнал направленно новый и '
        'монотонный, но здесь он не выигрывает ни semantic discrimination, ни bleeding '
        'у более дешёвых кандидатов. Одновременное добавление всех perceptual losses '
        'усложнит калибровку без подтверждённой пользы.', '',
        'Практический порядок ablation: Huber; Huber+Gradient; Huber+Gradient+LPIPS; '
        'затем заменить LPIPS на spatial ConvNeXt. Сравнивать при выровненных RMS вкладов '
        'градиента и одинаковом compute budget. Финальный выбор делать по held-out '
        'colorization quality: эта диагностика измеряет реакцию на сконструированные ошибки, '
        'а не качество обученной модели.', '',
        '## Исправления и ограничения', '',
        '- ConvNeXt: ImageNet mean/std checkpoint, spatial MSE по четырём стадиям.',
        '- DINO: patch tokens блоков 3/6/9/12 без CLS и registers.',
        '- Huber/Gradient считаются непосредственно по ab; perceptual losses — '
        'через дифференцируемое Lab→RGB при фиксированном L.',
        '- Gamut projection выполняется до интерполяции; RGB clipping не меняет L.',
        '- Semantic/plausible сопоставлены внутри исходника по средней величине ab; '
        'добавлены парные результаты, интервалы и within-image monotonicity.',
        '- Sensitivity shares сохранены как описательные доли; рейтинг разных loss по ним удалён.',
        '- Сохранены версии моделей, библиотек и хеши расчётного кода.', '',
        '**Поправка к фидбеку:** официальный pMF действительно использует global average pooling '
        'и расстояние между итоговыми векторами. Его preprocessing отличается от HF processor. '
        'Новый spatial ConvNeXt — отдельный кандидат для локальных ошибок, не точная копия pMF. '
        '[pMF ConvNeXt](https://github.com/Lyy-iiis/pMF/blob/main/models/convnext.py), '
        '[pMF auxiliary loss](https://github.com/Lyy-iiis/pMF/blob/main/utils/auxloss_util.py), '
        '[HF processor](https://huggingface.co/facebook/convnextv2-base-22k-224/blob/main/preprocessor_config.json).', '',
        'В диагностике используется full-image resize 224, в тренировочном модуле сохраняются '
        'paired random crops. Веса стадий, crops, веса общей смеси и качество обучения '
        'нужно проверять отдельными held-out абляциями. Старые коэффициенты ConvNeXt '
        'не следует считать откалиброванными для нового spatial loss.', '',
        'Протокол генерации и статус проверки меток записаны в `summary.json`. '
        'Численные данные: CSV рядом с этим отчётом. Конфигурация: `summary.json`.', '',
    ]
    (root / 'review.md').write_text('\n'.join(lines))
    # A standalone HTML table view of the same text, without third-party assets.
    blocks = []; in_table = False
    for line in lines:
        if line.startswith('|'):
            if line.startswith('|---'):
                continue
            cells = [html.escape(c.strip()) for c in line.strip('|').split('|')]
            if not in_table:
                blocks.append('<table><tr>' + ''.join('<th>'+c+'</th>' for c in cells) + '</tr>')
                in_table = True
            else:
                blocks.append('<tr>' + ''.join('<td>'+c+'</td>' for c in cells) + '</tr>')
        else:
            if in_table: blocks.append('</table>'); in_table = False
            if line.startswith('# '): blocks.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '): blocks.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line: blocks.append('<p>'+html.escape(line)+'</p>')
    if in_table: blocks.append('</table>')
    (root / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Исправленный loss experiment</title><style>'
        'body{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.5 system-ui}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}'
        'th{background:#eee}</style>' + '\n'.join(blocks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    summarize(parser.parse_args().output_dir)
