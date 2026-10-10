# Train Transformers with INT4: результаты и сравнение

## Содержание набора

В каталоге `outputs/` сохранены все артефакты двух запусков на задаче GLUE CoLA:

- `outputs/bert-base-uncased/quantize/.../fp16/` — обучение с 4-bit-квантизацией;
- `outputs/bert-base-uncased/classic/.../fp16/` — контрольный запуск без квантизации.

В каждом каталоге результатов находятся:

- `all_results.json` — итоговая метрика, время и FLOPs;
- `loss.json` — train/validation loss;
- `loss.png` — график loss;
- `lr.png` — график learning rate;
- `acc.txt` — текстовая история оценки.

Дополнительно сохранены:

- `logs/baseline_cola_seed27_20261010_120903.log` — полный лог baseline-запуска;
- `SingleDivide/task.sh` — исходный скрипт 4-bit запуска;
- `SingleDivide/test_mrpc.py` — скрипт-обёртка формирования команды обучения;
- `SingleDivide/test_glue.py` — основной скрипт обучения и оценки;
- `compose.yaml`, `Dockerfile`, `docker/requirements.txt` — файлы контейнерного окружения;
- `readme.md` и `SingleDivide/readme.md` — описание исходного проекта.
- `scripts/measure_model_size.py` — расчёт размера параметров FP16 и упакованного INT4.

## Конфигурация экспериментов

Общие параметры обоих запусков:

- модель: `bert-base-uncased`;
- задача: `cola`;
- seed: `27`;
- 3 эпохи, 804 шага оптимизации;
- batch size: `32`;
- learning rate: `2e-5`;
- максимальная длина последовательности: `128`;
- `fp16=True`;
- `dynamic=True`, `bmm=False`;
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB.

### 4-bit запуск

Использовались параметры:

```text
choice=quantize
training-bit=all4bit
hadamard=True
learnable_hadamard=False
input_quant_method=lsq
weight_quant_method=lsq
2gw=True
2gi=True
```

То есть квантизация применялась к активациям, весам и градиентам с 4 битами.

### Обычный запуск

Для baseline использовались:

```text
choice=classic
training-bit=exact
hadamard=False
2gw=False
2gi=False
```

Все линейные слои оставались обычными `torch.nn.Linear`; при этом запуск также выполнялся в режиме `fp16=True`, поэтому это сравнение INT4 с FP16-baseline, а не с FP32.

## Результаты

| Метрика | INT4 | Без квантизации (FP16) |
|---|---:|---:|
| Итоговый Matthews correlation (MCC) | 0.510582 | **0.595586** |
| Training time, с | 65.8036 | **13.0785** |
| FLOPs | 110,098,327 | 109,483,778 |
| FLOPs learn | 109,483,927 | 109,483,778 |
| Validation loss по эпохам | 0.46338; 0.51297; 0.52690 | 0.41355; 0.50706; 0.50851 |

MCC по эпохам для baseline:

```text
Epoch 0: 0.555946
Epoch 1: 0.546945
Epoch 2: 0.595586
```

Для INT4 в сохранённом `all_results.json` записан итоговый MCC `0.510582`.

## Интерпретация сравнения

- В данном запуске FP16-baseline показал MCC выше на `0.085004`.
- Baseline завершился примерно в `5.03` раза быстрее: `13.08` с против `65.80` с.
- Число FLOPs почти одинаково. Это ожидаемо: FLOPs отражают вычислительный объём модели, но не обязательно фактическую стоимость программных операций квантизации.
- Полученное замедление INT4 не доказывает, что INT4-арифметика медленнее аппаратного FP16. В текущей реализации присутствуют накладные расходы программных quantizer-операций и CUDA-обвязки; отдельного benchmark чистых матричных умножений не проводилось.
- Сравнение выполнено по одному seed и одной задаче, поэтому его нельзя считать статистически устойчивым выводом для всех GLUE-задач.

## Размер весов модели

В каталогах результатов нет сохранённых checkpoint-файлов (`pytorch_model.bin`,
`model.safetensors` и т.п.), поэтому размер фактически сериализованной обученной
модели по этим запускам измерить нельзя.  В текущем коде INT4 также используется
как training/fake quantization с дополнительными операциями, а не как готовый
упакованный checkpoint.

Для оценки размера именно весов был добавлен скрипт:

```bash
python scripts/measure_model_size.py
```

Для `bert-base-uncased` с `109,483,778` параметрами он даёт:

| Представление весов | Расчётный размер |
|---|---:|
| FP16, 2 байта на параметр | 208.82 MiB |
| Упакованный INT4, 0.5 байта на параметр | 52.21 MiB |

Таким образом, теоретический INT4 занимает **25%** размера FP16 и экономит
**75%** памяти под сами веса. Это оценка без metadata, scale/zero-point и
служебных структур. Например, при групповой квантизации scale-параметры нужно
добавлять отдельно через `--scale-bytes` и `--group-size`.

Важно: это не размер всего процесса. В runtime дополнительно расходуются
активации, градиенты, optimizer states, CUDA buffers и, в данной реализации,
возможные FP32-копии параметров. Для получения фактического размера файла
нужно отдельно сохранить checkpoint после обучения и измерить его через `du`.

## Процедура запуска

### 1. Подготовка окружения

Работа выполнялась на удалённой машине в `~/test_LSS`. Проект запускался внутри Docker-контейнера `bert-int4`, собранного по `Dockerfile`, с GPU-доступом и примонтированными каталогами:

```text
~/test_LSS/outputs  -> /workspace/SingleDivide/resultTime
~/test_LSS/hf-cache -> /workspace/.cache/huggingface
```

В контейнере использовались CUDA/PyTorch-зависимости из `docker/requirements.txt`. Перед запуском создавалась ссылка на локальную версию Transformers:

```bash
ln -sfn transformersLocal transformers
```

### 2. Запуск INT4

Эквивалентная команда задаётся в `SingleDivide/task.sh`:

```bash
cd /workspace/SingleDivide
bash task.sh
```

Она вызывает `test_mrpc.py` для `cola` с `--training-bit all4bit`, `--choice quantize`, LSQ для входов и весов, Hadamard-преобразованием и двумя 4-bit слоями для градиентов.

### 3. Запуск baseline

Контрольный прогон запускался той же обёрткой, но без квантизации:

```bash
cd /workspace/SingleDivide
python test_mrpc.py \
  --model_name_or_path bert-base-uncased \
  --choice classic \
  --training-bit exact \
  --hadamard False \
  --learnable_hadamard False \
  --ifq lsq --wfq lsq \
  --2gw False --2gi False \
  --task cola \
  --bmm False --dynamic True \
  --fp16 True
```

Внутри `test_mrpc.py` эта команда вызывает:

```bash
accelerate launch test_glue.py ...
```

с одинаковыми для сравнения параметрами: seed `27`, 3 эпохи, batch size `32`, learning rate `2e-5` и max length `128`.

### 4. Где искать результаты

Пути формируются автоматически:

```text
resultTime/bert-base-uncased/quantize/cola/seed=27/dynamic=True/bmm=False/fp16/
resultTime/bert-base-uncased/classic/cola/seed=27/dynamic=True/bmm=False/fp16/
```

Для удобства в этом архиве они представлены в `outputs/`.
