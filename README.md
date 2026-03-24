# Neural Viterbi vs Neural BCJR (AWGN)

Воспроизводимый Python-пакет для сравнения baseline-декодеров (Viterbi, BCJR) и learned-prototype декодеров (LLR-calibrated Neural Viterbi, LLR-calibrated Neural BCJR) для сверточных кодов в канале AWGN.

## Что происходит в проекте

Проект моделирует стандартный цифровой pipeline связи:

1. `u` - исходные информационные биты.
2. `c` - закодированные биты (сверточный код).
3. `x` - BPSK-символы (`x = 1 - 2*c`).
4. `noise` - AWGN-шум.
5. `y` - принятый сигнал (`x + noise`).
6. `llr` - soft-информация для декодирования.

Далее эти данные подаются в baseline и neural-прототипы декодеров, а затем сравниваются BER/FER/время.

## Папка `для запуска`

В проекте также есть папка `для запуска`, которая используется как демонстрационный и исследовательский слой поверх базового пакета.

В ней находятся:
- основной show-notebook для демонстрации результатов;
- research-конфиги для расширенной серии экспериментов;
- batch-скрипты для массового запуска;
- локальные backup-артефакты экспериментов.

Важно: тяжёлые результаты (`backups/results`, checkpoints, datasets) не хранятся в Git-репозитории и должны быть исключены через `.gitignore`.

## Что означает SNR

`SNR [dB]` - отношение мощности сигнала к мощности шума в децибелах.
- Больше SNR → меньше влияние шума.
- Меньше SNR → декодирование сложнее.

## Что такое BER / FER

- `BER` (Bit Error Rate) - доля ошибочно восстановленных битов.
- `FER` (Frame Error Rate) - доля блоков, где есть хотя бы одна ошибка.

Меньше BER/FER - лучше.

## Что сейчас реализовано как learned prototype

Текущая learned-часть реализована как **LLR calibration prototype**:
- нейросеть калибрует LLR/ветвевые метрики;
- далее используется классический Viterbi/BCJR core.

Это не заявляется как fully end-to-end differentiable trellis decoder core.

## Установка

```bash
pip install -e .
pip install -e ".[dev,notebooks]"
# optional
pip install -e ".[dev,notebooks,tf]"
```

## Быстрый старт (CLI)

```bash
python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_smoke.yaml
```

Или демонстрационный запуск:

```bash
python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
```

## Обучение neural-моделей

```bash
python -m comm_ai.training.train_neural_viterbi --config src/comm_ai/config/experiments/awgn_small.yaml
python -m comm_ai.training.train_neural_bcjr --config src/comm_ai/config/experiments/awgn_small.yaml
```

Важно: `train_neural_viterbi` и `train_neural_bcjr` по умолчанию ожидают существующий
`signals.npz` по пути `outputs/runs/<run_name>/signals.npz`, если не передан `--dataset`.
Обычно сначала запускают `run_experiment` для генерации сигналов.

## Повторное использование сигналов и checkpoint'ов

- Для повторного прогона на тех же сигналах выставьте `experiment.reuse_saved_signals: true`.
- По умолчанию checkpoint'ы ищутся в `outputs/runs/<run_name>/checkpoints/`.
- Пользовательские пути можно передать через `checkpoint_paths` в YAML.

## Артефакты запуска

В `outputs/runs/<run_name>/` сохраняются:
- `signals.npz`
- `results.csv`
- `ber_plot.png`
- `fer_plot.png`
- `timing_plot.png`
- `summary.md`
- `config_used.yaml`
- `run_metadata.json`
- `checkpoints/best_neural_viterbi.pt`
- `checkpoints/best_neural_bcjr.pt`

## Папки `data/raw` и `data/generated`

Эти директории оставлены как стандартные точки расширения для будущих сценариев загрузки/подготовки данных.
В текущей версии основные артефакты экспериментов пишутся в `outputs/runs/`.

## Ноутбуки

Основной пользовательский интерфейс для демонстрации:
- `00_quickstart.ipynb` - быстрый вход и полный мини-pipeline.
- `01_baselines_viterbi_bcjr.ipynb` - baseline-only сравнение.
- `02_train_neural_viterbi.ipynb` - обучение Neural Viterbi.
- `03_train_neural_bcjr.ipynb` - обучение Neural BCJR.
- `04_compare_all.ipynb` - главное сравнение всех декодеров.
- `05_reproduce_from_saved_signals.ipynb` - воспроизводимость на фиксированных сигналах.

## Запуск в Google Colab

```python
!git clone https://github.com/marak0s/dp.git
%cd dp
!pip install -e ".[notebooks]"
!python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_smoke.yaml
```

## Описание структуры кода

Подробная карта модулей: `src/comm_ai/PROJECT_STRUCTURE.md`.
