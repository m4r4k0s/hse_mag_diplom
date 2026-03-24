# PROJECT_STRUCTURE

Краткое описание структуры `comm_ai` и связей между компонентами.

## Пакет `comm_ai`

- `channel/` - физический канал и наблюдения:
  - `modulation.py` - BPSK (`c -> x`).
  - `awgn.py` - AWGN и связь `snr_db -> sigma^2`.
  - `llr.py` - вычисление soft-входа LLR.
- `codes/` - кодирование и trellis:
  - `trellis.py` - построение trellis для сверточного кода.
  - `convolutional.py` - свёрточный энкодер (`u -> c`).
- `decoders/` - декодеры:
  - `viterbi.py` - baseline Viterbi.
  - `bcjr.py` - baseline BCJR (log-domain).
  - `neural_viterbi.py` - LLR/branch-metric calibration + Viterbi.
  - `neural_bcjr.py` - LLR calibration + BCJR.
- `datasets/`:
  - `signals_dataset.py` - генерация, сохранение и загрузка `signals.npz`.
- `training/`:
  - `train_neural_viterbi.py` - обучение Neural Viterbi, CLI и Python API.
  - `train_neural_bcjr.py` - обучение Neural BCJR, CLI и Python API.
  - `losses.py` - функции потерь.
- `experiments/`:
  - `run_experiment.py` - главный orchestration-скрипт (CLI), генерация артефактов.
  - `evaluate.py` - расчёт BER/FER/time для выбранных декодеров.
- `utils/`:
  - `seed.py` - фиксирование seed.
  - `io.py` - YAML I/O.
  - `metrics.py` - BER/FER.
  - `plotting.py` - построение и сохранение графиков.
  - `reporting.py` - автосводка и интерпретация результатов.
  - `logging.py`, `timers.py` - вспомогательные утилиты.
- `config/`:
  - `default.yaml` + `experiments/*.yaml` - готовые сценарии запуска.

## Сквозной pipeline

1. `u` (информационные биты) генерируется в `SignalsDataset.generate`.
2. `u -> c` через `convolutional_encode`.
3. `c -> x` через BPSK-модуляцию.
4. `x + noise -> y` через AWGN.
5. `y -> llr` как soft-вход декодеров.
6. Декодеры оцениваются в `evaluate.py`, результаты сохраняются в `outputs/runs/<run_name>/`.

## Публичные точки входа

- Эксперимент: `python -m comm_ai.experiments.run_experiment --config <yaml>`
- Обучение Neural Viterbi: `python -m comm_ai.training.train_neural_viterbi --config <yaml>`
- Обучение Neural BCJR: `python -m comm_ai.training.train_neural_bcjr --config <yaml>`
