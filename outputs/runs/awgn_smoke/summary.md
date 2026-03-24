# Experiment summary

LLR-calibrated Neural Viterbi / Neural BCJR prototype.

## Artifacts
- signals.npz
- results.csv
- ber_plot.png
- fer_plot.png
- timing_plot.png
- config_used.yaml
- run_metadata.json
- checkpoints/best_neural_viterbi.pt
- checkpoints/best_neural_bcjr.pt

## Краткий обзор конфига

- `K=32` - длина информационного блока в битах.
- `num_blocks=4` - число блоков на каждую SNR-точку.
- `snr_db_list=[0]` - сетка SNR для оценки.
- `decoders=['viterbi', 'bcjr']` - запущенные декодеры.
- `seed=7` - seed воспроизводимости генерации сигналов.
- `training.enabled=False` - включено ли обучение neural-компонентов.
- `training.epochs=1` - число эпох обучения.
- `training.learning_rate=0.001` - шаг optimizer.

## Расшифровка столбцов таблицы

- `decoder` - название алгоритма декодирования.
- `snr_db` - SNR в децибелах.
- `ber` - доля ошибочно восстановленных битов (меньше - лучше).
- `fer` - доля блоков с хотя бы одной ошибкой (меньше - лучше).
- `decode_time_s` - среднее время декодирования в секундах (меньше - быстрее).
- `complexity_proxy` - прокси-оценка сложности (состояния trellis x длина блока).
- `trained_model_used` - признак использования обученной neural-модели.

## Results table

| decoder   |   snr_db |       ber |   fer |   decode_time_s |   complexity_proxy | trained_model_used   |
|:----------|---------:|----------:|------:|----------------:|-------------------:|:---------------------|
| bcjr      |        0 | 0.140625  |  0.75 |       0.0936891 |               2048 | False                |
| viterbi   |        0 | 0.0703125 |  0.75 |       0.0201194 |               2048 | False                |

## Итог по качеству и скорости

- Лучший алгоритм по BER: `viterbi`.
- Лучший алгоритм по FER: `bcjr`.
- Самый быстрый алгоритм: `viterbi`.
- Интерпретация: `viterbi` лидирует и по BER, и по времени в среднем по SNR.
