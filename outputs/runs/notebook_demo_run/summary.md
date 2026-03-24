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

- `K=64` - длина информационного блока в битах.
- `num_blocks=12` - число блоков на каждую SNR-точку.
- `snr_db_list=[0, 2]` - сетка SNR для оценки.
- `decoders=['viterbi', 'bcjr', 'neural_viterbi', 'neural_bcjr']` - запущенные декодеры.
- `seed=123` - seed воспроизводимости генерации сигналов.
- `training.enabled=True` - включено ли обучение neural-компонентов.
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

| decoder        |   snr_db |       ber |      fer |   decode_time_s |   complexity_proxy | trained_model_used   |
|:---------------|---------:|----------:|---------:|----------------:|-------------------:|:---------------------|
| bcjr           |        0 | 0.248698  | 0.833333 |       0.194474  |               4096 | False                |
| bcjr           |        2 | 0.0221354 | 0.25     |       0.192346  |               4096 | False                |
| neural_bcjr    |        0 | 0.244792  | 0.833333 |       0.199446  |               4096 | True                 |
| neural_bcjr    |        2 | 0.0898438 | 0.416667 |       0.197707  |               4096 | True                 |
| neural_viterbi |        0 | 0.269531  | 0.833333 |       0.0497654 |               4096 | True                 |
| neural_viterbi |        2 | 0.0221354 | 0.25     |       0.0501807 |               4096 | True                 |
| viterbi        |        0 | 0.25651   | 0.833333 |       0.0472974 |               4096 | False                |
| viterbi        |        2 | 0.0221354 | 0.25     |       0.0469226 |               4096 | False                |

## Итог по качеству и скорости

- Лучший алгоритм по BER: `bcjr`.
- Лучший алгоритм по FER: `bcjr`.
- Самый быстрый алгоритм: `viterbi`.
- Интерпретация: По качеству (BER) лидирует `bcjr`, по скорости - `viterbi`. Наблюдается компромисс качество/время.
