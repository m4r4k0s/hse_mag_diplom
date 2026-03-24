from __future__ import annotations

from typing import Any

import pandas as pd


def metric_columns_description_md() -> str:
    return """## Расшифровка столбцов таблицы

- `decoder` - название алгоритма декодирования.
- `snr_db` - SNR в децибелах.
- `ber` - доля ошибочно восстановленных битов (меньше - лучше).
- `fer` - доля блоков с хотя бы одной ошибкой (меньше - лучше).
- `decode_time_s` - среднее время декодирования в секундах (меньше - быстрее).
- `complexity_proxy` - прокси-оценка сложности (состояния trellis x длина блока).
- `trained_model_used` - признак использования обученной neural-модели.
"""


def analyze_results(df: pd.DataFrame) -> dict[str, str]:
    grouped = df.groupby("decoder", as_index=False).agg(
        ber=("ber", "mean"),
        fer=("fer", "mean"),
        decode_time_s=("decode_time_s", "mean"),
    )
    best_ber = grouped.loc[grouped["ber"].idxmin(), "decoder"]
    best_fer = grouped.loc[grouped["fer"].idxmin(), "decoder"]
    fastest = grouped.loc[grouped["decode_time_s"].idxmin(), "decoder"]

    if len({best_ber, fastest}) > 1:
        tradeoff = (
            f"По качеству (BER) лидирует `{best_ber}`, по скорости - `{fastest}`. "
            "Наблюдается компромисс качество/время."
        )
    else:
        tradeoff = f"`{best_ber}` лидирует и по BER, и по времени в среднем по SNR."

    return {
        "best_ber": str(best_ber),
        "best_fer": str(best_fer),
        "fastest": str(fastest),
        "tradeoff": tradeoff,
    }


def config_overview_md(cfg: dict[str, Any]) -> str:
    exp = cfg.get("experiment", {})
    tr = cfg.get("training", {})
    return "\n".join(
        [
            "## Краткий обзор конфига",
            "",
            f"- `K={exp.get('K')}` - длина информационного блока в битах.",
            f"- `num_blocks={exp.get('num_blocks')}` - число блоков на каждую SNR-точку.",
            f"- `snr_db_list={exp.get('snr_db_list')}` - сетка SNR для оценки.",
            f"- `decoders={exp.get('decoders')}` - запущенные декодеры.",
            f"- `seed={exp.get('seed')}` - seed воспроизводимости генерации сигналов.",
            f"- `training.enabled={tr.get('enabled')}` - включено ли обучение neural-компонентов.",
            f"- `training.epochs={tr.get('epochs')}` - число эпох обучения.",
            f"- `training.learning_rate={tr.get('learning_rate')}` - шаг optimizer."
        ]
    )


def analysis_md(df: pd.DataFrame) -> str:
    a = analyze_results(df)
    return "\n".join(
        [
            "## Итог по качеству и скорости",
            "",
            f"- Лучший алгоритм по BER: `{a['best_ber']}`.",
            f"- Лучший алгоритм по FER: `{a['best_fer']}`.",
            f"- Самый быстрый алгоритм: `{a['fastest']}`.",
            f"- Интерпретация: {a['tradeoff']}",
        ]
    )
