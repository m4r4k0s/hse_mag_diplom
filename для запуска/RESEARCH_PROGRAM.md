# Большая исследовательская программа

Ниже собран пакет из 28 конфигов для расширенного исследования. Конфиги разбиты на 6 пакетов, чтобы идти от контрольных запусков к тяжёлым mismatch-сценариям и финальным плотным графикам.

## Как читать эту программу

- **Pack 01**: контрольный baseline на AWGN. Нужен, чтобы честно зафиксировать, что на хорошо описываемом канале классические алгоритмы обычно сильнее.
- **Pack 02**: умеренно сложные каналы и mismatch. Здесь neural-модели могут сокращать разрыв и показывать устойчивость.
- **Pack 03**: тяжёлые mismatch-сценарии. Это главный блок для поиска практической полезности learned calibration.
- **Pack 04**: плотная SNR-сетка и большой объём статистики. Именно этот блок лучше всего подходит для красивых графиков в диплом.
- **Pack 05**: большие блоки K=512. Здесь проверяется масштабируемость и устойчивость при росте сложности задачи.
- **Pack 06**: neural-focused режимы с усиленным обучением. Эти конфиги нужны, чтобы честно проверить, начинают ли neural-модели выигрывать или хотя бы сильнее сокращать разрыв.

## Рекомендуемый порядок запуска

1. Сначала запустить Pack 01 и Pack 02.
2. Затем перейти к Pack 03 и посмотреть, появляются ли зоны, где neural-модели становятся конкурентнее.
3. После этого запустить Pack 04 для плотных дипломных графиков.
4. В конце - Pack 05 и Pack 06 как самые тяжёлые и длительные.

## Научный тезис

На простых и хорошо смоделированных каналах классические алгоритмы Viterbi и BCJR должны оставаться сильными. На тяжёлых mismatch-сценариях neural-модели могут сокращать разрыв, демонстрировать лучшую устойчивость или выигрывать в отдельных зонах параметров. Именно этот тезис и нужно подтверждать итоговой аналитикой.

## Перечень конфигов

- `research_p01_awgn_k128`: scenario=awgn, K=128, blocks=300, SNR=[-6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p01_awgn_k256`: scenario=awgn, K=256, blocks=300, SNR=[-6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p01_awgn_k512`: scenario=awgn, K=512, blocks=300, SNR=[-6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p01_awgn_k1024`: scenario=awgn, K=1024, blocks=300, SNR=[-6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p02_rayleigh_k128`: scenario=rayleigh, K=128, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p02_rayleigh_k256`: scenario=rayleigh, K=256, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p02_noise_mismatch_k128`: scenario=noise_mismatch, K=128, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p02_noise_mismatch_k256`: scenario=noise_mismatch, K=256, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p02_amplitude_mismatch_k128`: scenario=amplitude_mismatch, K=128, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p02_amplitude_mismatch_k256`: scenario=amplitude_mismatch, K=256, blocks=300, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=15, hidden=64
- `research_p03_burst_k128`: scenario=burst, K=128, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p03_burst_k256`: scenario=burst, K=256, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p03_rayleigh_mismatch_k128`: scenario=rayleigh_mismatch, K=128, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p03_rayleigh_mismatch_k256`: scenario=rayleigh_mismatch, K=256, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p03_impulsive_mismatch_k128`: scenario=impulsive_mismatch, K=128, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p03_impulsive_mismatch_k256`: scenario=impulsive_mismatch, K=256, blocks=500, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=20, hidden=128
- `research_p04_awgn_k256_dense`: scenario=awgn, K=256, blocks=800, SNR=[-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], epochs=25, hidden=128
- `research_p04_rayleigh_k256_dense`: scenario=rayleigh, K=256, blocks=800, SNR=[-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], epochs=25, hidden=128
- `research_p04_rayleigh_mismatch_k256_dense`: scenario=rayleigh_mismatch, K=256, blocks=800, SNR=[-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], epochs=25, hidden=128
- `research_p04_impulsive_mismatch_k256_dense`: scenario=impulsive_mismatch, K=256, blocks=800, SNR=[-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], epochs=25, hidden=128
- `research_p05_rayleigh_k512`: scenario=rayleigh, K=512, blocks=600, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=25, hidden=128
- `research_p05_burst_k512`: scenario=burst, K=512, blocks=600, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=25, hidden=128
- `research_p05_rayleigh_mismatch_k512`: scenario=rayleigh_mismatch, K=512, blocks=600, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=25, hidden=128
- `research_p05_impulsive_mismatch_k512`: scenario=impulsive_mismatch, K=512, blocks=600, SNR=[-8, -6, -4, -2, 0, 2, 4, 6], epochs=25, hidden=128
- `research_p06_noise_mismatch_k256_neuralfocus`: scenario=noise_mismatch, K=256, blocks=800, SNR=[-10, -8, -6, -4, -2, 0, 2, 4, 6], epochs=30, hidden=256
- `research_p06_amplitude_mismatch_k256_neuralfocus`: scenario=amplitude_mismatch, K=256, blocks=800, SNR=[-10, -8, -6, -4, -2, 0, 2, 4, 6], epochs=30, hidden=256
- `research_p06_rayleigh_mismatch_k256_neuralfocus`: scenario=rayleigh_mismatch, K=256, blocks=800, SNR=[-10, -8, -6, -4, -2, 0, 2, 4, 6], epochs=30, hidden=256
- `research_p06_impulsive_mismatch_k256_neuralfocus`: scenario=impulsive_mismatch, K=256, blocks=800, SNR=[-10, -8, -6, -4, -2, 0, 2, 4, 6], epochs=30, hidden=256
