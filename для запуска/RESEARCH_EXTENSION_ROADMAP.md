# Roadmap для следующего расширения

Текущий пакет использует уже реализованные сценарии: AWGN, Rayleigh, burst, noise mismatch, amplitude mismatch, rayleigh mismatch, impulsive mismatch.

Если потребуется сделать исследование ещё более прикладным, дальше стоит расширять код на следующие каналы и искажения:

- Rician fading
- phase offset mismatch
- carrier frequency offset
- inter-symbol interference (ISI)
- clipping / nonlinear distortion
- QPSK + AWGN
- QPSK + Rayleigh

Эти сценарии не добавлены в текущий пакет конфигов, потому что для них сначала нужно расширить генерацию данных и формулу LLR в коде.
