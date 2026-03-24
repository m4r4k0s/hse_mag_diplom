import numpy as np

from comm_ai.channel.awgn import awgn_channel, sigma2_from_snr_db
from comm_ai.channel.llr import bpsk_awgn_llr
from comm_ai.channel.modulation import bpsk_modulate
from comm_ai.codes.convolutional import convolutional_encode
from comm_ai.codes.trellis import build_trellis
from comm_ai.decoders.viterbi import viterbi_decode
from comm_ai.utils.metrics import ber


def test_viterbi_high_snr_low_ber() -> None:
    rng = np.random.default_rng(1)
    tr = build_trellis(7, (171, 133))
    u = rng.integers(0, 2, size=128)
    c = convolutional_encode(u, tr)
    x = bpsk_modulate(c)
    sigma2 = sigma2_from_snr_db(8.0)
    y, _ = awgn_channel(x, sigma2, rng)
    llr = bpsk_awgn_llr(y, sigma2)
    uh = viterbi_decode(llr, tr)
    assert ber(u, uh) < 0.05
