from comm_ai.codes.trellis import build_trellis


def test_trellis_states_and_transitions() -> None:
    tr = build_trellis(7, (171, 133))
    assert tr.num_states == 64
    assert tr.next_state.shape == (64, 2)
    assert tr.out_bits.shape == (64, 2, 2)
