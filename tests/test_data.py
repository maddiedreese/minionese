from minionese.data import INTENTS, build_examples


def test_data_is_deterministic_and_balanced():
    first = build_examples("train", 480, seed=7)
    second = build_examples("train", 480, seed=7)
    assert first == second
    assert set(row["intent"] for row in first) == set(INTENTS)
    assert {row["input_language"] for row in first} == {"english", "minionese"}


def test_heldout_full_phrasings_do_not_leak():
    train = {row["prompt"] for row in build_examples("train", 2400, seed=7)}
    test = {row["prompt"] for row in build_examples("test", 2400, seed=7)}
    assert train.isdisjoint(test)
