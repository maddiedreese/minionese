import mlx.core as mx

from minionese.model import ModelConfig, count_parameters, create_model


def test_model_shape_and_parameter_scale():
    config = ModelConfig()
    model = create_model(config)
    logits = model(mx.array([[1, 2, 3, 4]]))
    mx.eval(logits)
    assert logits.shape == (1, 4, config.vocab_size)
    assert 1_000_000 <= count_parameters(model) <= 2_000_000

