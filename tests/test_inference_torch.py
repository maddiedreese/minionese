import torch

from minionese.inference_torch import MinioneseRuntime, TorchMinioneseTransformer, TorchModelConfig


def test_torch_model_shape_and_parameter_scale():
    config = TorchModelConfig()
    model = TorchMinioneseTransformer(config)
    logits = model(torch.tensor([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_354_560


def test_export_loads_and_generates_expected_greedy_reply():
    runtime = MinioneseRuntime("artifacts/final")
    assert runtime.reply("hello") == "Bello! Tulaliloo, amiko!"
