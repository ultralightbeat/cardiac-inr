import math

import torch

from inr4d.models import SIREN, ReLUMLP, TimeEncoding, build_field, count_parameters


def test_siren_shapes_and_size():
    net = SIREN(4, 256, 3, 1)
    out = net(torch.rand(10, 4) * 2 - 1)
    assert out.shape == (10, 1)
    # 4*256+256 + 3*(256*256+256) + 256+1
    assert count_parameters(net) == 198_913


def test_siren_initialisation_bounds():
    net = SIREN(4, 64, 2, 1, first_omega_0=30.0, hidden_omega_0=30.0)
    first = net.body[0].linear.weight
    hidden = net.body[1].linear.weight
    assert first.abs().max() <= 1.0 / 4 + 1e-6
    assert hidden.abs().max() <= math.sqrt(6 / 64) / 30 + 1e-6


def test_relu_control_has_same_size():
    assert count_parameters(ReLUMLP(4, 256, 3, 1)) == count_parameters(SIREN(4, 256, 3, 1))


def test_time_encodings():
    c = torch.tensor([[0.1, 0.2, 0.3, -1.0], [0.1, 0.2, 0.3, 1.0]])
    assert TimeEncoding("none")(c).shape == (2, 3)
    assert torch.allclose(TimeEncoding("scaled", 0.25)(c)[:, 3], torch.tensor([-0.25, 0.25]))
    per = TimeEncoding("periodic", 1.0, num_frames=30)(c)
    assert per.shape == (2, 5)
    # frame 0 sits at angle 0: (sin, cos) = (0, 1)
    assert torch.allclose(per[0, 3:], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_siren_fits_a_sharp_1d_signal_better_than_relu():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 512).reshape(-1, 1)
    y = torch.sign(torch.sin(12 * x))  # square wave: many sharp edges

    def fit(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(400):
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    siren = SIREN(1, 64, 2, 1)
    relu = ReLUMLP(1, 64, 2, 1)
    assert fit(siren) < fit(relu)


def test_build_field_input_sizes():
    c = torch.rand(5, 4)
    for mode in ("none", "raw", "scaled", "periodic"):
        f = build_field("siren", mode, hidden_features=32, hidden_layers=1)
        assert f(c).shape == (5, 1)
