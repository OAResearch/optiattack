import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from client.core import torch_support  # noqa: E402

LABELS = ["a", "b", "c", "d", "e", "f", "g"]


def _model_and_preprocess():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.Tanh(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, len(LABELS)))
    model.eval()

    def preprocess(x):  # HxWxC float 0-255 -> 1xCxHxW
        return (x.permute(2, 0, 1).unsqueeze(0) - 127.5) / 127.5

    return model, preprocess


@pytest.mark.parametrize("kind", ["loss", "class"])
def test_torch_gradient_fn_matches_finite_difference(kind):
    model, preprocess = _model_and_preprocess()
    grad_fn = torch_support.torch_gradient_fn(model, preprocess, LABELS)
    rng = np.random.RandomState(0)
    image = (rng.rand(8, 8, 3) * 255).astype(np.float32)
    grad = grad_fn(image, "c", kind)
    assert grad.shape == image.shape and grad.dtype == np.float32

    def scalar(img):
        with torch.no_grad():
            logits = model(preprocess(torch.from_numpy(img.astype(np.float32))))[0].double()
        if kind == "class":
            return logits[2].item()
        return torch.nn.functional.cross_entropy(logits[None], torch.tensor([2])).item()

    direction = rng.randn(*image.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    eps = 2.0
    numeric = (scalar(image + eps * direction) - scalar(image - eps * direction)) / (2 * eps)
    analytic = float(np.sum(grad * direction))
    assert abs(numeric - analytic) / max(abs(analytic), 1e-8) < 0.05


def test_torch_gradient_fn_rejects_unknown_label_and_kind():
    model, preprocess = _model_and_preprocess()
    grad_fn = torch_support.torch_gradient_fn(model, preprocess, LABELS)
    image = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        grad_fn(image, "zzz", "loss")
    with pytest.raises(ValueError):
        grad_fn(image, "a", "bogus")


def test_torch_predict_fn_returns_topk_and_all_logits():
    model, preprocess = _model_and_preprocess()
    predict = torch_support.torch_predict_fn(model, preprocess, LABELS, top_k=5)
    data = (np.random.RandomState(2).rand(8 * 8 * 3) * 255).astype(np.uint8)
    out = predict(data, {"target": None})
    assert len(out["logits"]) == len(LABELS)
    assert len(out["predictions"]) == 5
    assert all(set(p) == {"label", "score"} and isinstance(p["score"], float) for p in out["predictions"])
    scores = [p["score"] for p in out["predictions"]]
    assert scores == sorted(scores, reverse=True)
    logits = np.array(out["logits"])
    soft = np.exp(logits - logits.max())
    soft /= soft.sum()
    order = np.argsort(soft)[::-1][:5]
    assert [p["label"] for p in out["predictions"]] == [LABELS[i] for i in order]
    np.testing.assert_allclose(scores, soft[order], rtol=1e-4)


def _lowest_label(predict, data):
    """Label with the smallest probability (never in the top 5 of 7 classes)."""
    logits = np.array(predict(data)["logits"])
    return LABELS[int(np.argmin(logits))]


def test_torch_predict_fn_target_outside_topk_replaces_lowest_entry():
    model, preprocess = _model_and_preprocess()
    predict = torch_support.torch_predict_fn(model, preprocess, LABELS, top_k=5)
    data = (np.random.RandomState(2).rand(8 * 8 * 3) * 255).astype(np.uint8)
    plain = predict(data)["predictions"]
    target = _lowest_label(predict, data)
    assert target not in [p["label"] for p in plain]
    out = predict(data, {"target": target})["predictions"]
    assert len(out) == 5
    assert out[:4] == plain[:4]
    assert out[4]["label"] == target
    scores = [p["score"] for p in out]
    assert scores == sorted(scores, reverse=True)
    # a target already in the top-k changes nothing
    assert predict(data, {"target": plain[2]["label"]})["predictions"] == plain


def test_torch_predict_fn_rejects_unknown_target():
    model, preprocess = _model_and_preprocess()
    predict = torch_support.torch_predict_fn(model, preprocess, LABELS)
    with pytest.raises(ValueError, match="zzz"):
        predict(np.zeros(8 * 8 * 3, dtype=np.uint8), {"target": "zzz"})


def test_torch_predict_fn_output_is_accepted_by_optiattack():
    nut_request = pytest.importorskip("core.utils.nut_request")
    nut_interface = pytest.importorskip("core.search.algorithms.baseline.nut_interface")
    model, preprocess = _model_and_preprocess()
    predict = torch_support.torch_predict_fn(model, preprocess, LABELS, top_k=5)
    data = (np.random.RandomState(2).rand(8 * 8 * 3) * 255).astype(np.uint8)
    target = _lowest_label(predict, data)
    out = predict(data, {"target": target})
    request = nut_request.NutRequest(out, target=target)
    assert request.max_score.label == out["predictions"][0]["label"]
    assert request.target_score.label == target and request.target_score.value > 0.0
    probs = nut_interface.to_probability_vector(request, LABELS)
    assert probs.shape == (len(LABELS),) and abs(probs.sum() - 1.0) < 1e-6
    for p in out["predictions"]:
        assert probs[LABELS.index(p["label"])] == pytest.approx(p["score"])
    logits = nut_interface.to_logit_vector(request, LABELS)
    np.testing.assert_allclose(logits, out["logits"])


def test_torch_predict_fn_explicit_input_shape():
    model, preprocess = _model_and_preprocess()
    predict = torch_support.torch_predict_fn(model, preprocess, LABELS, input_shape=(8, 8, 3))
    out = predict(np.zeros(8 * 8 * 3, dtype=np.uint8))
    assert len(out["logits"]) == len(LABELS)


def test_missing_torch_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pip install optiattack_client\[torch\]"):
        torch_support.torch_gradient_fn(object(), lambda x: x, LABELS)
    with pytest.raises(ImportError, match=r"pip install optiattack_client\[torch\]"):
        torch_support.torch_predict_fn(object(), lambda x: x, LABELS)
