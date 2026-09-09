"""Masked success training and duplicate-grouped held-out splits."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def grouped_split(questions, seed=42):
    groups = sorted(set(questions))
    if len(groups) < 5:
        raise ValueError("need at least five distinct prompts for 60/20/20 splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    a, b = int(len(groups) * .6), int(len(groups) * .8)
    membership = {q: i for i, part in enumerate((groups[:a], groups[a:b], groups[b:])) for q in part}
    return tuple(np.array([i for i, q in enumerate(questions) if membership[q] == s]) for s in range(3))


def masked_bce(logits, labels):
    mask = torch.isfinite(labels)
    if not mask.any():
        raise ValueError("split/batch has no observed success labels")
    observed = labels[mask]
    if not ((observed == 0) | (observed == 1)).all():
        raise ValueError("success labels must be 0, 1, or NaN (missing)")
    return F.binary_cross_entropy_with_logits(logits[mask], observed)


def fit(hidden, labels, train, validation, epochs=100, lr=.01, seed=42):
    if hidden.shape != (len(labels), 1024) or not np.isfinite(hidden).all():
        raise ValueError("invalid aligned embeddings")
    if labels.ndim != 2 or not np.isfinite(labels[train]).any(axis=0).all():
        raise ValueError("each model needs observed training labels")
    if not np.isfinite(labels[validation]).any(axis=0).all():
        raise ValueError("each model needs observed validation labels")
    if np.intersect1d(train, validation).size or epochs < 1:
        raise ValueError("invalid training/validation split or epochs")
    torch.manual_seed(seed)
    h = torch.tensor(hidden, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    W = torch.zeros((labels.shape[1], 1024), requires_grad=True)
    b = torch.zeros(labels.shape[1], requires_grad=True)
    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-4)
    best, checkpoint, history = float("inf"), None, []
    for epoch in range(epochs):
        loss = masked_bce(h[train] @ W.T + b, y[train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val = masked_bce(h[validation] @ W.T + b, y[validation]).item()
        history.append({"epoch": epoch + 1, "train_bce": loss.item(), "validation_bce": val})
        if val < best:
            best = val
            checkpoint = (W.detach().numpy().copy(), b.detach().numpy().copy(), epoch + 1)
    return (*checkpoint, history)
