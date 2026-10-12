# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-page staging for KVarN MLA cache stores.

VarN + per-channel RTN need the full 256-token tile (== one vLLM page), but
decode steps write single tokens into partially-filled pages. The shadow
buffers the rotated latents of a request's incomplete last page and emits
WHOLE pages for quantization: on every store touching an incomplete page the
entire page is re-quantized from shadow+new tokens and rewritten; complete
pages are emitted once and never revisited.

Lifecycle contract (KvarnPageShadowPool):
- ``stage_update(latents, slots)`` returns a list of
  ``(page_base_slot, page_latents [group, latent_dim], n_valid)`` for every
  page that must be (re)written by the writer kernel this call, in slot order.
- ``reset_page(page_base)`` / ``reset_all()`` on preemption, eviction, or
  page-table reassignment (the caller owns those events).
- The shadow holds *rotated* latents exactly as the writer should see them.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.kvarn_mla import KvarnMlaLayout

__all__ = ["KvarnPageShadowPool"]


class KvarnPageShadowPool:
    """Per-impl shadow of incomplete pages, in rotated-latent space."""

    def __init__(self, layout: KvarnMlaLayout, device: torch.device) -> None:
        self.layout = layout
        self.device = device
        self.group = layout.group
        self.latent_dim = layout.latent_dim
        # page_base_slot -> (latents [group, latent_dim] bf16, n_valid int)
        self._shadows: dict[int, tuple[torch.Tensor, int]] = {}

    def reset_page(self, page_base_slot: int) -> None:
        self._shadows.pop(page_base_slot, None)

    def reset_all(self) -> None:
        self._shadows.clear()

    def shadow_size(self) -> int:
        """Number of incomplete pages currently buffered."""
        return len(self._shadows)

    @torch.no_grad()
    def stage_update(
        self,
        latents: torch.Tensor,
        slots: torch.Tensor,
    ) -> list[tuple[int, torch.Tensor, int]]:
        """Fold ``latents`` (``[T, latent_dim]``, rotated) at ``slots`` (``[T]``)
        into page shadows and return the whole pages to (re)write.

        Pages fully completed by this call are emitted once and their shadow
        dropped; pages still incomplete are emitted (re-quantized whole) and
        kept.
        """
        assert latents.shape[0] == slots.shape[0]
        latents = latents.to(device=self.device, dtype=torch.bfloat16)
        pages: dict[int, int] = {}  # page_base -> max n_valid after this call
        touched: set[int] = set()
        for i in range(latents.shape[0]):
            slot = int(slots[i])
            page_base = slot - slot % self.group
            offset = slot - page_base
            shadow = self._shadows.get(page_base)
            if shadow is None:
                buf = torch.zeros(
                    self.group, self.latent_dim,
                    dtype=torch.bfloat16, device=self.device,
                )
                self._shadows[page_base] = (buf, 0)
                shadow = (buf, 0)
            buf, n_valid = shadow
            # Records are appended densely: the first write to a page starts
            # at offset 0; a re-store of an existing offset (spec-decode
            # rollback + rewrite) overwrites in place.
            buf[offset] = latents[i]
            if offset + 1 > n_valid:
                n_valid = offset + 1
            self._shadows[page_base] = (buf, n_valid)
            pages[page_base] = n_valid
            touched.add(page_base)

        out: list[tuple[int, torch.Tensor, int]] = []
        for page_base in sorted(pages):
            buf, n_valid = self._shadows[page_base]
            n_valid = pages[page_base]
            out.append((page_base, buf, n_valid))
            if n_valid == self.group:
                # Page complete: emit final and drop the shadow.
                del self._shadows[page_base]
        return out
