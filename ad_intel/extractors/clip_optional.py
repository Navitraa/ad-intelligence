from __future__ import annotations

def clip_embed_dim() -> int:
    try:
        import open_clip  # type: ignore
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        return int(model.text_projection.shape[1])  # typical 512
    except Exception:
        try:
            import torch
            import clip  # type: ignore
            model, _ = clip.load("ViT-B/32")
            return int(model.text_projection.shape[1])
        except Exception:
            raise RuntimeError('CLIP not available')
