from functools import lru_cache

import torch
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from backend.app.core.config import get_settings


@lru_cache(maxsize=1)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_trocr_components() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load TrOCR once and reuse it across requests."""
    settings = get_settings()
    processor = TrOCRProcessor.from_pretrained(settings.trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model)
    model.to(get_device())
    model.eval()
    return processor, model


def ocr_pdf_to_text(pdf_path: str) -> str:
    """Convert a multi-page PDF to text using TrOCR."""
    settings = get_settings()
    convert_kwargs = {}
    if settings.effective_poppler_path:
        convert_kwargs["poppler_path"] = settings.effective_poppler_path
    images = convert_from_path(pdf_path, **convert_kwargs)

    processor, model = get_trocr_components()
    lines = []
    for image in images:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(get_device())
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        lines.append(text)
    return "\n".join(lines)
