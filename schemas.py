from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    auto = "auto"
    bengali = "bengali"
    hindi = "hindi"
    english = "english"
    mixed = "mixed"


class HealthResponse(BaseModel):
    api: str = Field(..., description="API status")
    ollama_running: bool
    model_available: bool
    model: str
    utils: str = Field(..., description="Preprocessing module set in use")


class ImageOCRResponse(BaseModel):
    filename: str
    language: str
    model: str
    text: str = Field(..., description="Extracted text")
    char_count: int
    elapsed_seconds: float
    txt_url: Optional[str] = Field(None, description="Open this link to download the clean .txt file")


class PageResult(BaseModel):
    page: int
    text: str
    elapsed_seconds: float
    error: Optional[str] = Field(None, description="Set if this page failed")


class PDFOCRResponse(BaseModel):
    filename: str
    language: str
    model: str
    total_pages: int
    pages_processed: int
    pages_truncated: bool
    pages: List[PageResult]
    combined_text: str
    total_elapsed_seconds: float
    txt_url: Optional[str] = Field(None, description="Open this link to download the clean .txt file")
class DocxImageResult(BaseModel):
    image: int
    text: str
    elapsed_seconds: float
    error: Optional[str] = Field(None, description="Set if this image failed")


class DocxOCRResponse(BaseModel):
    filename: str
    language: str
    model: str
    text: str = Field(..., description="Typed text plus OCR text of embedded images, in document order")
    typed_char_count: int
    images_found: int
    images_processed: int
    images_skipped: int = Field(..., description="Icons, tiny or unreadable images that were ignored")
    images_truncated: bool = Field(..., description="True if some images were not processed (max_images)")
    images: List[DocxImageResult]
    total_elapsed_seconds: float
    txt_url: Optional[str] = Field(None, description="Open this link to download the clean .txt file")