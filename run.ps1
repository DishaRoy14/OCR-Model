Remove-Item Env:OCR_MODEL -ErrorAction SilentlyContinue
$env:OCR_NUM_CTX = "4096"
$env:OCR_ENHANCE = "off"
uvicorn main:app --host 127.0.0.1 --port 8000
