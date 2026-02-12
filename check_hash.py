import hashlib
from pathlib import Path

paths = [
    r"D:\tds_parser\pdfs\AM2XIRH02P1051MW.pdf",
    r"D:\tds_parser\pdfs\AM2XIRH04P1051MW.pdf",
    r"D:\tds_parser\pdfs\AM2XIRH08P1051MW.pdf",
]

for p in paths:
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    print(Path(p).name, h)
