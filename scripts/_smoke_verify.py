#!/usr/bin/env python3
"""Проверка, что скачанный файл — действительно целый PNG."""
import sys

from PIL import Image

path, model = sys.argv[1], sys.argv[2]
try:
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        size, fmt, mode = im.size, im.format, im.mode
except Exception as exc:
    print("%-18s ФАЙЛ БИТЫЙ: %s" % (model, exc), file=sys.stderr)
    sys.exit(1)

print("%-18s файл ok: %sx%s %s %s" % (model, size[0], size[1], fmt, mode), file=sys.stderr)
