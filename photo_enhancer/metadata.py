"""Carry photo metadata (EXIF and XMP) from the input to the enhanced output.

Date taken, camera, lens, exposure and GPS survive, so photo libraries keep
sorting and mapping enhanced copies correctly. Anything the enhancement makes
wrong is fixed or dropped:

- Orientation is already baked into the pixels by ``load_image`` (Pillow's
  ``exif_transpose`` also clears the tag from EXIF and XMP), so it's never
  applied twice.
- The recorded pixel dimensions are updated to the new size.
- The embedded thumbnail (EXIF IFD1) shows the old image, so it is not
  written. Pillow leaves it out when re-encoding.

``mode`` controls privacy: "keep" everything, "no-gps" drops location data
from both EXIF and XMP, "strip" writes no EXIF or XMP at all. The ICC color
profile is always kept, since it isn't personal and colors depend on it.
"""

from __future__ import annotations

import re

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import __version__

MODES = ("keep", "no-gps", "strip")

TAG_SOFTWARE = 0x0131
TAG_ORIENTATION = 0x0112
IFD_EXIF = 0x8769
IFD_GPS = 0x8825
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003

XMP_PNG_KEY = "XML:com.adobe.xmp"
# exif:GPSLatitude="..." attributes and <exif:GPSLatitude>...</exif:GPSLatitude> elements.
_XMP_GPS_ATTR = re.compile(rb'\s(?:exif|exifEX):GPS\w*="[^"]*"')
_XMP_GPS_ELEM = re.compile(rb"<((?:exif|exifEX):GPS\w*)\b[^>]*?(?:/>|>.*?</\1>)", re.DOTALL)


def read(im: Image.Image) -> dict:
    """Pull EXIF and XMP out of an opened (and already transposed) image."""
    exif = im.getexif()
    xmp = im.info.get("xmp") or im.info.get(XMP_PNG_KEY)
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    return {"exif": exif if len(exif) else None, "xmp": xmp or None}


def strip_xmp_gps(xmp: bytes) -> bytes | None:
    """Remove location from XMP. Returns None if any GPS data might remain."""
    cleaned = _XMP_GPS_ELEM.sub(b"", _XMP_GPS_ATTR.sub(b"", xmp))
    # Fail closed: if GPS fields survive in a form we didn't anticipate,
    # drop the whole packet rather than leak a location.
    return None if b"GPS" in cleaned else cleaned


def save_kwargs(info: dict | None, fmt: str, size: tuple[int, int], mode: str = "keep") -> dict:
    """Keyword arguments for ``Image.save`` that write the metadata in ``info``."""
    if mode not in MODES:
        raise ValueError(f"metadata must be one of: {', '.join(MODES)}")
    if not info or mode == "strip":
        return {}

    kwargs: dict = {}
    exif: Image.Exif | None = info.get("exif")
    if exif is not None:
        out = Image.Exif()
        out.update(exif)
        for ifd in (IFD_EXIF, IFD_GPS):
            if ifd in out:
                del out[ifd]
        # Nested IFDs are written from dict values under their pointer tags.
        exif_ifd = dict(exif.get_ifd(IFD_EXIF))
        if exif_ifd:
            if TAG_PIXEL_X in exif_ifd or TAG_PIXEL_Y in exif_ifd:
                exif_ifd[TAG_PIXEL_X], exif_ifd[TAG_PIXEL_Y] = size
            out[IFD_EXIF] = exif_ifd
        gps_ifd = dict(exif.get_ifd(IFD_GPS))
        if gps_ifd and mode == "keep":
            out[IFD_GPS] = gps_ifd
        out.pop(TAG_ORIENTATION, None)
        out[TAG_SOFTWARE] = f"photo-enhancer {__version__}"
        kwargs["exif"] = out

    xmp: bytes | None = info.get("xmp")
    if xmp and mode == "no-gps":
        xmp = strip_xmp_gps(xmp)
    if xmp:
        if fmt == "PNG":
            png_info = PngInfo()
            png_info.add_itxt(XMP_PNG_KEY, xmp.decode("utf-8", "replace"), zip=False)
            kwargs["pnginfo"] = png_info
        elif fmt in ("JPEG", "WEBP"):
            kwargs["xmp"] = xmp
    return kwargs
