"""EXIF/XMP is carried from input to output, with location optionally removed."""

import io

import pytest
from PIL import Image, ImageCms

from photo_enhancer import metadata
from photo_enhancer.pipeline import EnhanceOptions, encode_image, enhance_file, load_image

XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
    b'xmlns:exif="http://ns.adobe.com/exif/1.0/" tiff:Orientation="6" xmp:Rating="4" '
    b'exif:GPSLatitude="51,30.0N"><exif:GPSAltitude>12/1</exif:GPSAltitude></rdf:Description>'
    b"</rdf:RDF></x:xmpmeta>"
)


def camera_jpeg(path, orientation=6):
    """A 64x48 JPEG shot 'sideways' (orientation 6), like a phone in portrait."""
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "EOS 5D"
    exif[0x0112] = orientation
    exif[0x8769] = {0x9003: "2003:07:14 10:20:30", 0xA002: 64, 0xA003: 48}
    exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0), 3: "W", 4: (0.0, 7.0, 0.0)}
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (64, 48), (200, 100, 50)).save(path, "JPEG", exif=exif, xmp=XMP, icc_profile=srgb)
    return path


def reopen(path):
    im = Image.open(path)
    exif = im.getexif()
    xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp") or b""
    return im, exif, xmp.encode() if isinstance(xmp, str) else xmp


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
def test_keep_carries_date_camera_and_location(tmp_path, ext):
    src = camera_jpeg(tmp_path / "in.jpg")
    dst = tmp_path / f"out.{ext}"
    assert enhance_file(src, dst, EnhanceOptions(model="none", scale=2)) == (96, 128)
    im, exif, xmp = reopen(dst)
    assert im.size == (96, 128)  # rotated upright once
    assert exif[0x010F] == "Canon" and exif[0x0110] == "EOS 5D"
    assert 0x0112 not in exif  # rotation already applied to the pixels
    assert exif[0x0131].startswith("photo-enhancer")
    sub = exif.get_ifd(0x8769)
    assert sub[0x9003] == "2003:07:14 10:20:30"
    assert (sub[0xA002], sub[0xA003]) == (96, 128)
    assert exif.get_ifd(0x8825)[1] == "N"
    assert b'xmp:Rating="4"' in xmp and b"GPSLatitude" in xmp
    assert b"tiff:Orientation" not in xmp
    assert im.info.get("icc_profile")


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
def test_no_gps_removes_location_everywhere(tmp_path, ext):
    src = camera_jpeg(tmp_path / "in.jpg")
    dst = tmp_path / f"out.{ext}"
    enhance_file(src, dst, EnhanceOptions(model="none", scale=1), metadata="no-gps")
    _, exif, xmp = reopen(dst)
    assert exif[0x010F] == "Canon"
    assert exif.get_ifd(0x8769)[0x9003] == "2003:07:14 10:20:30"
    assert not exif.get_ifd(0x8825)
    assert b"GPS" not in xmp and b'xmp:Rating="4"' in xmp


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
def test_strip_removes_exif_and_xmp_but_keeps_color_profile(tmp_path, ext):
    src = camera_jpeg(tmp_path / "in.jpg")
    dst = tmp_path / f"out.{ext}"
    enhance_file(src, dst, EnhanceOptions(model="none", scale=1), metadata="strip")
    im, exif, xmp = reopen(dst)
    assert len(exif) == 0 and not xmp
    assert im.info.get("icc_profile")


def test_image_without_metadata_gets_none_added(photo):
    data = encode_image(photo, None, "JPEG", info=load_image(encode_image(photo, None, "PNG"))[2])
    assert len(Image.open(io.BytesIO(data)).getexif()) == 0


def test_xmp_gps_removal_fails_closed():
    assert metadata.strip_xmp_gps(b'<d exif:GPSLatitude="1" xmp:Rating="2"/>') == b'<d xmp:Rating="2"/>'
    # A GPS field in a namespace we don't recognise: drop the whole packet.
    assert metadata.strip_xmp_gps(b"<d><foo:GPSLatitude>1</foo:GPSLatitude></d>") is None


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        metadata.save_kwargs({"exif": None, "xmp": None}, "JPEG", (1, 1), "some")
