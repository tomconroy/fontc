"""
Convert a Glyphs 3 test source (.glyphs or .glyphspackage) to Glyphs file format 4.

Used to make the fixtures in resources/testdata/glyphs4 from their twins in
resources/testdata/glyphs3, so each pair should compile to the same font.
It applies the changes the format 4 spec describes, as Glyphs 4.1 writes them:
https://github.com/schriftgestalt/GlyphsSDK/blob/Glyphs4/GlyphsFileFormat/GlyphsFileFormatv4.md

- `.formatVersion = 4`; multi-line arrays get a trailing comma, short
  numeric tuples go on one line
- `familyName` becomes the `familyNames` property
- instances name themselves with a `styleNames` property and gain an `id`
- colors are normalized to 0..1; `Color Palettes` takes the v4 dict shape
- hints drop `target = up/down`; components without an alignment get -1
- packages move feature code to features/*.fea, kerning to kerning.plist and
  the note to note.md

Usage:
    python resources/scripts/glyphs3_to_glyphs4.py SRC DEST [--localize-axis-names]

DEST's extension picks the flavor: .glyphs (single file) or .glyphspackage.
--localize-axis-names writes axis names as localized `names` only.

Requires openstep_plist (a glyphsLib dependency).
"""

import argparse
import re
import shutil
import uuid
from pathlib import Path

import openstep_plist


_BARE = re.compile(r"^[A-Z_a-z.][A-Z_a-z0-9.]*$")
# keys whose array values Glyphs writes on one line
_COMPACT = {
    "pos",
    "scale",
    "slant",
    "origin",
    "target",
    "place",
    "coordinates",
    "start",
    "end",
    "fillColor",
    "strokeColor",
    "unicode",
}


def load(src: Path):
    def read(path):
        return openstep_plist.load(open(path, encoding="utf-8"), use_numbers=True)

    if src.suffix == ".glyphspackage":
        font = read(src / "fontinfo.plist")
        glyphs = {}
        for glyph_file in sorted((src / "glyphs").glob("*.glyph")):
            glyph = read(glyph_file)
            glyphs[glyph["glyphname"]] = glyph
        order = read(src / "order.plist") if (src / "order.plist").exists() else []
        font["glyphs"] = [glyphs.pop(n) for n in order if n in glyphs] + [
            glyphs[n] for n in sorted(glyphs)
        ]
        return font
    return read(src)


def color_to_v4(color):
    if color and color[0] == "p":
        return ["p", color[1], round(color[2] / 255, 6)]
    return [round(c / 255, 6) for c in color]


def convert_shape_attrs(attr):
    if "fillColor" in attr:
        attr["fillColor"] = color_to_v4(attr["fillColor"])
    gradient = attr.get("gradient")
    if gradient:
        gradient["colors"] = [[color_to_v4(c), o] for c, o in gradient["colors"]]
        if gradient.get("type") == "circle":
            gradient["type"] = "radial"


def convert_layer(layer):
    for shape in layer.get("shapes", []):
        if "ref" in shape:
            shape.setdefault("alignment", -1)
        convert_shape_attrs(shape.get("attr", {}))
    for hint in layer.get("hints", []):
        if hint.get("target") in ("up", "down"):
            del hint["target"]


def convert(font, localize_axis_names):
    font[".formatVersion"] = 4
    font[".appVersion"] = "4107"

    family = font.pop("familyName", None)
    if family is not None:
        props = font.setdefault("properties", [])
        existing = next((p for p in props if p["key"] == "familyNames"), None)
        if existing is None:
            props.insert(
                0, {"key": "familyNames", "values": [{"language": "dflt", "value": family}]}
            )
        elif not any(v["language"] == "dflt" for v in existing["values"]):
            existing["values"].insert(0, {"language": "dflt", "value": family})
        props.sort(key=lambda p: p["key"])

    if localize_axis_names:
        for axis in font.get("axes", []):
            axis["names"] = [{"language": "dflt", "value": axis.pop("name")}]

    for instance in font.get("instances", []):
        name = instance.pop("name", "Regular")
        instance.setdefault("properties", []).insert(
            0, {"key": "styleNames", "values": [{"language": "dflt", "value": name}]}
        )
        instance["properties"].sort(key=lambda p: p["key"])
        instance["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, name)).upper()

    for param in font.get("customParameters", []) + [
        p for m in font.get("fontMaster", []) for p in m.get("customParameters", [])
    ]:
        if param["name"] == "Color Palettes" and isinstance(param["value"], list):
            param["value"] = {
                "palettes": [
                    {"colors": [color_to_v4(c) for c in palette]} for palette in param["value"]
                ]
            }

    if "kerning" in font:
        font["kerningLTR"] = font.pop("kerning")

    for glyph in font.get("glyphs", []):
        for layer in glyph.get("layers", []):
            convert_layer(layer)
            if "background" in layer:
                convert_layer(layer["background"])
    return font


def fmt_number(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(round(value, 6))
    return str(value)


def fmt_string(value):
    if _BARE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def is_scalar(value):
    return not isinstance(value, (dict, list))


def fmt(value, key=None, compact=False):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return fmt_number(value)
    if isinstance(value, str):
        return fmt_string(value)
    if isinstance(value, bytes):
        return "<" + value.hex() + ">"
    if isinstance(value, dict):
        if compact:
            body = "".join(f"{fmt_string(k)} = {fmt(v, k, True)};" for k, v in sorted(value.items()))
            return "{" + body + "}"
        lines = [f"{fmt_string(k)} = {fmt(v, k)};" for k, v in sorted(value.items())]
        return "{\n" + "".join(line + "\n" for line in lines) + "}"
    if isinstance(value, list):
        nodes = key == "nodes"
        one_line = compact or (key in _COMPACT and all(is_scalar(v) for v in value))
        # a color is a tuple too, as are gradient color stops
        one_line = one_line or (key == "colors" and all(is_scalar(v) for v in value))
        if one_line:
            return "(" + ",".join(fmt(v, None, True) for v in value) + ")"
        items = [fmt(v, None, nodes or _is_tuple(v)) for v in value]
        return "(\n" + "".join(item + ",\n" for item in items) + ")"
    raise TypeError(f"can't write {value!r}")


def _is_tuple(value):
    # color stops: (color, offset) and color tuples inside palettes
    return isinstance(value, list) and all(
        is_scalar(v) or (isinstance(v, list) and all(is_scalar(x) for x in v)) for v in value
    )


def write_plist(path: Path, value):
    path.write_text(fmt(value) + "\n", encoding="utf-8")


def feature_files(font):
    """Assign each class, prefix and feature a file in features/, Glyphs 4 style."""
    used = set()
    files = {}
    for kind, prefix, key in (
        ("classes", "@", "name"),
        ("featurePrefixes", "_", "name"),
        ("features", "", "tag"),
    ):
        for entry in font.get(kind, []):
            stem = prefix + entry[key]
            candidate, n = stem, 1
            while candidate.lower() in used:
                n += 1
                candidate = f"{stem}.{n}"
            used.add(candidate.lower())
            code = entry.pop("code", "")
            if not code.endswith("\n"):
                code += "\n"
            entry["file"] = candidate + ".fea"
            files[entry["file"]] = code
    return files


def glyph_file_name(name):
    out = []
    for c in name:
        if c.isupper():
            out.append(c + "_")
        elif c.isalnum() or c in "._-":
            out.append(c)
        else:
            out.append(f"{ord(c):04X}")
    return "".join(out) + ".glyph"


def write_package(font, dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "glyphs").mkdir(parents=True)
    glyphs = font.pop("glyphs")
    write_plist(dest / "order.plist", [g["glyphname"] for g in glyphs])
    for glyph in glyphs:
        write_plist(dest / "glyphs" / glyph_file_name(glyph["glyphname"]), glyph)

    files = feature_files(font)
    if files:
        (dest / "features").mkdir()
        for name, code in files.items():
            (dest / "features" / name).write_text(code, encoding="utf-8")

    kerning = {
        k: font.pop(k)
        for k in ("kerningLTR", "kerningRTL", "kerningVertical", "kerningContext")
        if k in font
    }
    if kerning:
        write_plist(dest / "kerning.plist", kerning)

    note = font.pop("note", "")
    if note:
        (dest / "note.md").write_text(note, encoding="utf-8")
    font.pop("DisplayStrings", None)
    write_plist(dest / "fontinfo.plist", font)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("src", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--localize-axis-names", action="store_true")
    args = parser.parse_args()

    font = convert(load(args.src), args.localize_axis_names)
    if args.dest.suffix == ".glyphspackage":
        write_package(font, args.dest)
    else:
        write_plist(args.dest, font)


if __name__ == "__main__":
    main()
