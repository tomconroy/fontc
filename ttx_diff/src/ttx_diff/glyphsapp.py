#!/usr/bin/env python3
"""Compare fontc against Glyphs.app's own exporter (or any two builds).

Glyphs is the reference implementation of the .glyphs format, so the font
Glyphs itself exports is ground truth for how fontc should read a source. This
drives Glyphs headlessly through glyphs-cli (https://pypi.org/project/glyphs-cli/,
macOS only, needs an installed and licensed Glyphs.app), builds the equivalent
font with fontc, and compares the two:

1. ttx tables, with ttx_diff's normalizations plus a few for Glyphs (see
   `normalize_glyphsapp_noise`): identical/different per table, and a unified
   diff of each different table written next to the dumps.
2. behaviour, because Glyphs and fontc legitimately pack tables differently:
   harfbuzz shaping of sample strings (glyph names, advances, offsets) and
   per-glyph advances and outline bounds, at every master and instance
   location of a variable font (instantiated with fontTools' instancer).

A side is `fontc` or `glyphsapp:<build>` (`glyphsapp:4108`, `glyphsapp:3`);
each side can read its own source, so the same harness compares Glyphs 4 on
a format-4 source against Glyphs 3 on its format-3 twin.

Modes:
    variable    variable TTF; fontc's default build vs Glyphs' variable instance
                (one is synthesized if the source has none)
    static-tt   static TTF instances; `fontc --instance NAME`
    static-cff  static CFF instances; `fontc --instance NAME --flavor otf`

Usage:
    python -m ttx_diff.glyphsapp --mode variable --reference glyphsapp:4108 \\
        --fontc_path fontc --normalizer_path otl-normalizer --outdir out \\
        WghtVar.glyphs

    # Glyphs 4 on the v4 source vs Glyphs 3.5 on its v3 twin
    python -m ttx_diff.glyphsapp --mode static-cff --compiler glyphsapp:4108 \\
        --reference glyphsapp:3532 --reference_source v3/WghtVar.glyphs \\
        v4/WghtVar.glyphs

Exit codes: 0 identical (tables and behaviour), 2 different or a build failed.
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from absl import app, flags
from fontTools.ttLib import TTFont
from lxml import etree

from ttx_diff import core
from ttx_diff.core import BuildFail, eprint, rel_user

FLAGS = flags.FLAGS

MODES = ("variable", "static-tt", "static-cff")

# Where glyphs-cli finds a Python framework for `glyphs run`; Glyphs 4 keeps its
# own, Glyphs 3 uses the GlyphsPythonPlugin one. Override with --glyphs_python.
_GLYPHS_PYTHON_GLOBS = {
    "4": "~/Library/Application Support/Glyphs 4/Repositories/Python/Python.framework/Versions/*/Python",
    "3": "~/Library/Application Support/Glyphs 3/Repositories/GlyphsPythonPlugin/Python.framework/Versions/*/Python",
}

# Run inside Glyphs (`glyphs run`) when the source has no variable instance:
# `glyphs export` only exports instances that exist in the file, and a
# `--script` runs after the instance list is fixed, so we generate directly.
_GENERATE_VARIABLE_SCRIPT = r"""
import json, os
from GlyphsApp import Glyphs, GSInstance, INSTANCETYPEVARIABLE, VARIABLE, PLAIN
opts = json.loads(os.environ["GLYPHSAPP_REF_OPTS"])
font = Glyphs.font
inst = next((i for i in font.instances if i.type == INSTANCETYPEVARIABLE), None)
synthesized = inst is None
if synthesized:
    inst = GSInstance.alloc().initWithType_(INSTANCETYPEVARIABLE)
    inst.name = "Regular"
    font.instances.append(inst)
os.makedirs(opts["out"], exist_ok=True)
result = inst.generate(
    format=VARIABLE,
    fontPath=opts["out"],
    autoHint=opts["autohint"],
    removeOverlap=opts["removeOverlap"],
    useSubroutines=opts["useSubroutines"],
    useProductionNames=opts["useProductionNames"],
    containers=[PLAIN],
)
with open(opts["report"], "w") as f:
    json.dump({"result": result if result in (None, True) else str(result),
               "synthesized": synthesized, "instanceName": inst.name}, f)
"""


def define_flags():
    flags.DEFINE_enum("mode", "variable", MODES, "What to build and compare.")
    flags.DEFINE_string("compiler", "fontc", "Side A: 'fontc' or 'glyphsapp:<build>'.")
    flags.DEFINE_string(
        "reference", "glyphsapp:4", "Side B: 'fontc' or 'glyphsapp:<build>'."
    )
    flags.DEFINE_string(
        "reference_source", None, "Source for side B, if not the positional one."
    )
    flags.DEFINE_string(
        "instance",
        "all",
        "Static modes: a named instance (style name), a comma-separated list, "
        "or 'all' (every static instance a Glyphs side exports).",
    )
    flags.DEFINE_string("fontc_path", None, "fontc binary (default: PATH).")
    flags.DEFINE_string(
        "normalizer_path", None, "otl-normalizer binary (default: PATH)."
    )
    flags.DEFINE_string("glyphs_path", None, "glyphs-cli binary (default: PATH).")
    flags.DEFINE_multi_string(
        "glyphs_python",
        [],
        "Python framework for `glyphs run`, as BUILD=PATH (or a bare PATH for "
        "every build). Defaults to the frameworks Glyphs 3/4 install.",
    )
    flags.DEFINE_string("outdir", None, "Directory for builds and comparisons.")
    flags.DEFINE_bool("rebuild", False, "Rebuild fonts that already exist in outdir.")
    flags.DEFINE_bool(
        "production_names",
        True,
        "Production glyph names on both sides (Glyphs useProductionNames / "
        "fontc's default); false compares nice names.",
    )
    flags.DEFINE_bool(
        "behaviour", True, "Also shape strings and compare glyph metrics/bounds."
    )
    flags.DEFINE_float(
        "tolerance",
        1.0,
        "Behaviour check: a bounds or position difference must exceed this many "
        "units to count.",
    )
    flags.DEFINE_integer(
        "show_diff", 40, "Print up to this many lines of each table diff."
    )
    flags.DEFINE_bool("json", False, "Print the result as JSON.")
    # read by ttx_diff.core's normalizations
    flags.DEFINE_float(
        "off_by_one_budget",
        0.1,
        "Percentage of glyf/gvar values allowed to differ by one without counting.",
    )
    flags.DEFINE_bool("unwrap_extensions", True, "Strip Extension lookup wrappers.")


# ---------------------------------------------------------------- building


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


def build_key(compiler: str, source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:8]
    return f"{_slug(compiler)}-{_slug(source.name)}-{digest}"


def font_ext(mode: str) -> str:
    return ".otf" if mode == "static-cff" else ".ttf"


def glyphs_cli() -> str:
    path = FLAGS.glyphs_path or shutil.which("glyphs")
    if not path:
        sys.exit("No glyphs-cli: pip install glyphs-cli, or pass --glyphs_path")
    return path


def glyphs_python(build: str) -> Optional[str]:
    """The Python framework `glyphs run` should use for this app build."""
    for entry in FLAGS.glyphs_python:
        key, sep, path = entry.partition("=")
        if not sep:
            return entry
        if key == build:
            return path
    import glob

    major = build[0]
    matches = sorted(glob.glob(os.path.expanduser(_GLYPHS_PYTHON_GLOBS.get(major, ""))))
    return matches[-1] if matches else None


def glyphs_settings(mode: str) -> dict:
    return {
        "outlineFormats": ["cff" if mode == "static-cff" else "tt"],
        "containerFormats": ["standard"],
        # fontc neither hints nor removes overlaps; subroutines only add noise
        # (we desubroutinize before comparing anyway)
        "autohint": False,
        "removeOverlap": False,
        "useSubroutines": False,
        "useProductionNames": bool(FLAGS.production_names),
    }


def _run(cmd: Sequence, **kwargs) -> subprocess.CompletedProcess:
    return core.log_and_run([str(c) for c in cmd], **kwargs)


def _read_reports(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _exported(report: dict) -> bool:
    path = report.get("exportFilePath")
    return bool(path) and not report.get("errors") and Path(path).is_file()


def _glyphs_export(
    build: str, source: Path, out_dir: Path, mode: str, selectors: List[dict]
) -> Tuple[subprocess.CompletedProcess, List[dict]]:
    config = {
        "settings": glyphs_settings(mode),
        "sourceFiles": [
            {
                "filePath": str(source),
                "instances": [{"selector": sel} for sel in selectors],
            },
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "export-config.json"
    config_path.write_text(json.dumps(config, indent=2))
    report = out_dir / "export-report.jsonl"
    if report.exists():
        report.unlink()
    # --plugins "": third-party plug-ins must not shape the reference build (and
    # some crash a headless process on exit)
    cmd = [
        glyphs_cli(),
        "export",
        "--app",
        build,
        "--plugins",
        "",
        "--config",
        config_path,
        "-o",
        out_dir,
        "--json-output",
        report,
    ]
    proc = _run(cmd)
    return proc, _read_reports(report)


def _glyphs_generate_variable(build: str, source: Path, out_dir: Path) -> dict:
    python = glyphs_python(build)
    if python is None:
        raise BuildFail(
            ["glyphs", "run"],
            f"no Python framework for Glyphs {build}; pass --glyphs_python",
        )
    gen_dir = out_dir / "generate"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    report = out_dir / "generate-report.json"
    if report.exists():
        report.unlink()
    script = out_dir / "generate_variable.py"
    script.write_text(_GENERATE_VARIABLE_SCRIPT)
    opts = dict(glyphs_settings("variable"), out=str(gen_dir), report=str(report))
    del opts["outlineFormats"], opts["containerFormats"]
    cmd = [
        glyphs_cli(),
        "run",
        "--app",
        build,
        "--plugins",
        "",
        "--python",
        python,
        script,
        "-i",
        source,
    ]
    proc = _run(cmd, env=dict(os.environ, GLYPHSAPP_REF_OPTS=json.dumps(opts)))
    fonts = sorted(gen_dir.glob("*.ttf")) if gen_dir.exists() else []
    if proc.returncode != 0 or not report.is_file() or len(fonts) != 1:
        raise BuildFail(cmd, proc.stdout[-core.MAX_ERR_LEN :] or "no font generated")
    info = json.loads(report.read_text())
    info["path"] = str(fonts[0])
    return info


def build_glyphsapp(
    build: str, source: Path, mode: str, out_dir: Path, instances: Optional[List[str]]
) -> Dict[str, dict]:
    """Export with Glyphs; returns {instance name: {"path", "warnings", "errors", ...}}."""
    if mode == "variable":
        proc, reports = _glyphs_export(
            build,
            source,
            out_dir,
            mode,
            [{"type": "variable", "predicate": "exports == YES"}],
        )
        if proc.returncode != 0 and "No instance matches" in proc.stdout:
            info = _glyphs_generate_variable(build, source, out_dir)
            info.update(warnings=[], errors=[], via="glyphs run (instance.generate)")
            dest = out_dir / "variable.ttf"
            shutil.move(info["path"], dest)
            info["path"] = str(dest)
            return {"variable": info}
        reports = [r for r in reports if _exported(r)]
        if proc.returncode != 0 or not reports:
            raise BuildFail(
                ["glyphs", "export", str(source)], proc.stdout[-core.MAX_ERR_LEN :]
            )
        if len(reports) > 1:
            eprint(f"WARN: {len(reports)} variable instances; comparing the first")
        r = reports[0]
        dest = out_dir / "variable.ttf"
        shutil.move(r["exportFilePath"], dest)
        return {"variable": dict(r, path=str(dest), via="glyphs export")}

    # a selector exports instances even when they're switched off, so 'all'
    # asks for what the app's own File > Export would ship
    selectors = (
        [{"type": "single", "predicate": "exports == YES"}]
        if instances is None
        else [{"type": "single", "name": n} for n in instances]
    )
    proc, reports = _glyphs_export(build, source, out_dir, mode, selectors)
    if not reports:
        raise BuildFail(
            ["glyphs", "export", str(source)], proc.stdout[-core.MAX_ERR_LEN :]
        )
    result = {}
    for r in reports:
        name = r["instanceName"]
        if not _exported(r):
            # Glyphs names a path even when the export failed; the errors say why
            result[name] = dict(r, path=None)
            continue
        dest = out_dir / f"{_slug(name)}{font_ext(mode)}"
        shutil.move(r["exportFilePath"], dest)
        result[name] = dict(r, path=str(dest), via="glyphs export")
    return result


def build_fontc(
    fontc_bin: Path, source: Path, mode: str, out_dir: Path, instances: List[str]
) -> Dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [None] if mode == "variable" else instances
    result = {}
    for name in targets:
        stem = "variable" if name is None else _slug(name)
        out = out_dir / f"{stem}{font_ext(mode)}"
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [fontc_bin, "--build-dir", tmp, "-o", out, source]
            if not FLAGS.production_names:
                cmd.append("--no-production-names")
            if name is not None:
                cmd += ["--instance", name]
            if mode == "static-cff":
                cmd += ["--flavor", "otf"]
            proc = _run(cmd)
        key = name or "variable"
        if proc.returncode != 0 or not out.is_file():
            result[key] = {
                "path": None,
                "errors": [proc.stdout[-core.MAX_ERR_LEN :]],
                "command": " ".join(map(str, cmd)),
            }
        else:
            result[key] = {"path": str(out), "errors": [], "warnings": []}
    return result


def build_side(
    compiler: str,
    source: Path,
    mode: str,
    outdir: Path,
    instances: Optional[List[str]],
    fontc_bin: Optional[Path] = None,
) -> Dict[str, dict]:
    """Build (or reuse) one side; cached in outdir/builds/<key>/<mode>/."""
    out_dir = outdir / "builds" / build_key(compiler, source) / mode
    manifest = out_dir / "manifest.json"
    if manifest.is_file() and not FLAGS.rebuild:
        cached = json.loads(manifest.read_text())
        wanted = ["variable"] if mode == "variable" else instances
        if wanted is None or all(n in cached for n in wanted):
            eprint(f"reusing {rel_user(out_dir)}")
            return cached
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        if compiler == "fontc":
            if mode != "variable" and instances is None:
                raise BuildFail(["fontc"], "fontc needs explicit --instance names")
            result = build_fontc(fontc_bin, source, mode, out_dir, instances)
        elif compiler.startswith("glyphsapp:"):
            result = build_glyphsapp(
                compiler.split(":", 1)[1], source, mode, out_dir, instances
            )
        else:
            raise ValueError(f"unknown compiler '{compiler}'")
    except BuildFail as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {"__failure__": {"command": " ".join(e.command), "stderr": e.msg}}
    manifest.write_text(json.dumps(result, indent=2))
    return result


# ------------------------------------------------------------ table compare


def prepare_for_ttx(src: Path, dest: Path):
    """Copy a font for dumping; CFF is desubroutinized so packing doesn't matter."""
    font = TTFont(src)
    if "CFF " in font:
        font["CFF "].cff.desubroutinize()
    if "CFF2" in font:
        font["CFF2"].cff.desubroutinize()
    font.save(dest)


def _drop(ttx, xpath):
    for el in ttx.xpath(xpath):
        el.getparent().remove(el)


def normalize_glyphsapp_noise(ttx: etree.ElementTree):
    """Things that differ for reasons unrelated to how a source is read."""
    # timestamps: Glyphs ignores SOURCE_DATE_EPOCH
    for attr in ("created", "modified"):
        for el in ttx.xpath(f"//head/{attr}"):
            el.attrib["value"] = "(normalized)"
    # version strings and unique ids name the compiler/its version
    for name_id in ("3", "5"):
        for el in ttx.xpath(f"//name/namerecord[@nameID='{name_id}']"):
            el.text = "(normalized)"
    # Glyphs writes a stub DSIG into statics; fontc never does
    _drop(ttx, "/ttFont/DSIG")
    # glyph ids differ when glyph order does; GlyphOrder itself is still compared
    # through the 'GlyphOrder' table entry


def generate_table_output(
    a_font: Path, b_font: Path, work: Path, normalizer: Path
) -> Dict[str, Dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, src in (("a", a_font), ("b", b_font)):
        dest = work / f"{label}{src.suffix}"
        for stale in work.glob(f"{label}.*"):
            stale.unlink()
        prepare_for_ttx(src, dest)
        paths[label] = dest
    ttx = {k: core.run_ttx(p) for k, p in paths.items()}
    gpos = {k: core.run_normalizer(normalizer, p, "gpos") for k, p in paths.items()}
    gdef = {k: core.run_normalizer(normalizer, p, "gdef") for k, p in paths.items()}
    a = etree.parse(ttx["a"])
    b = etree.parse(ttx["b"])
    core.fill_in_gvar_deltas(a, paths["a"], b, paths["b"])
    for t in (a, b):
        normalize_glyphsapp_noise(t)
    # ttx_diff's normalizations treat side 2 as fontmake (to be made to look
    # like fontc); neither side is fontc-shaped here, so apply those to both
    glyph_map_a = {
        el.attrib["name"]: int(el.attrib["id"])
        for el in a.xpath("//GlyphOrder/GlyphID")
    }
    core.sort_indices(a, "GPOS", "//Feature", "LookupListIndex")
    core.sort_indices(a, "GSUB", "//LangSys", "FeatureIndex")
    core.sort_indices(a, "GSUB", "//DefaultLangSys", "FeatureIndex")
    core.reorder_contextual_class_based_rules(a, "GSUB", glyph_map_a)
    core.reorder_contextual_class_based_rules(a, "GPOS", glyph_map_a)
    core.reduce_diff_noise(a, b)
    out = {
        "a": core.extract_comparables(a, work, "a"),
        "b": core.extract_comparables(b, work, "b"),
    }
    for k in ("a", "b"):
        out[k][core.MARK_KERN_NAME] = gpos[k]
        if gdef[k]:
            out[k][core.LIG_CARET_NAME] = gdef[k]
    return out


def _as_text(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


def compare_tables(output: Dict[str, Dict[str, str]], work: Path) -> dict:
    a, b = output["a"], output["b"]
    for stale in work.glob("*.diff"):
        stale.unlink()
    tables = {}
    for tag in sorted(set(a) | set(b)):
        if tag not in b:
            tables[tag] = {"status": "only_a"}
        elif tag not in a:
            tables[tag] = {"status": "only_b"}
        elif a[tag] == b[tag]:
            tables[tag] = {"status": "identical"}
        else:
            sa, sb = _as_text(a[tag]), _as_text(b[tag])
            diff = list(
                difflib.unified_diff(
                    sa.splitlines(), sb.splitlines(), "a", "b", lineterm="", n=2
                )
            )
            diff_path = work / f"{_slug(tag)}.diff"
            diff_path.write_text("\n".join(diff) + "\n")
            tables[tag] = {
                "status": "different",
                "ratio": core.diff_ratio(sa, sb),
                "diff": str(diff_path),
                "diff_lines": diff,
            }
    return tables


# ---------------------------------------------------------- behaviour check

# hb turns these on by default; anything else in GSUB/GPOS is shaped with the
# feature explicitly enabled
_HB_DEFAULT_FEATURES = {
    "abvm",
    "blwm",
    "calt",
    "ccmp",
    "clig",
    "curs",
    "dist",
    "kern",
    "liga",
    "locl",
    "mark",
    "mkmk",
    "rclt",
    "rlig",
    "rvrn",
    "abvs",
    "blws",
    "pres",
    "psts",
    "akhn",
    "rphf",
    "blwf",
    "half",
    "pstf",
    "vatu",
    "cjct",
    "init",
    "medi",
    "fina",
    "isol",
    "med2",
    "fin2",
    "fin3",
    "stch",
    "ltra",
    "ltrm",
    "rtla",
    "rtlm",
    "frac",
    "numr",
    "dnom",
}


def _avar_inverse(font: TTFont, tag: str, value: float) -> float:
    from fontTools.varLib.models import piecewiseLinearMap

    if "avar" not in font:
        return value
    segments = font["avar"].segments.get(tag)
    if not segments:
        return value
    return piecewiseLinearMap(value, {v: k for k, v in segments.items()})


def _denormalize(axis, v: float) -> float:
    if v < 0:
        return axis.defaultValue + v * (axis.defaultValue - axis.minValue)
    return axis.defaultValue + v * (axis.maxValue - axis.defaultValue)


def variation_locations(font: TTFont) -> List[Dict[str, float]]:
    """User-space locations worth checking: default, named instances, masters.

    Masters are recovered from the peaks of gvar/HVAR regions, mapped back
    through avar, so a misplaced master in one font shows up as a location the
    other font is checked at too.
    """
    if "fvar" not in font:
        return []
    axes = font["fvar"].axes
    locs = [{a.axisTag: a.defaultValue for a in axes}]
    for inst in font["fvar"].instances:
        locs.append(dict(inst.coordinates))
    peaks = set()
    if "gvar" in font:
        for variations in font["gvar"].variations.values():
            for tv in variations:
                peaks.add(
                    tuple(
                        sorted((tag, p[1]) for tag, p in tv.axes.items() if p[1] != 0)
                    )
                )
    for peak in peaks:
        loc = {a.axisTag: a.defaultValue for a in axes}
        for tag, norm in peak:
            axis = next(a for a in axes if a.axisTag == tag)
            loc[tag] = round(_denormalize(axis, _avar_inverse(font, tag, norm)), 3)
        locs.append(loc)
    unique = []
    for loc in locs:
        key = {k: round(v, 3) for k, v in loc.items()}
        if key not in unique:
            unique.append(key)
    return unique


def _instantiate(font_path: Path, loc: Optional[Dict[str, float]]) -> TTFont:
    from fontTools.varLib import instancer

    font = TTFont(font_path)
    if loc is None or "fvar" not in font:
        return font
    limits = {a.axisTag: loc.get(a.axisTag, a.defaultValue) for a in font["fvar"].axes}
    return instancer.instantiateVariableFont(font, limits, static=True)


def glyph_metrics(font: TTFont) -> Dict[str, Tuple[int, Optional[Tuple[float, ...]]]]:
    from fontTools.pens.boundsPen import BoundsPen

    glyphset = font.getGlyphSet()
    out = {}
    for name in font.getGlyphOrder():
        pen = BoundsPen(glyphset)
        glyphset[name].draw(pen)
        out[name] = (font["hmtx"][name][0], pen.bounds)
    return out


def sample_runs(fonts: Iterable[TTFont]) -> List[Tuple[str, Tuple[str, ...]]]:
    """(text, features) pairs: every char, every pair, the whole set, and the
    whole set with each non-default layout feature switched on."""
    chars, features = set(), set()
    for font in fonts:
        chars |= set(font.getBestCmap() or {})
        for tag in ("GSUB", "GPOS"):
            if tag in font and font[tag].table.FeatureList:
                features |= {
                    r.FeatureTag for r in font[tag].table.FeatureList.FeatureRecord
                }
    text = [chr(c) for c in sorted(chars) if c > 0x20]
    runs = [(c, ()) for c in text]
    pairs = text[:40]
    runs += [(x + y, ()) for x in pairs for y in pairs]
    whole = "".join(text)
    runs.append((whole, ()))
    for feat in sorted(features - _HB_DEFAULT_FEATURES):
        runs.append((whole, (feat,)))
    return runs


def shape(font_path: Path, order: List[str], text: str, feats, loc) -> list:
    import uharfbuzz as hb

    blob = hb.Blob.from_file_path(str(font_path))
    hb_font = hb.Font(hb.Face(blob))
    if loc:
        hb_font.set_variations(loc)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {f: True for f in feats})
    return [
        (
            order[i.codepoint] if i.codepoint < len(order) else f"gid{i.codepoint}",
            i.cluster,
            p.x_advance,
            p.y_advance,
            p.x_offset,
            p.y_offset,
        )
        for i, p in zip(buf.glyph_infos, buf.glyph_positions)
    ]


def _shaped_equal(x, y, tol) -> bool:
    if len(x) != len(y):
        return False
    for gx, gy in zip(x, y):
        if gx[:2] != gy[:2]:
            return False
        if any(abs(p - q) > tol for p, q in zip(gx[2:], gy[2:])):
            return False
    return True


def compare_behaviour(a_path: Path, b_path: Path, examples: int = 5) -> dict:
    try:
        import uharfbuzz  # noqa: F401
    except ImportError:
        return {"skipped": "uharfbuzz is not installed (pip install uharfbuzz)"}
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    tol = FLAGS.tolerance
    a_font, b_font = TTFont(a_path), TTFont(b_path)
    locs = [None]
    if "fvar" in a_font or "fvar" in b_font:
        locs = []
        for loc in variation_locations(a_font) + variation_locations(b_font):
            if loc not in locs:
                locs.append(loc)
    a_order, b_order = a_font.getGlyphOrder(), b_font.getGlyphOrder()
    runs = sample_runs([a_font, b_font])
    result = {
        "locations": locs,
        "runs_per_location": len(runs),
        "cmap": {
            "only_a": sorted(
                f"U+{c:04X}"
                for c in set(a_font.getBestCmap()) - set(b_font.getBestCmap())
            ),
            "only_b": sorted(
                f"U+{c:04X}"
                for c in set(b_font.getBestCmap()) - set(a_font.getBestCmap())
            ),
        },
        "shaping": {"count": 0, "examples": []},
        "glyphs": {"count": 0, "examples": [], "only_a": [], "only_b": []},
    }
    for loc in locs:
        where = (
            "default"
            if not loc
            else ",".join(f"{k}={v:g}" for k, v in sorted(loc.items()))
        )
        for text, feats in runs:
            sa = shape(a_path, a_order, text, feats, loc)
            sb = shape(b_path, b_order, text, feats, loc)
            if not _shaped_equal(sa, sb, tol):
                result["shaping"]["count"] += 1
                if len(result["shaping"]["examples"]) < examples:
                    result["shaping"]["examples"].append(
                        {
                            "at": where,
                            "text": text[:40],
                            "features": list(feats),
                            "a": sa[:8],
                            "b": sb[:8],
                        }
                    )
        ma = glyph_metrics(_instantiate(a_path, loc))
        mb = glyph_metrics(_instantiate(b_path, loc))
        result["glyphs"]["only_a"] = sorted(set(ma) - set(mb))
        result["glyphs"]["only_b"] = sorted(set(mb) - set(ma))
        for name in sorted(set(ma) & set(mb)):
            (adv_a, box_a), (adv_b, box_b) = ma[name], mb[name]
            box_differs = (box_a is None) != (box_b is None) or (
                box_a is not None
                and any(abs(p - q) > tol for p, q in zip(box_a, box_b))
            )
            if abs(adv_a - adv_b) > tol or box_differs:
                result["glyphs"]["count"] += 1
                if len(result["glyphs"]["examples"]) < examples:
                    result["glyphs"]["examples"].append(
                        {
                            "at": where,
                            "glyph": name,
                            "a": [adv_a, box_a],
                            "b": [adv_b, box_b],
                        }
                    )
    return result


def behaviour_is_clean(beh: dict) -> bool:
    if "skipped" in beh:
        return True
    return not (
        beh["shaping"]["count"]
        or beh["glyphs"]["count"]
        or beh["glyphs"]["only_a"]
        or beh["glyphs"]["only_b"]
        or beh["cmap"]["only_a"]
        or beh["cmap"]["only_b"]
    )


# ------------------------------------------------------------------- driver


def compare_fonts(
    a_font: Path, b_font: Path, work: Path, normalizer: Path, behaviour=True
) -> dict:
    output = generate_table_output(a_font, b_font, work, normalizer)
    tables = compare_tables(output, work)
    result = {"tables": tables}
    if behaviour:
        result["behaviour"] = compare_behaviour(a_font, b_font)
    result["identical_tables"] = all(
        t["status"] == "identical" for t in tables.values()
    )
    result["clean_behaviour"] = behaviour_is_clean(
        result.get("behaviour", {"skipped": "off"})
    )
    return result


def run_comparison(
    a_compiler: str,
    a_source: Path,
    b_compiler: str,
    b_source: Path,
    mode: str,
    outdir: Path,
    instances: Optional[List[str]],
    fontc_bin: Optional[Path],
    normalizer: Path,
    behaviour: bool = True,
) -> dict:
    """Build both sides and compare every instance they share."""
    glyphs_first = sorted(
        [(a_compiler, a_source, "a"), (b_compiler, b_source, "b")],
        key=lambda s: not s[0].startswith("glyphsapp:"),
    )
    built = {}
    names = instances
    for compiler, source, side in glyphs_first:
        built[side] = build_side(compiler, source, mode, outdir, names, fontc_bin)
        if names is None and mode != "variable" and "__failure__" not in built[side]:
            names = [n for n in built[side] if not n.startswith("__")]
    report = {
        "mode": mode,
        "a": {"compiler": a_compiler, "source": str(a_source)},
        "b": {"compiler": b_compiler, "source": str(b_source)},
        "instances": {},
    }
    for side in ("a", "b"):
        if "__failure__" in built[side]:
            report[side]["failure"] = built[side]["__failure__"]
    if any("failure" in report[s] for s in ("a", "b")):
        return report
    for name in names or ["variable"]:
        entry = {}
        fa, fb = built["a"].get(name, {}), built["b"].get(name, {})
        for side, info in (("a", fa), ("b", fb)):
            entry[f"{side}_build"] = {
                k: info.get(k) for k in ("path", "errors", "warnings", "via")
            }
        if not fa.get("path") or not fb.get("path"):
            missing = [s for s, i in (("a", fa), ("b", fb)) if not i.get("path")]
            entry["failure"] = f"no font from {' and '.join(missing)}"
            report["instances"][name] = entry
            continue
        work = (
            outdir
            / "compare"
            / f"{build_key(a_compiler, a_source)}__{build_key(b_compiler, b_source)}"
            / mode
            / _slug(name)
        )
        entry.update(
            compare_fonts(
                Path(fa["path"]), Path(fb["path"]), work, normalizer, behaviour
            )
        )
        report["instances"][name] = entry
    return report


def print_report(report: dict):
    print(f"MODE {report['mode']}")
    for side in ("a", "b"):
        s = report[side]
        print(f"  {side}: {s['compiler']} {rel_user(s['source'])}")
        if "failure" in s:
            print(f"     FAILED: {s['failure']['stderr'][-500:]}")
    for name, entry in report["instances"].items():
        print(f"INSTANCE {name}")
        for side in ("a", "b"):
            b = entry.get(f"{side}_build", {})
            for kind in ("errors", "warnings"):
                for msg in b.get(kind) or []:
                    print(f"  {side} {kind[:-1]}: {msg}")
        if "failure" in entry:
            print(f"  {entry['failure']}")
            continue
        for tag, t in entry["tables"].items():
            if t["status"] == "identical":
                print(f"  Identical '{tag}'")
            elif t["status"] == "different":
                print(
                    f"  DIFF '{tag}' ({t['ratio']:.3%} similar) {rel_user(t['diff'])}"
                )
                lines = t["diff_lines"][2:]
                for line in lines[: FLAGS.show_diff]:
                    print(f"      {line}")
                if len(lines) > FLAGS.show_diff:
                    print(f"      ... {len(lines) - FLAGS.show_diff} more lines")
            else:
                print(f"  Only {t['status'][-1]} produced '{tag}'")
        beh = entry.get("behaviour")
        if beh is None:
            continue
        if "skipped" in beh:
            print(f"  BEHAVIOUR skipped: {beh['skipped']}")
            continue
        print(
            f"  BEHAVIOUR at {len(beh['locations'])} location(s) x {beh['runs_per_location']} runs: "
            f"{beh['shaping']['count']} shaping differences, "
            f"{beh['glyphs']['count']} glyph metric/bounds differences"
        )
        for key in ("only_a", "only_b"):
            if beh["cmap"][key]:
                print(f"    cmap {key}: {beh['cmap'][key][:10]}")
            if beh["glyphs"][key]:
                print(f"    glyphs {key}: {beh['glyphs'][key][:10]}")
        for ex in beh["shaping"]["examples"] + beh["glyphs"]["examples"]:
            print(f"    {json.dumps(ex)}")


def report_is_clean(report: dict) -> bool:
    if any("failure" in report[s] for s in ("a", "b")) or not report["instances"]:
        return False
    return all(
        "failure" not in e and e["identical_tables"] and e["clean_behaviour"]
        for e in report["instances"].values()
    )


def _jsonable(report: dict) -> dict:
    out = json.loads(json.dumps(report, default=str))
    for entry in out.get("instances", {}).values():
        for t in entry.get("tables", {}).values():
            t.pop("diff_lines", None)
    return out


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: python -m ttx_diff.glyphsapp [flags] SOURCE")
    source = Path(argv[1]).resolve()
    ref_source = (
        Path(FLAGS.reference_source).resolve() if FLAGS.reference_source else source
    )
    for s in (source, ref_source):
        if not s.exists():
            sys.exit(f"No such source: {s}")
    fontc_bin = None
    if "fontc" in (FLAGS.compiler, FLAGS.reference):
        fontc_bin = Path(
            FLAGS.fontc_path or shutil.which("fontc") or sys.exit("No fontc")
        )
    normalizer = Path(
        FLAGS.normalizer_path
        or shutil.which("otl-normalizer")
        or sys.exit("No otl-normalizer")
    )
    if shutil.which("ttx") is None:
        sys.exit("No ttx")
    outdir = Path(FLAGS.outdir or "glyphsapp_diff_output").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    instances = (
        None
        if FLAGS.instance == "all"
        else [s.strip() for s in FLAGS.instance.split(",")]
    )
    report = run_comparison(
        FLAGS.compiler,
        source,
        FLAGS.reference,
        ref_source,
        FLAGS.mode,
        outdir,
        instances,
        fontc_bin,
        normalizer,
        FLAGS.behaviour,
    )
    if FLAGS.json:
        print(json.dumps(_jsonable(report), indent=2))
    else:
        print_report(report)
    sys.exit(0 if report_is_clean(report) else 2)


def cli():
    define_flags()
    app.run(main)


if __name__ == "__main__":
    cli()
