# ttx-diff

[![PyPI](https://img.shields.io/pypi/v/ttx-diff)](https://pypi.org/project/ttx-diff/)
[![Python Version](https://img.shields.io/pypi/pyversions/ttx-diff)](https://pypi.org/project/ttx-diff/)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](#license)

A tool for comparing font compiler outputs between fontc (Rust) and fontmake (Python).

## Overview

`ttx-diff` is a helper utility that compares binary font outputs from two different font compilers:
- **fontc**: The Rust-based font compiler from Google Fonts
- **fontmake**: The Python-based font compiler

The tool converts each binary font to TTX (XML) format, normalizes expected
differences, and provides a detailed comparison summary.

Note that `ttx_diff` is a pure Python package (located in `src/ttx_diff/`)
rather than a typical Rust crate within this workspace.

## Installation

### From PyPI

```bash
pip install ttx-diff
```

### From source

```bash
git clone https://github.com/googlefonts/fontc.git
cd fontc/ttx_diff
pip install -e .
```

## Requirements

- Python 3.10 or higher
- `fontc` and `otl-normalizer` binaries (see below)

All Python dependencies (fontmake, fonttools, etc.) are installed automatically.

### Getting fontc and otl-normalizer

The tool needs the `fontc` and `otl-normalizer` binaries. You can:

1. **Specify paths explicitly** (recommended for most users):
   ```bash
   ttx-diff --fontc_path /path/to/fontc --normalizer_path /path/to/otl-normalizer source.glyphs
   ```

2. **Add them to your PATH**: If `fontc` and `otl-normalizer` are in your PATH, they'll be found automatically

3. **Run from fontc repository**: If you run from the fontc repository root, the tool will automatically build the binaries for you

## Usage

**Note**: Unlike the original `ttx_diff.py` script, this standalone version can be run from any directory. You don't need to be in the fontc repository.

Rebuild with both fontmake and fontc and compare:

```bash
ttx-diff --fontc_path /path/to/fontc --normalizer_path /path/to/otl-normalizer path/to/source.glyphs
```

If the binaries are in your PATH:

```bash
ttx-diff path/to/source.glyphs
```

Compare two precompiled fonts directly (without building from source):

```bash
ttx-diff --fontc_font path/to/fontc.ttf --fontmake_font path/to/fontmake.ttf
```

This is useful for comparing font differences when you already have both builds.

Rebuild only fontc's font and reuse existing fontmake output:

```bash
ttx-diff --rebuild fontc path/to/source.glyphs
```

Output results in machine-readable JSON format, as used by the [`fontc_crater`](https://github.com/googlefonts/fontc/tree/main/fontc_crater) tool.

```bash
ttx-diff --json path/to/source.glyphs
```

Compare using gftools build pipeline:

```bash
ttx-diff --compare gftools --config config.yaml path/to/source.glyphs
```

## Comparing against Glyphs.app

`glyphsapp-diff` (`python -m ttx_diff.glyphsapp`) compares fontc against the
font Glyphs.app itself exports, the ground truth for how a `.glyphs` source
should read. It needs **macOS, an installed and licensed Glyphs.app, and
[glyphs-cli](https://pypi.org/project/glyphs-cli/)**, which runs the app's
exporter headlessly (the app doesn't have to be running).
`pip install -e .[glyphsapp]` adds `uharfbuzz` for the behaviour check.

A side is `fontc` or `glyphsapp:<build>` (`glyphsapp:4108`, or `glyphsapp:3`
for the newest Glyphs 3). Each side can read its own source, so it also
compares two Glyphs versions, or two formats of one design:

```bash
# variable TTF: fontc's default build vs Glyphs' variable instance
# (synthesized with `glyphs run` if the source has none)
glyphsapp-diff --mode variable --reference glyphsapp:4108 Font.glyphs

# static TTF: `fontc --instance Bold` vs Glyphs' export of Bold
glyphsapp-diff --mode static-tt --instance Bold --reference glyphsapp:4108 Font.glyphs

# static CFF: `fontc --instance NAME --flavor otf`, for every exported instance
glyphsapp-diff --mode static-cff --instance all --reference glyphsapp:4108 Font.glyphs

# Glyphs 4 on a format 4 source vs Glyphs 3 on its format 3 twin
glyphsapp-diff --mode variable --compiler glyphsapp:4108 --reference glyphsapp:3532 \
  --reference_source v3/Font.glyphs v4/Font.glyphs
```

It reports each table as identical or different, with a unified diff written
next to the normalized dumps. It uses ttx_diff's normalizations plus a few for
Glyphs: timestamps, name IDs 3 and 5 and Glyphs' stub `DSIG` are ignored, and
CFF is desubroutinized. Glyphs exports without autohinting, overlap removal or
subroutines, the way fontc builds, and with third-party plug-ins disabled.

Because the two compilers legitimately pack tables differently, it also checks
behaviour: harfbuzz shapes every character, every pair and each non-default
feature, comparing glyph names, advances and offsets; and each glyph's advance
and outline bounds are compared. A variable font is checked at its default,
every named instance and every master location (instantiated with fontTools'
instancer). Builds are cached in `--outdir`; `--rebuild` forces them.

## Development

Running tests

```bash
pip install -e .[test]
pytest
```

Running tests with coverage

```bash
pytest --cov=ttx_diff --cov-report=html
```

## Releasing

See <https://googlefonts.github.io/python#make-a-release>.
