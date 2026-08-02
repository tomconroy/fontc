"""Tests for the ttx-diff CLI."""

import subprocess
import sys

import pytest

from ttx_diff.core import output_font_path, source_is_variable


def _write_designspace(tmp_path, axes, sources):
    from fontTools.designspaceLib import (
        AxisDescriptor,
        DesignSpaceDocument,
        SourceDescriptor,
    )

    ds = DesignSpaceDocument()
    for tag, name, mn, df, mx in axes:
        a = AxisDescriptor()
        a.tag, a.name = tag, name
        a.minimum, a.default, a.maximum = mn, df, mx
        ds.addAxis(a)
    for loc in sources:
        s = SourceDescriptor()
        s.location = loc
        ds.addSource(s)
    path = tmp_path / "test.designspace"
    ds.write(path)
    return path


def _write_glyphs(tmp_path, masters, virtual_masters=None, name="test.glyphs"):
    """Create a minimal .glyphs file using glyphsLib and save to tmp_path."""
    from glyphsLib.classes import (
        GSAxis,
        GSCustomParameter,
        GSFont,
        GSFontMaster,
        GSGlyph,
        GSLayer,
    )

    font = GSFont()
    font.familyName = "Test"
    font.upm = 1000
    font.versionMajor = 1
    axis = GSAxis()
    axis.name = "Weight"
    axis.axisTag = "wght"
    font.axes = [axis]
    for val in masters:
        m = GSFontMaster()
        m.axes = [val]
        font.masters.append(m)
    if virtual_masters:
        for vm_val in virtual_masters:
            font.customParameters.append(
                GSCustomParameter(
                    "Virtual Master", [{"Axis": "Weight", "Location": vm_val}]
                )
            )
    glyph = GSGlyph("space")
    glyph.unicode = "0020"
    for m in font.masters:
        layer = GSLayer()
        layer.layerId = m.id
        layer.width = 200
        glyph.layers.append(layer)
    font.glyphs.append(glyph)
    path = tmp_path / name
    font.save(str(path))
    return path


def test_version():
    import ttx_diff

    assert hasattr(ttx_diff, "__version__")
    assert isinstance(ttx_diff.__version__, str)


def test_cli_missing_source():
    """Test CLI with non-existent source file."""
    result = subprocess.run(
        [sys.executable, "-m", "ttx_diff", "/nonexistent/file.glyphs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No such source" in result.stderr


class TestSourceIsVariable:
    def test_ufo_is_static(self, tmp_path):
        ufo = tmp_path / "test.ufo"
        ufo.mkdir()
        (ufo / "metainfo.plist").write_text("")
        assert not source_is_variable(ufo)

    def test_designspace_multiple_sources(self, tmp_path):
        path = _write_designspace(
            tmp_path,
            axes=[("wght", "Weight", 400, 400, 700)],
            sources=[{"Weight": 400}, {"Weight": 700}],
        )
        assert source_is_variable(path)

    def test_designspace_single_source_with_axis_range(self, tmp_path):
        # 1 source but axis has min != max, like NotoSerifMakasar
        # https://github.com/googlefonts/fontc/issues/1860
        path = _write_designspace(
            tmp_path,
            axes=[("wght", "Weight", 400, 400, 700)],
            sources=[{"Weight": 400}],
        )
        assert source_is_variable(path)

    def test_designspace_point_axes_only(self, tmp_path):
        path = _write_designspace(
            tmp_path,
            axes=[("wght", "Weight", 400, 400, 400)],
            sources=[{"Weight": 400}],
        )
        assert not source_is_variable(path)

    def test_designspace_mixed_point_and_range_axes(self, tmp_path):
        # Like Mingzat: wght 400-700, wdth 100-100, XXXX 0-0
        # https://github.com/googlefonts/fontc/issues/1860
        path = _write_designspace(
            tmp_path,
            axes=[
                ("wght", "Weight", 400, 400, 700),
                ("wdth", "Width", 100, 100, 100),
                ("XXXX", "Custom", 0, 0, 0),
            ],
            sources=[{"Weight": 400, "Width": 100, "Custom": 0}],
        )
        assert source_is_variable(path)

    def test_glyphs_multiple_masters(self, tmp_path):
        path = _write_glyphs(tmp_path, masters=[400, 700])
        assert source_is_variable(path)

    def test_glyphs_single_master(self, tmp_path):
        path = _write_glyphs(tmp_path, masters=[400])
        assert not source_is_variable(path)

    def test_glyphs_virtual_masters_extend_axis_range(self, tmp_path):
        # 1 master at wght=400, virtual master at wght=700
        path = _write_glyphs(tmp_path, masters=[400], virtual_masters=[700])
        assert source_is_variable(path)


@pytest.fixture
def flavor():
    """Set --flavor for the duration of a test, then put it back."""
    from absl import flags

    import ttx_diff.__main__  # noqa: F401  (defines the flag)

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    previous = flags.FLAGS.flavor

    def setter(value):
        flags.FLAGS.flavor = value

    yield setter
    flags.FLAGS.flavor = previous


class TestFlavor:
    def test_ttf_is_the_default(self, tmp_path, flavor):
        assert output_font_path(tmp_path, "fontc") == tmp_path / "fontc.ttf"
        assert output_font_path(tmp_path, "fontmake") == tmp_path / "fontmake.ttf"

    def test_otf_names(self, tmp_path, flavor):
        flavor("otf")
        assert output_font_path(tmp_path, "fontc") == tmp_path / "fontc.otf"
        assert output_font_path(tmp_path, "fontmake") == tmp_path / "fontmake.otf"

    def test_otf_rejects_variable_source(self, tmp_path):
        # fontc has no CFF2 writer, so there is nothing to compare
        path = _write_glyphs(tmp_path, masters=[400, 700])
        result = subprocess.run(
            [sys.executable, "-m", "ttx_diff", "--flavor", "otf", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--flavor otf requires a static source" in result.stderr

    def test_otf_rejects_gftools_compare(self, tmp_path):
        ufo = tmp_path / "test.ufo"
        ufo.mkdir()
        (ufo / "metainfo.plist").write_text("")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ttx_diff",
                "--flavor",
                "otf",
                "--compare",
                "gftools",
                str(ufo),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--flavor otf is only supported with --compare default" in result.stderr
