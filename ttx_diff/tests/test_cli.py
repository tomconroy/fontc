"""Tests for the ttx-diff CLI."""

import re
import subprocess
import sys

import pytest

from ttx_diff.core import (
    BuildFail,
    build_fontmake,
    output_font_path,
    resolve_instance,
    source_is_variable,
)


def _write_designspace(
    tmp_path, axes, sources, instances=None, name="test.designspace"
):
    """Write a designspace.

    Each axis is (tag, name, min, default, max) and may carry a sixth element,
    the axis map, as [(user, design), ...]. Sources are design locations;
    instances are (name, design location) pairs.
    """
    from fontTools.designspaceLib import (
        AxisDescriptor,
        DesignSpaceDocument,
        InstanceDescriptor,
        SourceDescriptor,
    )

    ds = DesignSpaceDocument()
    for axis in axes:
        tag, axis_name, mn, df, mx = axis[:5]
        a = AxisDescriptor()
        a.tag, a.name = tag, axis_name
        a.minimum, a.default, a.maximum = mn, df, mx
        a.map = list(axis[5]) if len(axis) > 5 else []
        ds.addAxis(a)
    for loc in sources:
        s = SourceDescriptor()
        s.location = loc
        ds.addSource(s)
    for instance_name, loc in instances or ():
        i = InstanceDescriptor()
        i.name = instance_name
        i.familyName = "Test"
        i.styleName = instance_name.removeprefix("Test ")
        i.filename = f"instance_ufos/{instance_name.replace(' ', '')}.ufo"
        i.designLocation = dict(loc)
        ds.addInstance(i)
    path = tmp_path / name
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


@pytest.fixture
def instance():
    """Set --instance for the duration of a test, then put it back."""
    from absl import flags

    import ttx_diff.__main__  # noqa: F401  (defines the flag)

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    previous = flags.FLAGS.instance

    def setter(value):
        flags.FLAGS.instance = value

    yield setter
    flags.FLAGS.instance = previous


def _variable_designspace(tmp_path, instances, axes=None, name="test.designspace"):
    return _write_designspace(
        tmp_path,
        axes=axes or [("wght", "Weight", 400, 400, 700)],
        sources=[{"Weight": 400}, {"Weight": 700}],
        instances=instances,
        name=name,
    )


def _record_fontmake(monkeypatch, write_output=True):
    """Stub out the fontmake subprocess, returning the list of commands run."""
    import ttx_diff.core as core

    commands = []

    def fake_build(cmd, build_dir, **kwargs):
        commands.append([str(c) for c in cmd])
        if write_output:
            out = cmd[cmd.index("--output-path") + 1]
            (build_dir / out).write_bytes(b"not really a font")

    monkeypatch.setattr(core, "build", fake_build)
    return commands


class TestInstance:
    def test_default_picks_the_instance_at_the_default_location(self, tmp_path):
        path = _variable_designspace(
            tmp_path,
            instances=[
                ("Test Bold", {"Weight": 700}),
                ("Test Regular", {"Weight": 400}),
            ],
        )
        resolved = resolve_instance(path, "@default")
        assert resolved.name == "Test Regular"
        assert resolved.index == 1
        assert resolved.is_default
        # fontc is pinned by location, fontmake by (escaped) name
        assert resolved.fontc_arg() == "wght=400"
        assert re.fullmatch(resolved.fontmake_arg(), "Test Regular")

    def test_default_tiebreaks_on_document_order(self, tmp_path):
        path = _variable_designspace(
            tmp_path,
            instances=[
                ("Test Regular", {"Weight": 400}),
                ("Test Book", {"Weight": 400}),
            ],
        )
        assert resolve_instance(path, "@default").name == "Test Regular"

    def test_no_instance_at_the_default_location_is_a_skip(self, tmp_path):
        path = _variable_designspace(
            tmp_path, instances=[("Test Bold", {"Weight": 700})]
        )
        with pytest.raises(SystemExit) as e:
            resolve_instance(path, "@default")
        assert e.value.code == "SKIP: no named instance at the default location"

    def test_source_without_instances_is_a_skip(self, tmp_path):
        path = _variable_designspace(tmp_path, instances=[])
        with pytest.raises(SystemExit) as e:
            resolve_instance(path, "@default")
        assert e.value.code == "SKIP: source has no named instances"

    def test_duplicate_instance_name_is_a_skip(self, tmp_path):
        # fontmake selects by regex and errors on >1 match, so we cannot ask it
        # for one of two instances that share a name
        path = _variable_designspace(
            tmp_path,
            instances=[("Test Regular", {"Weight": 400})] * 2,
        )
        with pytest.raises(SystemExit) as e:
            resolve_instance(path, "@default")
        assert e.value.code == "SKIP: ambiguous instance name"

    def test_explicit_name_selects_that_instance(self, tmp_path):
        path = _variable_designspace(
            tmp_path,
            instances=[
                ("Test Regular", {"Weight": 400}),
                ("Test Bold", {"Weight": 700}),
            ],
        )
        resolved = resolve_instance(path, "Test Bold")
        assert resolved.name == "Test Bold"
        assert not resolved.is_default
        assert resolved.fontc_arg() == "wght=700"

    def test_unknown_name_is_an_error_not_a_skip(self, tmp_path):
        path = _variable_designspace(
            tmp_path, instances=[("Test Regular", {"Weight": 400})]
        )
        with pytest.raises(SystemExit) as e:
            resolve_instance(path, "Test Nope")
        assert "no instance named 'Test Nope'" in str(e.value.code)

    def test_user_location_maps_back_through_a_nonlinear_axis_map(self, tmp_path):
        # design 75 is 3/4 of the design range but user 650, not user 700:
        # instance locations are design space, fontc's --instance is user space
        path = _write_designspace(
            tmp_path,
            axes=[
                (
                    "wght",
                    "Weight",
                    100,
                    400,
                    900,
                    [(100, 0), (400, 50), (900, 100)],
                )
            ],
            sources=[{"Weight": 0}, {"Weight": 50}, {"Weight": 100}],
            instances=[
                ("Test Regular", {"Weight": 50}),
                ("Test Semibold", {"Weight": 75}),
            ],
        )
        # the default source sits at design 50, i.e. user 400
        default = resolve_instance(path, "@default")
        assert default.name == "Test Regular"
        assert default.fontc_arg() == "wght=400"
        assert resolve_instance(path, "Test Semibold").fontc_arg() == "wght=650"

    def test_non_injective_axis_map_is_a_skip(self, tmp_path):
        # user 400 and user 500 both sit at design 50: fontc pins in user space
        # and cannot express the design-space pin fontmake uses
        path = _write_designspace(
            tmp_path,
            axes=[
                (
                    "wght",
                    "Weight",
                    100,
                    400,
                    900,
                    [(100, 0), (400, 50), (500, 50), (900, 100)],
                )
            ],
            sources=[{"Weight": 0}, {"Weight": 50}, {"Weight": 100}],
            instances=[("Test Regular", {"Weight": 50})],
        )
        with pytest.raises(SystemExit) as e:
            resolve_instance(path, "@default")
        assert e.value.code == "SKIP: non-injective axis map (fontc pins in user space)"

    def test_static_source_is_a_skip(self, tmp_path):
        ufo = tmp_path / "test.ufo"
        ufo.mkdir()
        (ufo / "metainfo.plist").write_text("")
        result = subprocess.run(
            [sys.executable, "-m", "ttx_diff", "--instance", "@default", str(ufo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert (
            "SKIP: static source (instance mode requires a variable source)"
            in result.stderr
        )

    def test_print_instances_lists_the_default_pick(self, tmp_path):
        path = _variable_designspace(
            tmp_path,
            instances=[
                ("Test Bold", {"Weight": 700}),
                ("Test Regular", {"Weight": 400}),
            ],
        )
        result = subprocess.run(
            [sys.executable, "-m", "ttx_diff", "--print_instances", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert "<- @default" not in lines[0]
        assert lines[1].endswith("<- @default")
        assert "user [wght=400]" in lines[1]

    def test_instanced_fontmake_build_is_static_and_keeps_overlaps(
        self, tmp_path, monkeypatch, flavor, instance
    ):
        # the bug this guards: a variable *source* with a static *output*. Get
        # this wrong and fontmake builds a variable font, or removes overlaps
        # that fontc cannot remove, and every outline differs.
        path = _variable_designspace(
            tmp_path, instances=[("Test Regular", {"Weight": 400})]
        )
        resolved = resolve_instance(path, "@default")
        commands = _record_fontmake(monkeypatch)
        build_fontmake(path, tmp_path, resolved)
        (cmd,) = commands
        assert cmd[:3] == ["fontmake", "-o", "ttf"]
        assert "--keep-overlaps" in cmd
        assert cmd[cmd.index("-i") + 1] == resolved.fontmake_arg()

    def test_instanced_otf_build_is_cff(self, tmp_path, monkeypatch, flavor, instance):
        flavor("otf")
        path = _variable_designspace(
            tmp_path, instances=[("Test Regular", {"Weight": 400})]
        )
        resolved = resolve_instance(path, "@default")
        commands = _record_fontmake(monkeypatch)
        build_fontmake(path, tmp_path, resolved)
        (cmd,) = commands
        assert cmd[:3] == ["fontmake", "-o", "otf"]
        assert "--keep-overlaps" in cmd
        assert cmd[cmd.index("--optimize-cff") + 1] == "1"

    def test_variable_build_is_unchanged(self, tmp_path, monkeypatch, flavor):
        path = _variable_designspace(
            tmp_path, instances=[("Test Regular", {"Weight": 400})]
        )
        commands = _record_fontmake(monkeypatch)
        build_fontmake(path, tmp_path)
        (cmd,) = commands
        assert cmd[:3] == ["fontmake", "-o", "variable"]
        assert "--keep-overlaps" not in cmd
        assert "-i" not in cmd

    def test_fontmake_writing_nothing_is_a_build_failure(
        self, tmp_path, monkeypatch, flavor, instance
    ):
        # `-i` that matches no instance exits 0 and writes nothing; that has to
        # be a legible failure, not an assertion further down
        path = _variable_designspace(
            tmp_path, instances=[("Test Regular", {"Weight": 400})]
        )
        resolved = resolve_instance(path, "@default")
        _record_fontmake(monkeypatch, write_output=False)
        with pytest.raises(BuildFail) as e:
            build_fontmake(path, tmp_path, resolved)
        assert "produced no output" in e.value.msg
