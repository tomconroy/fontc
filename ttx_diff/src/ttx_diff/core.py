#!/usr/bin/env python3
"""Helper for comparing fontc (Rust) vs fontmake (Python) binaries.

Turns each into ttx, eliminates expected sources of difference and prints
a brief summary of the result.

fontmake should be installed in an active virtual environment.

Usage:
    # rebuild with fontmake and fontc and compare
    python resources/scripts/ttx_diff.py ../OswaldFont/sources/Oswald.glyphs

    # rebuild the fontc copy, reuse the prior fontmake copy (if present), and compare
    # useful if you are making changes to fontc meant to narrow the diff
    python resources/scripts/ttx_diff.py --rebuild fontc ../OswaldFont/sources/Oswald.glyphs

    # compare two precompiled fonts directly (no compilation from source)
    python resources/scripts/ttx_diff.py --fontc_font path/to/fontc.ttf --fontmake_font path/to/fontmake.ttf

    # compare CFF (.otf) output instead of glyf (.ttf); static sources only
    python resources/scripts/ttx_diff.py --flavor otf ../resources/testdata/Static-Regular.ufo

    # compare a single static instance of a variable source against
    # `fontmake -i`; --instance also takes a literal instance name
    python resources/scripts/ttx_diff.py --instance @default ../OswaldFont/sources/Oswald.glyphs

JSON:
    If the `--json` flag is passed, this tool will output JSON.

    If both compilers ran successfully, this dictionary will have a single key,
    "success", which will contain a dictionary, where keys are the tags of tables
    (or another identifier) and the value is either a float representing the
    'difference ratio' (where 1.0 means identical and 0.0 means maximally
   dissimilar) or, if only one compiler produced that table, the name of that
    compiler as a string.
    For example, the output `{"success": { "GPOS": 0.99, "vmxt": "fontmake" }}`
    means that the "GPOS" table was 99% similar, and only `fontmake` produced
    the "vmtx" table (and all other tables were identical).

    If one or both of the compilers fail to exit successfully, we will return a
    dictionary with the single key, "error", where the payload is a dictionary
    where keys are the name of the compiler that failed, and the body is a
    dictionary with "command" and "stderr" fields, where the "command" field
    is the command that was used to run that compiler.
"""

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, NoReturn, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml
from absl import flags
from cdifflib import CSequenceMatcher as SequenceMatcher
from fontTools.designspaceLib import DesignSpaceDocument
from fontTools.misc.fixedTools import otRound
from fontTools.ttLib import TTFont
from fontTools.varLib.iup import iup_delta
from glyphsLib import GSFont
from lxml import etree

from ttx_diff import __version__

# environment variable used by GFTOOLS
GFTOOLS_FONTC_PATH = "GFTOOLS_FONTC_PATH"


FLAGS = flags.FLAGS
# used instead of a tag for the normalized mark/kern output
MARK_KERN_NAME = "(mark/kern)"
LIG_CARET_NAME = "ligcaret"
# outline flavors we can compare; the value doubles as the file extension
# and as fontmake's `-o` build type for a static source
FLAVOR_TTF = "ttf"
FLAVOR_OTF = "otf"
# maximum chars of stderr to include when reporting errors; prevents
# too much bloat when run in CI
MAX_ERR_LEN = 1000

# --instance policies are namespaced with a leading '@' so that a source with an
# instance literally named 'default' is still reachable by name
INSTANCE_DEFAULT = "@default"

# Reasons we decline to compare a target. These are a contract with
# fontc_crater (see `skip`): keep them short, stable, and free of paths, so a
# report can group targets by reason.
SKIP_OTF_VARIABLE = "variable source (fontc cannot write CFF2)"
SKIP_INSTANCE_STATIC = "static source (instance mode requires a variable source)"
SKIP_NO_INSTANCES = "source has no named instances"
SKIP_NO_DEFAULT_INSTANCE = "no named instance at the default location"
SKIP_AMBIGUOUS_INSTANCE = "ambiguous instance name"
SKIP_NON_INJECTIVE_MAP = "non-injective axis map (fontc pins in user space)"

# fontc and fontmake's builds may be off by a second or two in the
# head.created/modified; setting this makes them the same
if "SOURCE_DATE_EPOCH" not in os.environ:
    os.environ["SOURCE_DATE_EPOCH"] = str(int(time.time()))


# print to stderr
def eprint(*objects):
    print(*objects, file=sys.stderr)


def skip(reason: str, detail: Optional[str] = None) -> NoReturn:
    """Exit saying this target is not applicable to this run.

    Not a compiler failure: `--flavor otf` on a variable source, `--instance` on
    a static source and "this source has no instance at the default location"
    all mean "there is nothing here to compare", and should read that way in a
    report. The 'SKIP: ' prefix is the contract with fontc_crater, which turns
    the reason into 'skipped: <reason>' (see ttx_diff_runner.rs); anything
    target-specific belongs in `detail`, which is printed but not matched on.
    """
    if detail is not None:
        eprint(detail)
    sys.exit(f"SKIP: {reason}")


_timing_log: List[Tuple[str, float, int]] = []
_timing_depth: int = 0


@contextmanager
def timed(label: str):
    global _timing_depth
    depth = _timing_depth
    _timing_depth += 1
    idx = len(_timing_log)
    _timing_log.append((label, 0.0, depth))
    start = time.time()
    yield
    _timing_depth -= 1
    _timing_log[idx] = (label, time.time() - start, depth)


def to_xml_string(e) -> str:
    xml = etree.tostring(e)
    # some table diffs were mismatched because of inconsistency in ending newline
    xml = xml.strip()
    return xml


@cache
def home_str() -> str:
    return str(Path("~").expanduser())


def rel_user(fragment: Any) -> str:
    """If an absolute path is in the home directory convert to ~ form for display.

    Makes cli output closer to what you'd actually type, e.g. ~/something
    Instead of /usr/blah/blah/blah/something"""
    fragment = str(fragment)
    home = home_str()
    if fragment.startswith(home):
        fragment = "~" + fragment[len(home) :]
    return fragment


# execute a command after logging it to stderr.
# All additional kwargs are passed to subprocess.run
def log_and_run(cmd: Sequence, cwd=None, **kwargs):
    # Convert to ~ format because it's annoying to see really long usr paths
    log_cmd = " ".join(rel_user(c) for c in cmd)
    if cwd is not None:
        eprint(f"  (cd {rel_user(cwd)} && {log_cmd})")
    else:
        eprint(f"  ({log_cmd})")
    return subprocess.run(
        cmd,
        text=True,
        cwd=cwd,
        # combine stderr and stdout for the purposes of capturing diagnostics
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )


def run_ttx(font_file: Path):
    ttx_file = font_file.with_suffix(".ttx")
    # if this exists we're allowed to reuse it
    if ttx_file.is_file():
        eprint(f"reusing {rel_user(ttx_file)}")
        return ttx_file

    cmd = [
        "ttx",
        "-o",
        ttx_file.name,
        font_file.name,
    ]
    log_and_run(cmd, font_file.parent, check=True)
    return ttx_file


# generate a simple text repr for gpos for this font, with retry
def run_normalizer(normalizer_bin: Path, font_file: Path, table: str):
    if table == "gpos":
        out_path = font_file.with_suffix(".markkern.txt")
    elif table == "gdef":
        out_path = font_file.with_suffix(f".{LIG_CARET_NAME}.txt")
    else:
        raise ValueError(f"unknown table for normalizer: '{table}'")

    if out_path.exists():
        eprint(f"reusing {rel_user(out_path)}")
    NUM_RETRIES = 5
    for i in range(NUM_RETRIES + 1):
        try:
            return try_normalizer(normalizer_bin, font_file, out_path, table)
        except subprocess.CalledProcessError as e:
            time.sleep(0.1)
            if i >= NUM_RETRIES:
                raise e
            eprint(f"normalizer failed with code '{e.returncode}'', retrying")


# we had a bug where this would sometimes hang in mysterious ways, so we may
# call it multiple times if it fails
def try_normalizer(normalizer_bin: Path, font_file: Path, out_path: Path, table: str):
    NORMALIZER_TIMEOUT = 60 * 10  # ten minutes
    if not out_path.is_file():
        cmd = [
            normalizer_bin.absolute(),
            font_file.name,
            "-o",
            out_path.name,
            "--table",
            table,
        ]
        log_and_run(cmd, font_file.parent, check=True, timeout=NORMALIZER_TIMEOUT)
        # if we finished running and there's no file then there's no output:
        if not out_path.is_file():
            return ""
    with open(out_path) as f:
        return f.read()


class BuildFail(Exception):
    """An exception raised if a compiler fails."""

    def __init__(self, cmd: Sequence, msg: str):
        self.command = list(str(c) for c in cmd)
        self.msg = msg


# run a font compiler
def build(cmd: Sequence, build_dir: Optional[Path], **kwargs):
    output = log_and_run(cmd, build_dir, **kwargs)
    if output.returncode != 0:
        raise BuildFail(cmd, output.stderr or output.stdout)


def font_suffix() -> str:
    """The file extension for the outline flavor we're comparing, e.g. '.otf'."""
    return "." + FLAGS.flavor


def output_font_path(build_dir: Path, compiler: str) -> Path:
    """Where a given compiler's binary lands, e.g. build_dir/'fontc.otf'."""
    return build_dir / (compiler + font_suffix())


def build_fontc(
    source: Path,
    fontc_bin: Path,
    build_dir: Path,
    instance: Optional["ResolvedInstance"] = None,
):
    out_file = output_font_path(build_dir, "fontc")
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    cmd = [
        fontc_bin,
        "--build-dir",
        ".",
        "-o",
        out_file.name,
        source,
        "--emit-debug",
    ]
    if FLAGS.flavor != FLAVOR_TTF:
        cmd += ["--flavor", FLAGS.flavor]
    if FLAGS.keep_direction:
        cmd.append("--keep-direction")
    if not FLAGS.production_names:
        cmd.append("--no-production-names")
    if instance is not None:
        # a location, not a name: fontc's named instances are keyed by style
        # name ("Bold") where fontmake's -i wants the DesignSpace instance name
        # ("Family Bold"), so only the location means the same thing to both
        cmd += ["--instance", instance.fontc_arg()]
    build(cmd, build_dir)


def build_fontmake(
    source: Path, build_dir: Path, instance: Optional["ResolvedInstance"] = None
):
    out_file = output_font_path(build_dir, "fontmake")
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    # what matters is whether the *output* is variable, not the source: in
    # instance mode a variable source produces a static font, which must be
    # built as a static (`-o ttf`/`-o otf`) and, crucially, must keep its
    # overlaps -- fontmake removes them for static builds and fontc cannot, so
    # keying this off the source would make every outline differ
    variable_output = instance is None and source_is_variable(source)
    if FLAGS.flavor == FLAVOR_OTF:
        # guarded in main(), but this is the only place that knows the
        # buildtype so be explicit rather than silently building something else
        assert not variable_output, "otf flavor requires a static output"
        buildtype = "otf"
    elif variable_output:
        buildtype = "variable"
    else:
        buildtype = "ttf"
    # exactly one -o per invocation: fontmake mutates the interpolated UFOs in
    # place, so asking one run for both ttf and otf corrupts the second output
    cmd = [
        "fontmake",
        "-o",
        buildtype,
        "--output-path",
        out_file.name,
        "--drop-implied-oncurves",
        # helpful for troubleshooting
        "--debug-feature-file",
        "debug.fea",
    ]
    if FLAGS.flavor == FLAVOR_OTF:
        # 1 = specialize the charstring operators but do not subroutinize.
        # fontc's CFF writer specializes too; subroutinization (the default, 2)
        # rewrites the whole table via cffsubr/compreffor purely to save space,
        # so leaving it on would make every charstring differ for reasons that
        # have nothing to do with correctness.
        cmd += ["--optimize-cff", "1"]
    if FLAGS.keep_direction:
        cmd.append("--keep-direction")
    if not FLAGS.production_names:
        cmd.append("--no-production-names")
    if FLAGS.keep_overlaps and not variable_output:
        cmd.append("--keep-overlaps")
    if instance is not None:
        cmd += ["-i", instance.fontmake_arg()]
        # a 'TTFAutohint options' custom parameter makes fontmake run
        # ttfautohint on static builds; fontc never will (a post-process,
        # out of scope like overlap removal), so compare unhinted statics.
        # Scoped to instance mode to leave static-source runs untouched.
        cmd.append("--no-autohint")
    cmd.append(str(source))

    build(cmd, build_dir)

    # fontmake exits 0 and writes nothing when `-i` matches no instance, so a
    # missing output here is a real (and otherwise silent) failure
    if not out_file.exists():
        detail = (
            f" (does '-i {instance.fontmake_arg()}' match an instance?)"
            if instance is not None
            else ""
        )
        raise BuildFail(cmd, f"fontmake exited 0 but produced no output{detail}")


@contextmanager
def modified_gftools_config(
    maybe_path_to_move_config: Optional[Path],
    cmdline: List[str],
    extra_args: Sequence[str],
) -> Generator[None, None, None]:
    """Modify the gftools config file to add extra arguments.

    A temporary config file is created with the extra args added to the
    `extraFontmakeArgs` key, and replaces the original config file in the
    command line arguments' list, which is modified in-place.
    The temporary file is deleted after the context manager exits.
    If the extra_args list is empty, the context manager does nothing.

    Args:
        cmdline: The command line arguments passed to gftools. This must include
            the path to a config.yaml file.
        extra_args: Extra arguments to add to the config file. Can be empty.
    """
    if extra_args:
        try:
            config_idx, config_path = next(
                (
                    (i, arg)
                    for i, arg in enumerate(cmdline)
                    if arg.endswith((".yaml", ".yml"))
                )
            )
        except StopIteration:
            raise ValueError(
                "No config file found in command line arguments. "
                "Please provide a config.yaml file."
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["extraFontmakeArgs"] = " ".join(
            config.get("extraFontmakeArgs", "").split(" ") + extra_args
        )

        config_dir = maybe_path_to_move_config or Path(config_path).parent

        with NamedTemporaryFile(
            mode="w",
            prefix="config_",
            suffix=".yaml",
            delete=False,
            dir=config_dir,
        ) as f:
            yaml.dump(config, f)
        temp_path = Path(f.name)

        cmdline[config_idx] = temp_path

    # if the build later fails for any reason, we still want to delete the temp file
    try:
        yield
    finally:
        if extra_args:
            temp_path.unlink()


def run_gftools(
    source: Path, config: str, build_dir: Path, fontc_bin: Optional[Path] = None
):
    config_path = Path(config)
    tool = "fontmake" if fontc_bin is None else "fontc"
    out_file = output_font_path(build_dir, tool)
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    out_dir = build_dir / "gftools_temp_dir"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    source_for_gftools = source_for_gftools_single_source(source, config_path)
    maybe_new_config_path = path_to_move_external_config(source, config_path)

    cmd = [
        "gftools",
        "builder",
        config,
        "--experimental-simple-output",
        out_dir,
        "--experimental-single-source",
        source_for_gftools,
    ]
    if fontc_bin is not None:
        cmd += ["--experimental-fontc", fontc_bin]

    extra_args = []
    if FLAGS.keep_overlaps:
        # (we only want this for the statics but it's a noop for variables)
        extra_args.append("--keep-overlaps")
    if not FLAGS.production_names:
        extra_args.append("--no-production-names")

    with modified_gftools_config(maybe_new_config_path, cmd, extra_args):
        build(cmd, None)

    # return a concise error if gftools produces != one output
    contents = list(out_dir.iterdir()) if out_dir.exists() else list()
    if not contents:
        raise BuildFail(cmd, "gftools produced no output")
    elif len(contents) != 1:
        contents = [p.name for p in contents]
        raise BuildFail(cmd, f"gftools produced multiple outputs: {contents}")
    copy(contents[0], out_file)

    if out_dir.exists():
        shutil.rmtree(out_dir)


def source_for_gftools_single_source(source: Path, config_path: Path) -> str:
    """Compute the value for gftools --experimental-single-source.

    Returns the source path relative to the effective config directory:
    the source repo root for external configs (where the config gets moved),
    or the config file's parent directory for same-repo configs.
    """
    maybe_new_config_path = path_to_move_external_config(source, config_path)
    effective_config_dir = maybe_new_config_path or config_path.resolve().parent
    return os.path.relpath(source, effective_config_dir)


def path_to_move_external_config(source: Path, config: Path) -> Optional[Path]:
    source_repo = find_repo_root(source)
    config_repo = find_repo_root(config)

    if source_repo != config_repo:
        return source_repo


def find_repo_root(path: Path) -> Optional[Path]:
    # if this is an actual repo:
    if path.is_dir() and path.joinpath(".git").exists():
        return path
    # if this is from a tarball:
    if path.is_dir() and looks_like_it_has_appended_sha(path.name):
        return path
    elif path.parent == path:
        return None
    else:
        return find_repo_root(path.parent)


def looks_like_it_has_appended_sha(name: str) -> bool:
    split = name.split("_")
    if len(split) < 2:
        return False
    try:
        int(split[-1], 16)
        return True
    except ValueError:
        return False


def source_is_variable(path: Path) -> bool:
    if path.suffix == ".ufo":
        return False
    if path.suffix == ".designspace":
        dspace = DesignSpaceDocument.fromfile(path)
        return any(
            a.minimum != a.default or a.maximum != a.default for a in dspace.axes
        )
    if path.suffix == ".glyphs" or path.suffix == ".glyphspackage":
        font = GSFont(path)
        # Virtual masters can extend axis min/max beyond what real masters define
        # https://github.com/googlefonts/glyphsLib/blob/75c07d42/Lib/glyphsLib/builder/axes.py#L168-L173
        virtual_masters = [
            cp.value
            for cp in font.customParameters
            if cp.name == "Virtual Master" and not getattr(cp, "disabled", False)
        ]
        for i, axis in enumerate(font.axes):
            values = [m.axes[i] for m in font.masters]
            for vm in virtual_masters:
                for entry in vm:
                    if entry.get("Axis") == axis.name:
                        values.append(entry["Location"])
            if min(values) != max(values):
                return True
        return False
    # fallback to variable, the existing default, but we should never get here?
    return True


def _fmt_coord(value: float) -> str:
    """Format an axis coordinate the way a human would type it: 400, not 400.0."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


@dataclasses.dataclass(frozen=True)
class ResolvedInstance:
    """One named instance, in the two vocabularies the compilers speak.

    fontmake selects by DesignSpace instance *name*; fontc selects by user-space
    *location*. They are not interchangeable: for a .glyphs source fontmake's
    name is "Family Style" while fontc's named-instance name is the style name
    alone, so we resolve once here and hand each compiler its own dialect.
    """

    # the DesignSpace instance 'name' attribute -> fontmake -i
    name: str
    # (axis tag, user-space value) per axis, in axis order -> fontc --instance
    user_location: Tuple[Tuple[str, float], ...]
    # (axis name, design-space value) per axis; @default matches on this
    design_location: Tuple[Tuple[str, float], ...]
    # position in the designspace document, which is the tiebreak
    index: int
    # does this instance sit at the default source's location?
    is_default: bool

    def fontmake_arg(self) -> str:
        """The `-i` pattern that selects exactly this instance.

        fontmake matches with `re.fullmatch` (FontProject.interpolate_instance_ufos),
        not string equality, so a name containing regex metacharacters -- or a
        space, which re.escape also escapes -- has to be escaped.
        """
        return re.escape(self.name)

    def fontc_arg(self) -> str:
        """The `--instance` location, e.g. 'wght=400,wdth=87.5'."""
        return ",".join(f"{tag}={_fmt_coord(v)}" for tag, v in self.user_location)

    def describe(self) -> str:
        design = " ".join(f"{n}={_fmt_coord(v)}" for n, v in self.design_location)
        return (
            f"{self.index}: {self.name!r} design [{design}] user [{self.fontc_arg()}]"
        )


@cache
def _designspace_for(path: Path) -> DesignSpaceDocument:
    """The DesignSpace that fontmake will interpolate instances from.

    For .glyphs sources this runs the same conversion fontmake runs, so the
    instance names are exactly the ones `-i` will match against: glyphsLib
    synthesizes them as "familyName styleName" and drops inactive,
    non-family-included and VARIABLE-type instances along the way.
    """
    if path.suffix == ".designspace":
        return DesignSpaceDocument.fromfile(path)
    if path.suffix in (".glyphs", ".glyphspackage"):
        # imported lazily: only instance mode needs them
        import ufoLib2
        from glyphsLib import to_designspace

        return to_designspace(
            GSFont(path),
            ufo_module=ufoLib2,
            minimal=True,
            store_editor_state=False,
        )
    skip(f"cannot list the instances of a '{path.suffix}' source")


def _axis_map_is_non_injective(axis) -> bool:
    """True if two user values share a design value on this axis.

    fontmake pins an instance at its design location; fontc is given user
    coordinates and converts back. A flat segment in the map makes that round
    trip lossy (the two toolchains disagree about which user value a design
    value came from), so such a source has no meaningful comparison.
    """
    outputs = [design for _user, design in getattr(axis, "map", None) or []]
    return len(outputs) != len(set(outputs))


def instances_of(source: Path) -> List[ResolvedInstance]:
    """Every named instance of a source, in document order."""
    doc = _designspace_for(source)
    tag_for_axis = {axis.name: axis.tag for axis in doc.axes}
    # the default location is design space on both sides of this comparison:
    # a source with an axis <map> has an axis default (user) that is not its
    # default source's coordinate (design), and comparing the two would be
    # wrong for most .glyphs sources
    default_source = doc.findDefault()
    default_location = (
        default_source.getFullDesignLocation(doc)
        if default_source is not None
        else dict(doc.newDefaultLocation())
    )
    resolved = []
    for index, instance in enumerate(doc.instances):
        # sparse instance locations are completed from the default
        design = instance.getFullDesignLocation(doc)
        user = instance.getFullUserLocation(doc)
        resolved.append(
            ResolvedInstance(
                name=instance.name,
                user_location=tuple(
                    (tag_for_axis[name], float(value)) for name, value in user.items()
                ),
                design_location=tuple(
                    (name, float(value)) for name, value in design.items()
                ),
                index=index,
                is_default=design == default_location,
            )
        )
    return resolved


def resolve_instance(source: Path, spec: str) -> ResolvedInstance:
    """Pick the one instance to build, from a policy or a name.

    Skips (rather than fails) when the source cannot answer the question, so
    that a corpus sweep reports "not applicable" instead of "broken".
    """
    instances = instances_of(source)
    if not instances:
        skip(SKIP_NO_INSTANCES, f"'{rel_user(source)}' declares no instances")
    non_injective = [
        axis.name
        for axis in _designspace_for(source).axes
        if _axis_map_is_non_injective(axis)
    ]
    if non_injective:
        skip(
            SKIP_NON_INJECTIVE_MAP,
            f"'{rel_user(source)}' has a flat segment on axes {non_injective}",
        )

    if spec == INSTANCE_DEFAULT:
        matches = [instance for instance in instances if instance.is_default]
        if not matches:
            skip(
                SKIP_NO_DEFAULT_INSTANCE,
                f"'{rel_user(source)}' instances:\n  "
                + "\n  ".join(i.describe() for i in instances),
            )
    elif spec.startswith("@"):
        sys.exit(
            f"unknown --instance policy '{spec}'; expected '{INSTANCE_DEFAULT}' or "
            "an instance name"
        )
    else:
        matches = [instance for instance in instances if instance.name == spec]
        if not matches:
            sys.exit(
                f"no instance named '{spec}' in '{rel_user(source)}'; instances:\n  "
                + "\n  ".join(i.describe() for i in instances)
            )
    # instances_of returns document order, so the first match is the lowest index
    chosen = matches[0]
    # fontmake selects with re.fullmatch and errors out ("output_path requires a
    # single input") if the pattern matches more than one instance; an escaped
    # name matches only itself, so this fires exactly when names are duplicated
    pattern = chosen.fontmake_arg()
    twins = [i for i in instances if re.fullmatch(pattern, i.name)]
    if len(twins) > 1:
        skip(
            SKIP_AMBIGUOUS_INSTANCE,
            f"'{rel_user(source)}' has {len(twins)} instances named "
            f"{chosen.name!r}; fontmake cannot be pointed at one of them",
        )
    return chosen


def print_instances(source: Path):
    """--print_instances: show what --instance can be given, and what @default picks."""
    for instance in instances_of(source):
        marker = "  <- @default" if instance.is_default else ""
        print(f"{instance.describe()}{marker}")


def copy(old, new):
    shutil.copyfile(old, new)
    return new


def get_name_to_id_map(ttx: etree.ElementTree):
    return {
        el.attrib["nameID"]: el.text
        for el in ttx.xpath(
            "name/namerecord[@platformID='3' and @platEncID='1' and @langID='0x409']"
        )
    }


def name_id_to_name(ttx, xpath, attr):
    id_to_name = get_name_to_id_map(ttx)
    for el in ttx.xpath(xpath):
        if attr is None:
            if el.text is None:
                continue
            name_id = el.text
            # names <= 255 have specific assigned slots, names > 255 not
            if int(name_id) <= 255:
                continue
            name = id_to_name.get(name_id, f"NonExistingNameID {name_id}").strip()
            el.text = name
        else:
            if attr not in el.attrib:
                continue
            # names <= 255 have specific assigned slots, names > 255 not
            name_id = el.attrib[attr]
            if int(name_id) <= 255:
                continue
            name = id_to_name.get(name_id, f"NonExistingNameID {name_id}").strip()
            el.attrib[attr] = name


def normalize_name_ids(ttx: etree.ElementTree):
    name = ttx.find("name")
    if name is None:
        return

    records = name.xpath(".//namerecord")
    for record in records:
        name.remove(record)
        # User-defined name IDs get replaced by there value where they are
        # used, so the ID itself is not interesting and we replace them with a
        # fixed value
        if int(record.attrib["nameID"]) > 255:
            record.attrib["nameID"] = "USER_ID"

    records = sorted(
        records,
        key=lambda x: (
            x.attrib["platformID"],
            x.attrib["platEncID"],
            x.attrib["langID"],
            x.attrib["nameID"],
            x.text,
        ),
    )

    for record in records:
        name.append(record)

    # items keep their indentation when we reorder them, so reindent everything
    etree.indent(name)


def find_table(ttx, tag):
    return select_one(ttx, f"/ttFont/{tag}")


def select_one(container, xpath):
    el = container.xpath(xpath)
    if len(el) != 1:
        raise IndexError(f"Wanted 1 name element, got {len(el)}")
    return el[0]


def drop_weird_names(ttx):
    drops = list(
        ttx.xpath(
            "//name/namerecord[@platformID='1' and @platEncID='0' and @langID='0x0']"
        )
    )
    for drop in drops:
        drop.getparent().remove(drop)


def strip_fontc_version_tag(ttx):
    # fontc appends ";fontc <version>" to the version string (nameID 5); fontmake
    # does not, so strip it before comparing. Mirror the Rust predicate in
    # name.rs: only ";fontc " followed by a digit (the SemVer) is the stamp --
    # a human note like ";fontc is great" is left intact, and matched only
    # through the end of that segment so any other content/whitespace survives.
    for record in ttx.xpath("//name/namerecord[@nameID='5']"):
        if record.text:
            record.text = re.sub(r";fontc (?=\d)[^;\s]*", "", record.text)


def erase_checksum(ttx):
    el = select_one(ttx, "//head/checkSumAdjustment")
    del el.attrib["value"]


def stat_like_fontmake(ttx):
    try:
        el = find_table(ttx, "STAT")
    except IndexError:
        return
    ver = select_one(el, "Version")
    if ver.attrib["value"] != "0x00010002":
        # nop
        return

    # fontc likes to write STAT 1.2, fontmake prefers 1.1
    # Version 1.2 adds support for the format 4 axis value table
    # So until such time as we start writing format 4 axis value tables it doesn't matter
    ver.attrib["value"] = "0x00010001"


def normalize_all_offcurve_starting_point(contour):
    """Rotate an all-offcurve contour to start at point closest to origin.

    When multiple points are equidistant from the origin, the first one encountered
    (lowest index) is selected as the starting point.

    This is to address differences in starting points of all-offcurve TrueType
    quadratic contours between fontc and fontmake.

    fontmake preserves the original off-curve starting point from the UFO source,
    while fontc creates a synthetic on-curve point at the midpoint between first
    and last off-curve points due to kurbo::BezPath needing some on-curve point
    to move_to. When contours are reversed (default behavior without --keep-direction
    flag), this results in different starting points in the final TrueType output.

    See: <https://github.com/googlefonts/fontc/issues/1653>
    """
    pts = contour.xpath("./pt")
    if not pts or not all(pt.get("on") == "0" for pt in pts):
        return  # Not an all-offcurve contour

    # Find point closest to origin
    min_idx = min(
        range(len(pts)),
        key=lambda i: int(pts[i].get("x", 0)) ** 2 + int(pts[i].get("y", 0)) ** 2,
    )

    # Rotate if needed
    if min_idx != 0:
        contour[:] = pts[min_idx:] + pts[:min_idx]
        # Fix indentation after rotation
        etree.indent(contour, level=3)


# https://github.com/googlefonts/fontc/issues/1107
def normalize_glyf_contours(
    fontc_ttx: etree.ElementTree, fontmake_ttx: etree.ElementTree
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Reorders contours when they are identical between fontc and fontmake.

    If contours differ (e.g., different starting points), leaves
    them in their original order to avoid misleading diffs.

    For all-offcurve contours, normalizes the starting point to be the point
    closest to the origin before comparison.

    Returns a tuple of two dicts, one for fontc and one for fontmake, containing
    the new order of prior point indices for each glyph, later used for sorting
    gvar contours.
    """
    fontc_point_orders: dict[str, list[int]] = {}
    fontmake_point_orders: dict[str, list[int]] = {}

    # Get glyphs from both TTX trees
    fontc_glyphs = {g.attrib["name"]: g for g in fontc_ttx.xpath("//glyf/TTGlyph")}
    fontmake_glyphs = {
        g.attrib["name"]: g for g in fontmake_ttx.xpath("//glyf/TTGlyph")
    }

    # Only process glyphs that exist in both outputs
    for glyph_name in fontc_glyphs.keys() & fontmake_glyphs.keys():
        fontc_glyph = fontc_glyphs[glyph_name]
        fontmake_glyph = fontmake_glyphs[glyph_name]

        fontc_contours = fontc_glyph.xpath("./contour")
        fontmake_contours = fontmake_glyph.xpath("./contour")

        # Skip glyphs with mismatched contour counts
        if len(fontc_contours) != len(fontmake_contours):
            continue

        # Normalize all-offcurve contours to start at point closest to origin
        for contour in fontc_contours + fontmake_contours:
            normalize_all_offcurve_starting_point(contour)

        # Compare contours as sets to see if they're identical (ignoring order)
        fontc_strings = {to_xml_string(c) for c in fontc_contours}
        fontmake_strings = {to_xml_string(c) for c in fontmake_contours}

        if fontc_strings == fontmake_strings:
            # Contours are identical, just in different order - normalize both
            _normalize_single_glyph(fontc_glyph, fontc_contours, fontc_point_orders)
            _normalize_single_glyph(
                fontmake_glyph, fontmake_contours, fontmake_point_orders
            )
        # If sets don't match, skip normalization - leave original order

    return fontc_point_orders, fontmake_point_orders


def _normalize_single_glyph(
    glyph: etree.Element,
    contours: list[etree.Element],
    point_orders: dict[str, list[int]],
):
    """Helper function to normalize contour order within a single glyph.

    Contours are sorted alphabetically by xml string representation.

    The `point_orders` dict is used to store the new order of prior point
    indices for this glyph.
    """
    # annotate each contour with the range of point indices it covers
    with_range: list[tuple[range, etree.Element]] = []
    points_seen = 0
    for contour in contours:
        points_here = len(contour.xpath("./pt"))
        with_range.append((range(points_seen, points_seen + points_here), contour))
        points_seen += points_here
    annotated = sorted(with_range, key=lambda a: to_xml_string(a[1]))

    # sort by string representation, and skip if nothing has changed
    normalized = [contour for _, contour in annotated]
    if normalized == contours:
        return

    # normalized contours should be inserted before any other TTGlyph's
    # subelements (e.g. instructions)
    for contour in contours:
        glyph.remove(contour)
    non_contours = list(glyph)
    for el in non_contours:
        glyph.remove(el)
    glyph.extend(normalized + non_contours)

    # store new indices order
    name = glyph.attrib["name"]
    point_orders[name] = [idx for indices, _ in annotated for idx in indices]


def normalize_gvar_contours(ttx: etree.ElementTree, point_orders: dict[str, list[int]]):
    """Reorder gvar points to match normalised glyf order."""

    for glyph in ttx.xpath("//gvar/glyphVariations"):
        name = glyph.attrib["glyph"]
        order = point_orders.get(name)

        # skip glyph if glyf normalisation did not change its point order
        if order is None:
            continue

        # apply the same order to every tuple
        for tup in glyph.xpath("./tuple"):
            deltas = tup.xpath("./delta")
            assert len(order) + 4 == len(deltas), "gvar is not dense"

            # reorder and change index to match new position
            reordered = []
            for new_idx, old_idx in enumerate(order):
                delta = deltas[old_idx]  # always present as gvars are densified
                delta.attrib["pt"] = str(new_idx)
                reordered.append(delta)
            reordered += deltas[-4:]  # phantom points

            # normalized points should be inserted after any other tuple
            # subelements (e.g. coordinates)
            for delta in deltas:
                tup.remove(delta)
            non_deltas = list(tup)
            for el in non_deltas:
                tup.remove(el)
            tup.extend(non_deltas + reordered)


def normalize_null_tags(ttx: etree.ElementTree, xpath: str, attr):
    """replace the tag 'NONE' with the tag '    '"""
    for el in ttx.xpath(xpath):
        if attr:
            if el.attrib.get(attr, "") == "NONE":
                el.attrib[attr] = "    "
        else:
            if el.text == "NONE":
                el.text = "    "


# https://github.com/googlefonts/fontc/issues/1173
def erase_type_from_stranded_points(ttx):
    for contour in ttx.xpath("//glyf/TTGlyph/contour"):
        points = contour.xpath("./pt")
        if len(points) == 1:
            points[0].attrib["on"] = "irrelevent"


# only fontc emits name 25 currently, don't treat that as an error
def allow_fontc_only_variations_postscript_prefix(fontc, fontmake):
    xpath_to_name_25 = "/ttFont/name/namerecord[@nameID='25']"
    fontc_name25 = fontc.xpath(xpath_to_name_25)
    fontmake_name25 = fontmake.xpath(xpath_to_name_25)
    if fontc_name25 and not fontmake_name25:
        for n in fontc_name25:
            n.getparent().remove(n)


def allow_some_off_by_ones(fontc, fontmake, container, name_attr, coord_holder):
    fontmake_containers = fontmake.xpath(f"//{container}")
    coord_tag = coord_holder.rpartition("/")[-1]
    fontmake_num_coords = sum(
        sum(1 for _ in c.iter(coord_tag)) for c in fontmake_containers
    )
    off_by_one_budget = int(FLAGS.off_by_one_budget / 100.0 * fontmake_num_coords)
    spent = 0
    if off_by_one_budget == 0:
        return

    # put all the containers into a dict to make querying more efficient:
    fontc_items = {x.attrib[name_attr]: x for x in fontc.xpath(f"//{container}")}
    for fontmake_container in fontmake_containers:
        name = fontmake_container.attrib[name_attr]
        fontc_container = fontc_items.get(name)
        if fontc_container is None:
            continue

        fontc_els = [el for el in fontc_container.iter() if el.tag == coord_tag]
        fontmake_els = [el for el in fontmake_container.iter() if el.tag == coord_tag]

        if len(fontc_els) != len(fontmake_els):
            eprint(
                f"length of {container} '{name}' differs ({len(fontc_els)}/{len(fontmake_els)}), skipping"
            )
            continue

        for fontmake_el, fontc_el in zip(fontc_els, fontmake_els):
            for attr in ("x", "y"):
                delta_x = abs(
                    float(fontmake_el.attrib[attr]) - float(fontc_el.attrib[attr])
                )
                if 0.0 < delta_x <= 1.0:
                    fontc_el.attrib["diff_adjusted"] = "1"
                    fontmake_el.attrib["diff_adjusted"] = "1"
                    fontc_el.attrib[attr] = fontmake_el.attrib[attr]
                    spent += 1
                if spent >= off_by_one_budget:
                    eprint(
                        f"WARN: ran out of budget ({off_by_one_budget}) to fix off-by-ones in {container}"
                    )
                    return

    if spent > 0:
        eprint(
            f"INFO fixed {spent} off-by-ones in {container} (budget {off_by_one_budget})"
        )


# In various cases we have a list of indices where the order doesn't matter;
# often fontc sorts these and fontmake doesn't, so this lets us sort them there.
def sort_indices(ttx, table_tag: str, container_xpath: str, el_name: str):
    table = ttx.find(table_tag)
    if table is None:
        return
    containers = table.xpath(f"{container_xpath}")
    for container in containers:
        indices = [el for el in container.iter() if el.tag == el_name]
        values = sorted(int(v.attrib["value"]) for v in indices)
        for i, index in enumerate(indices):
            index.attrib["value"] = str(values[i])


# the same sets can be assigned different ids, but normalizer
# will resolve them to the actual glyphs and here we can just sort
def sort_gdef_mark_filter_sets(ttx: etree.ElementTree):
    markglyphs = ttx.xpath("//GDEF//MarkGlyphSetsDef")
    if markglyphs is None or len(markglyphs) == 0:
        return

    assert len(markglyphs) == 1
    markglyphs = markglyphs[0]

    coverages = sorted(
        markglyphs.findall("Coverage"), key=lambda cov: [g.attrib["value"] for g in cov]
    )
    for c in coverages:
        markglyphs.remove(c)

    id_map = {c.attrib["index"]: str(i) for (i, c) in enumerate(coverages)}
    for i, c in enumerate(coverages):
        c.attrib["index"] = str(i)
        markglyphs.append(c)

    # remap any MarkFilteringSet nodes that might not get normalized (in contextual
    # pos lookups e.g., or in GSUB)

    for mark_set in ttx.xpath("//MarkFilteringSet"):
        mark_set.attrib["value"] = id_map.get(mark_set.attrib["value"])

    # items keep their indentation when we reorder them, so reindent everything
    etree.indent(markglyphs, level=3)


LOOKUPS_TO_SKIP = set([2, 3, 4, 5, 6])  # pairpos, cursive, markbase, marklig, markmark


def remove_mark_and_kern_and_curs_lookups(ttx):
    gpos = ttx.find("GPOS")
    if gpos is None:
        return
    # './/Lookup' xpath selects all the 'Lookup' elements that are descendants of
    # the current 'GPOS' node - no matter where they are under it.
    # Most importantly, this _excludes_ GSUB lookups, which shouldn't be pruned.
    for lookup in gpos.xpath(".//Lookup"):
        lookup_type_el = lookup.find("LookupType")
        lookup_type = int(lookup_type_el.attrib["value"])
        is_extension = lookup_type == 9
        if is_extension:
            # For extension lookups, take the lookup type of the wrapped subtables;
            # all extension subtables share the same lookup type so checking only
            # the first is enough.
            ext_subtable = lookup.find("ExtensionPos")
            lookup_type = int(ext_subtable.find("ExtensionLookupType").attrib["value"])
        if lookup_type not in LOOKUPS_TO_SKIP:
            continue
        # remove all the elements but the type:
        to_remove = [child for child in lookup if child.tag != "LookupType"]
        for child in to_remove:
            lookup.remove(child)
        if is_extension:
            lookup_type_el.attrib["value"] = str(lookup_type)


# this all gets handled by otl-normalizer
def remove_gdef_lig_caret_and_var_store(ttx: etree.ElementTree):
    gdef = ttx.find("GDEF")
    if gdef is None:
        return

    for ident in ["LigCaretList", "VarStore"]:
        subtable = gdef.find(ident)
        if subtable is not None:
            gdef.remove(subtable)


# reassign class ids within a ClassDef, matching the fontc behaviour.
# returns a map of new -> old ids, which can be used to reorder elements that
# used the class ids as indices
def remap_class_def_ids_like_fontc(
    class_def: etree.ElementTree, glyph_map: Dict[str, int]
) -> Dict[int, int]:
    current_classes = defaultdict(list)
    for glyph in class_def.xpath(".//ClassDef"):
        cls = glyph.attrib["class"]
        current_classes[cls].append(glyph.attrib["glyph"])

    # match the sorting used in write-fonts by using the min GID as the tiebreaker
    # https://github.com/googlefonts/fontations/blob/3fcc52e/write-fonts/src/tables/layout/builders.rs#L183-L189
    new_order = sorted(
        current_classes.values(), key=lambda s: (-len(s), min(glyph_map[g] for g in s))
    )
    new_order_map = {name: i + 1 for (i, cls) in enumerate(new_order) for name in cls}
    result = dict()
    for glyph in class_def.xpath(".//ClassDef"):
        cls = glyph.attrib["class"]
        new = new_order_map.get(glyph.attrib["glyph"])
        glyph.attrib["class"] = str(new)
        result[new] = int(cls)
    return result


def reorder_rules(lookup: etree.ElementTree, new_order: Dict[int, int], rule_name: str):
    # the rules can exist as siblings of other non-rule elements, so we can't just
    # clear all children and set them in the same order.
    # instead we remove them and then append them back in order, using 'addnext'
    # to ensure that we're inserting them at the right location.
    orig_order = [el for el in lookup.iterchildren(tag=rule_name)]
    if len(orig_order) == 0:
        return
    prev_el = orig_order[0].getprevious()
    for el in orig_order:
        lookup.remove(el)

    for ix in range(len(orig_order)):
        prev_ix = new_order.get(ix, ix)
        el = orig_order[prev_ix]
        el.set("index", str(ix))
        prev_el.addnext(el)
        prev_el = el

    # there was a funny issue where if we moved the last element elsewhere
    # in the ordering it would end up having incorrect indentation, so just
    # reindent everything to be safe.
    # compute the actual nesting depth instead of hardcoding level=4, since
    # the subtable may be wrapped inside an Extension element.
    depth = sum(1 for _ in lookup.iterancestors())
    etree.indent(lookup, level=depth)


# for each named child in container, remap the 'value' attribute using the new ordering
def remap_values(
    container: etree.ElementTree, new_order: Dict[int, int], child_name: str
):
    # for the original use we need to map from new to old, but here we need the reverse
    rev_map = {v: k for (k, v) in new_order.items()}
    for el in container.iterchildren(child_name):
        old = int(el.attrib["value"])
        el.attrib["value"] = str(rev_map[old])


# fontmake and fontc assign glyph classes differently for class-based tables;
# fontc uses GIDs but fontmake uses glyph names, so we reorder them to be consistent.
def reorder_contextual_class_based_rules(
    ttx: etree.ElementTree, tag: str, glyph_map: Dict[str, int]
):
    if tag == "GSUB":
        context_name = "ContextSubst"
        class_set_name = "SubClassSet"
        class_rule_name = "SubClassRule"

    elif tag == "GPOS":
        context_name = "ContextPos"
        class_set_name = "PosClassSet"
        class_rule_name = "PosClassRule"
    else:
        raise ValueError("must be one of 'GPOS' or 'GSUB'")

    chain_name = f"Chain{context_name}"
    chain_class_set_name = f"Chain{class_set_name}"
    chain_class_rule_name = f"Chain{class_rule_name}"

    table = ttx.find(tag)
    if table is None:
        return
    for lookup in table.xpath(".//Lookup"):
        # first handle the non-chaining case, then handle the chaining case.
        # use .//{name} instead of {name} so we also find subtables wrapped
        # inside ExtensionPos/ExtensionSubst elements.
        for ctx in lookup.findall(f".//{context_name}"):
            if ctx is None or int(ctx.attrib["Format"]) != 2:
                continue

            input_class_order = remap_class_def_ids_like_fontc(
                ctx.find("ClassDef"), glyph_map
            )
            reorder_rules(ctx, input_class_order, class_set_name)
            for class_set in ctx.findall(class_set_name):
                for class_rule in class_set.findall(class_rule_name):
                    remap_values(class_rule, input_class_order, "Class")

        for chain_ctx in lookup.findall(f".//{chain_name}"):
            if chain_ctx is None or int(chain_ctx.attrib["Format"]) != 2:
                continue
            input_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("InputClassDef"), glyph_map
            )
            reorder_rules(chain_ctx, input_class_order, chain_class_set_name)
            backtrack_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("BacktrackClassDef"), glyph_map
            )
            lookahead_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("LookAheadClassDef"), glyph_map
            )
            for class_set in chain_ctx.findall(chain_class_set_name):
                for class_rule in class_set.findall(chain_class_rule_name):
                    remap_values(class_rule, input_class_order, "Input")
                    remap_values(class_rule, backtrack_class_order, "Backtrack")
                    remap_values(class_rule, lookahead_class_order, "LookAhead")


def fill_in_gvar_deltas(
    fontc: etree.ElementTree,
    fontc_path: Path,
    fontmake: etree.ElementTree,
    fontmake_path: Path,
):
    fontc_font = TTFont(fontc_path)
    fontmake_font = TTFont(fontmake_path)
    dense_fontc_count = densify_gvar(fontc_font, fontc)
    dense_fontmake_count = densify_gvar(fontmake_font, fontmake)

    if dense_fontc_count + dense_fontmake_count > 0:
        eprint(
            f"densified {dense_fontc_count} glyphVariations in fontc, {dense_fontmake_count} in fontmake"
        )


def densify_gvar(font: TTFont, ttx: etree.ElementTree):
    gvar = ttx.find("gvar")
    if gvar is None:
        return 0
    glyf = font["glyf"]
    hMetrics = font["hmtx"].metrics
    vMetrics = getattr(font.get("vmtx"), "metrics", None)

    total_deltas_filled = 0
    for variations in gvar.xpath(".//glyphVariations"):
        coords, g = glyf._getCoordinatesAndControls(
            variations.attrib["glyph"], hMetrics, vMetrics
        )
        total_deltas_filled += int(densify_one_glyph(coords, g.endPts, variations))

    return total_deltas_filled


def densify_one_glyph(coords, ends, variations: etree.ElementTree):
    did_work = False
    for tuple_ in variations.findall("tuple"):
        deltas = [None] * len(coords)
        for delta in tuple_.findall("delta"):
            idx = int(delta.attrib["pt"])
            if idx >= len(deltas):
                continue
            deltas[idx] = (int(delta.attrib["x"]), int(delta.attrib["y"]))

        if any(d is None for d in deltas):
            did_work = True
            filled_deltas = iup_delta(deltas, coords, ends)
            for delta in tuple_.findall("delta"):
                tuple_.remove(delta)

            new_deltas = [
                {"pt": str(i), "x": str(otRound(x)), "y": str(otRound(y))}
                for (i, (x, y)) in enumerate(filled_deltas)
            ]
            for attrs in new_deltas:
                new_delta = etree.Element("delta", attrs)
                tuple_.append(new_delta)

            etree.indent(tuple_, level=3)

    return did_work


def _dedent(el, space="  "):
    """Remove one level of indentation from an element subtree.

    Strips ``space`` (default: two spaces) from the end of every
    whitespace-only ``.text`` and ``.tail``, except the root element's
    ``.tail`` which belongs to the parent context.
    """
    for node in el.iter():
        if node.text and not node.text.strip() and node.text.endswith(space):
            node.text = node.text[: -len(space)]
        if (
            node is not el
            and node.tail
            and not node.tail.strip()
            and node.tail.endswith(space)
        ):
            node.tail = node.tail[: -len(space)]


def unwrap_extension_lookups(ttx: etree.ElementTree):
    """Strip Extension wrappers from GPOS/GSUB lookups.

    When a lookup uses ExtensionPos (type 9) or ExtensionSubst (type 7),
    replace it with the inner subtable so that Extension-promoted and
    non-promoted versions of the same lookup compare as equal.
    """
    for tag, ext_name, ext_type in [
        ("GPOS", "ExtensionPos", "9"),
        ("GSUB", "ExtensionSubst", "7"),
    ]:
        table = ttx.find(tag)
        if table is None:
            continue
        for lookup in table.xpath(".//Lookup"):
            lookup_type_el = lookup.find("LookupType")
            if lookup_type_el is None or lookup_type_el.attrib.get("value") != ext_type:
                continue

            extensions = lookup.findall(ext_name)
            if not extensions:
                continue

            # get the real lookup type from the first extension subtable
            inner_type_el = extensions[0].find("ExtensionLookupType")
            if inner_type_el is None:
                continue
            real_type = inner_type_el.attrib["value"]

            # update the LookupType
            lookup_type_el.attrib["value"] = real_type

            # replace each Extension element with its inner subtable
            for ext in extensions:
                inner = None
                for child in ext:
                    if child.tag not in ("ExtensionLookupType",):
                        inner = child
                        break
                if inner is None:
                    continue
                # transfer the index attribute from the Extension to the inner subtable,
                # inserting it before existing attributes so the attribute order
                # matches non-Extension subtables (index="0" Format="2" etc.)
                if "index" in ext.attrib:
                    old_attrib = dict(inner.attrib)
                    inner.attrib.clear()
                    inner.set("index", ext.attrib["index"])
                    inner.attrib.update(old_attrib)
                # replace Extension element with the unwrapped subtable
                parent = ext.getparent()
                idx = list(parent).index(ext)
                parent.remove(ext)
                parent.insert(idx, inner)
                # ext.tail has the correct Lookup-level whitespace;
                # transfer it and strip the extra indent from the subtree.
                inner.tail = ext.tail
                _dedent(inner)


def reduce_diff_noise(fontc: etree.ElementTree, fontmake: etree.ElementTree):
    fontmake_glyph_map = {
        el.attrib["name"]: int(el.attrib["id"])
        for el in fontmake.xpath("//GlyphOrder/GlyphID")
    }

    if flags.FLAGS.unwrap_extensions:
        # unwrap Extension lookups before other normalizations so that
        # contextual class remapping etc. see the inner subtables directly.
        for ttx in (fontc, fontmake):
            unwrap_extension_lookups(ttx)

    with timed("sort indices"):
        sort_indices(fontmake, "GPOS", "//Feature", "LookupListIndex")
        sort_indices(fontmake, "GSUB", "//LangSys", "FeatureIndex")
        sort_indices(fontmake, "GSUB", "//DefaultLangSys", "FeatureIndex")
    reorder_contextual_class_based_rules(fontmake, "GSUB", fontmake_glyph_map)
    reorder_contextual_class_based_rules(fontmake, "GPOS", fontmake_glyph_map)
    for ttx in (fontc, fontmake):
        # different name ids with the same value is fine

        with timed("name id to name"):
            name_id_to_name(ttx, "fvar//NamedInstance", "subfamilyNameID")
            name_id_to_name(ttx, "fvar//NamedInstance", "postscriptNameID")
            name_id_to_name(ttx, "STAT//AxisNameID", "value")
            name_id_to_name(ttx, "fvar//AxisNameID", None)
            name_id_to_name(ttx, "GPOS/FeatureList//UINameID", "value")
            name_id_to_name(ttx, "GSUB/FeatureList//UINameID", "value")
            name_id_to_name(ttx, "GSUB/FeatureList//FeatUILabelNameID", "value")
            name_id_to_name(ttx, "GSUB/FeatureList//FeatUITooltipTextNameID", "value")
            name_id_to_name(ttx, "GSUB/FeatureList//SampleTextNameID", "value")
            name_id_to_name(ttx, "GSUB/FeatureList//FirstParamUILabelNameID", "value")
            name_id_to_name(ttx, "STAT//ValueNameID", "value")
            name_id_to_name(ttx, "STAT//ElidedFallbackNameID", "value")
        normalize_null_tags(ttx, "//OS_2/achVendID", "value")

        # deal with https://github.com/googlefonts/fontmake/issues/1003
        drop_weird_names(ttx)

        strip_fontc_version_tag(ttx)

        # for matching purposes checksum is just noise
        erase_checksum(ttx)

        stat_like_fontmake(ttx)

        remove_mark_and_kern_and_curs_lookups(ttx)

        erase_type_from_stranded_points(ttx)
        with timed("gdef work"):
            remove_gdef_lig_caret_and_var_store(ttx)
            sort_gdef_mark_filter_sets(ttx)

        # sort names within the name table (do this at the end, so ids are correct
        # for earlier steps)
        normalize_name_ids(ttx)

    # Normalize glyf contour order but only when contours are identical

    with timed("normalize glyf contours"):
        fontc_point_orders, fontmake_point_orders = normalize_glyf_contours(
            fontc, fontmake
        )
    with timed("normalize gvar contours"):
        normalize_gvar_contours(fontc, fontc_point_orders)
        normalize_gvar_contours(fontmake, fontmake_point_orders)

    if FLAGS.instance is None:
        allow_fontc_only_variations_postscript_prefix(fontc, fontmake)
    # in instance mode we deliberately leave name 25 alone: it is the Variations
    # PostScript Name Prefix, which is meaningless in a static font, so fontc
    # emitting one is a diff worth seeing rather than noise to hide

    with timed("allow off-by-ones"):
        allow_some_off_by_ones(fontc, fontmake, "glyf/TTGlyph", "name", "/contour/pt")
        allow_some_off_by_ones(
            fontc, fontmake, "gvar/glyphVariations", "glyph", "/tuple/delta"
        )


# given a font file, return a dictionary of tags -> size in bytes
def get_table_sizes(fontfile: Path) -> dict[str, int]:
    cmd = ["ttx", "-l", str(fontfile)]
    stdout = log_and_run(cmd, check=True).stdout
    result = dict()

    for line in stdout.strip().splitlines()[3:]:
        split = line.split()
        result[split[0]] = int(split[2])

    return result


# return a dict of table tag  -> size difference
# only when size difference exceeds some threshold
def check_sizes(fontmake_font: Path, fontc_font: Path):
    THRESHOLD = 1 / 10
    fontmake = get_table_sizes(fontmake_font)
    fontc = get_table_sizes(fontc_font)

    output = dict()
    shared_keys = set(fontmake.keys() & fontc.keys())

    for key in shared_keys:
        fontmake_len = fontmake[key]
        fontc_len = fontc[key]
        if fontc_len < fontmake_len:
            continue
        len_ratio = min(fontc_len, fontmake_len) / max(fontc_len, fontmake_len)
        if (1 - len_ratio) > THRESHOLD:
            rel_len = fontc_len - fontmake_len
            eprint(f"{key} {fontmake_len} {fontc_len} {len_ratio:.3} {rel_len}")
            output[key] = rel_len
    return output


# returns a dictionary of {"compiler_name":  {"tag": "xml_text"}}
def generate_output(
    build_dir: Path, otl_norm_bin: Path, fontmake_font: Path, fontc_font: Path
):
    with timed("ttx fontc"):
        fontc_ttx = run_ttx(fontc_font)
    with timed("ttx fontmake"):
        fontmake_ttx = run_ttx(fontmake_font)
    with timed("normalize fontc gpos"):
        fontc_gpos = run_normalizer(otl_norm_bin, fontc_font, "gpos")
    with timed("normalize fontmake gpos"):
        fontmake_gpos = run_normalizer(otl_norm_bin, fontmake_font, "gpos")
    with timed("normalize fontc gdef"):
        fontc_gdef = run_normalizer(otl_norm_bin, fontc_font, "gdef")
    with timed("normalize fontmake gdef"):
        fontmake_gdef = run_normalizer(otl_norm_bin, fontmake_font, "gdef")

    fontc = etree.parse(fontc_ttx)
    fontmake = etree.parse(fontmake_ttx)
    with timed("fill_in_gvar_deltas"):
        fill_in_gvar_deltas(fontc, fontc_font, fontmake, fontmake_font)
    with timed("reduce_diff_noise"):
        reduce_diff_noise(fontc, fontmake)

    with timed("extract_comparables fontc"):
        fontc = extract_comparables(fontc, build_dir, "fontc")
    with timed("extract_comparables fontmake"):
        fontmake = extract_comparables(fontmake, build_dir, "fontmake")
    with timed("check_sizes"):
        size_diffs = check_sizes(fontmake_font, fontc_font)
    fontc[MARK_KERN_NAME] = fontc_gpos
    fontmake[MARK_KERN_NAME] = fontmake_gpos
    if len(fontc_gdef):
        fontc[LIG_CARET_NAME] = fontc_gdef
    if len(fontmake_gdef):
        fontmake[LIG_CARET_NAME] = fontmake_gdef
    result = {"fontc": fontc, "fontmake": fontmake}
    if len(size_diffs) > 0:
        result["sizes"] = size_diffs

    return result


def print_output(build_dir: Path, output: dict[str, dict[str, Any]]):
    fontc = output["fontc"]
    fontmake = output["fontmake"]
    print("COMPARISON")
    t1 = set(fontc.keys())
    t2 = set(fontmake.keys())
    if t1 != t2:
        if t1 - t2:
            tags = ", ".join(f"'{t}'" for t in sorted(t1 - t2))
            print(f"  Only fontc produced {tags}")

        if t2 - t1:
            tags = ", ".join(f"'{t}'" for t in sorted(t2 - t1))
            print(f"  Only fontmake produced {tags}")

    for tag in sorted(t1 & t2):
        t1s = fontc[tag]
        t2s = fontmake[tag]
        if t1s == t2s:
            print(f"  Identical '{tag}'")
        else:
            difference = diff_ratio(t1s, t2s)
            p1 = build_dir / path_for_output_item(tag, "fontc")
            p2 = build_dir / path_for_output_item(tag, "fontmake")
            print(f"  DIFF '{tag}', {rel_user(p1)} {rel_user(p2)} ({difference:.3%})")
    if output.get("sizes"):
        print("SIZE DIFFERENCES")
    for tag, diff in output.get("sizes", {}).items():
        print(f"SIZE DIFFERENCE: '{tag}': {diff}B")


def jsonify_output(output: dict[str, dict[str, Any]]):
    fontc = output["fontc"]
    fontmake = output["fontmake"]
    sizes = output.get("sizes", {})
    all_tags = set(fontc.keys()) | set(fontmake.keys())
    out = dict()
    same_lines = 0
    different_lines = 0
    for tag in all_tags:
        if tag not in fontc:
            different_lines += len(fontmake[tag])
            out[tag] = "fontmake"
        elif tag not in fontmake:
            different_lines += len(fontc[tag])
            out[tag] = "fontc"
        else:
            s1 = fontc[tag]
            s2 = fontmake[tag]
            if s1 != s2:
                ratio = diff_ratio(s1, s2)
                n_lines = max(len(s1), len(s2))
                same_lines += int(n_lines * ratio)
                different_lines += int(n_lines * (1 - ratio))
                out[tag] = ratio
            else:
                same_lines += len(s1)

    # then also add in size differences, if any
    for tag, size_diff in sizes.items():
        out[f"sizeof({tag})"] = size_diff
        # hacky: we don't want to be perfect if we have a size diff,
        # so let's pretend that whatever our size diff is, it corresponds
        # to some fictional table 100 lines liong
        different_lines += 100

    overall_diff_ratio = same_lines / (same_lines + different_lines)
    out["total"] = overall_diff_ratio
    return {"success": out}


def print_json(output):
    as_json = json.dumps(output, indent=2)
    print(as_json)


# given the ttx for a font, return a map of tags -> xml text for each root table.
# also writes the xml to individual files
def extract_comparables(font_xml, build_dir: Path, compiler: str) -> dict[str, str]:
    comparables = dict()
    tables = {e.tag: e for e in font_xml.getroot()}
    for tag in sorted(e.tag for e in font_xml.getroot()):
        table_str = to_xml_string(tables[tag])
        path = build_dir / f"{compiler}.{tag}.ttx"
        path.write_bytes(table_str)
        comparables[tag] = table_str

    return comparables


# the line-wise ratio of difference, i.e. the fraction of lines that are the same
def diff_ratio(text1: str, text2: str) -> float:
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    m = SequenceMatcher(None, lines1, lines2)
    return m.quick_ratio()


def path_for_output_item(tag_or_normalizer_name: str, compiler: str) -> str:
    if tag_or_normalizer_name == MARK_KERN_NAME:
        return f"{compiler}.markkern.txt"
    else:
        return f"{compiler}.{tag_or_normalizer_name}.ttx"


# log or print as json any compilation failures (and exit if there were any)
def report_errors_and_exit_if_there_were_any(errors: dict):
    if len(errors) == 0:
        return
    for error in errors.values():
        cmd = error["command"]
        stderr = error["stderr"]
        eprint(f"command '{cmd}' failed: '{stderr}'")

    if FLAGS.json:
        print_json({"error": errors})
    sys.exit(2)


# for reproducing crater results we have a syntax that lets you specify a
# repo url as the source.
# in this scheme we pass the path to the particular source (relative the repo root)
# as a url fragment
def resolve_source(source: str) -> Path:
    if source.startswith("git@") or source.startswith("https://"):
        source_url = urlparse(source)
        repo_path = source_url.fragment
        org_name = source_url.path.split("/")[-2]
        repo_name = source_url.path.split("/")[-1]
        sha = source_url.query
        local_repo = (
            Path.home() / ".fontc_crater_cache" / org_name / repo_name
        ).resolve()
        if not local_repo.parent.is_dir():
            local_repo.parent.mkdir(parents=True)
        if not local_repo.is_dir():
            cmd = ("git", "clone", source_url._replace(fragment="", query="").geturl())
            print("Running", " ".join(cmd), "in", local_repo.parent)
            subprocess.run(cmd, cwd=local_repo.parent, check=True)
        else:
            print(f"Reusing existing {rel_user(local_repo)}")

        if len(sha) > 0:
            log_and_run(("git", "checkout", sha), cwd=local_repo, check=True)
        source = local_repo / repo_path
    else:
        source = Path(source)
    if not source.exists():
        sys.exit(f"No such source: {source}")
    return source


def delete_things_we_must_rebuild(
    rebuild: str, fontmake_font: Path, fontc_font: Path, skip_fonts: bool = False
):
    # we delete all resources that we have to rebuild. The rest of the script
    # will assume it can reuse anything that still exists.
    # (with_suffix() carries the flavor through: fontc.otf -> fontc.ttx etc.)
    for tool, font_path in [("fontmake", fontmake_font), ("fontc", fontc_font)]:
        must_rebuild = rebuild in [tool, "both"]
        if must_rebuild:
            paths = [
                font_path.with_suffix(".ttx"),
                font_path.with_suffix(".markkern.txt"),
                font_path.with_suffix(".ligcaret.txt"),
            ]
            if not skip_fonts:
                paths.append(font_path)
            for path in paths:
                if path.exists():
                    os.remove(path)


# returns the path to the compiled binary
def build_crate(manifest_path: Path):
    cmd = ["cargo", "build", "--release", "--manifest-path", str(manifest_path)]
    log_and_run(cmd, cwd=None, check=True)


def get_fontc_and_normalizer_binary_paths(root_dir: Path) -> Tuple[Path, Path]:
    fontc_path = FLAGS.fontc_path
    norm_path = FLAGS.normalizer_path
    if fontc_path is None:
        fontc_manifest_path = root_dir / "fontc" / "Cargo.toml"
        fontc_path = root_dir / "target" / "release" / "fontc"
        build_crate(fontc_manifest_path)
        assert fontc_path.is_file(), "failed to build fontc?"
    else:
        fontc_path = Path(fontc_path)
        assert fontc_path.is_file(), f"fontc path '{fontc_path}' does not exist"
    if norm_path is None:
        otl_norm_manifest_path = root_dir / "otl-normalizer" / "Cargo.toml"
        norm_path = root_dir / "target" / "release" / "otl-normalizer"
        build_crate(otl_norm_manifest_path)
        assert norm_path.is_file(), "failed to build otl-normalizer?"
    else:
        norm_path = Path(norm_path)
        assert norm_path.is_file(), f"normalizer path '{norm_path}' does not exist"

    return (fontc_path, norm_path)


def get_crate_path(
    bin_path: Optional[str], root_dir: Optional[Path], crate_name: str
) -> Path:
    """Get path to a crate binary, building it if in fontc repo, or finding in PATH.

    Args:
        bin_path: Path provided via CLI flag
        root_dir: Path to fontc repository root (if we're in one)
        crate_name: Name of the crate (e.g., "fontc" or "otl-normalizer")

    Returns:
        Path to the binary

    Raises:
        SystemExit: If binary cannot be found or built
    """
    if bin_path:
        path = Path(bin_path)
        if not path.is_file():
            sys.exit(f"Specified {crate_name} path '{path}' does not exist")
        return path

    # If we're in the fontc repo, try to build it
    if root_dir is not None:
        manifest_path = root_dir / crate_name / "Cargo.toml"
        if manifest_path.is_file():
            built_path = root_dir / "target" / "release" / crate_name
            build_crate(manifest_path)
            if built_path.is_file():
                return built_path

    # Try to find in PATH
    which_result = shutil.which(crate_name)
    if which_result:
        return Path(which_result)

    # Give helpful error message
    sys.exit(
        f"Could not find '{crate_name}' binary. Please either:\n"
        f"  1. Specify the path with --{crate_name}_path flag\n"
        f"  2. Install {crate_name} and ensure it's in your PATH\n"
        f"  3. Run from the fontc repository root to build it automatically"
    )


def main(argv):
    if FLAGS.version:
        print(f"ttx-diff version {__version__}")
        sys.exit(0)

    has_source = len(argv) == 2

    if (FLAGS.fontc_font is None) != (FLAGS.fontmake_font is None):
        sys.exit(
            "When using precompiled fonts, both --fontc_font and --fontmake_font must be provided"
        )

    has_precompiled_fonts = FLAGS.fontc_font is not None

    if has_precompiled_fonts and has_source:
        sys.exit("Cannot specify both a source file and precompiled fonts")

    if not has_precompiled_fonts and not has_source:
        sys.exit(
            "Either a source file or both --fontc_font and --fontmake_font must be provided"
        )

    source = resolve_source(argv[1]).resolve() if has_source else None

    if FLAGS.print_instances:
        if source is None:
            sys.exit("--print_instances needs a source file")
        print_instances(source)
        sys.exit(0)

    instance = None
    if FLAGS.instance is not None:
        if source is None:
            sys.exit("--instance needs a source file, not precompiled fonts")
        if FLAGS.compare != "default":
            sys.exit(
                f"--instance is only supported with --compare default, not "
                f"'{FLAGS.compare}'"
            )
        # the mirror image of the otf guard below: that one needs a static
        # source, this one needs a variable source to interpolate from
        if not source_is_variable(source):
            skip(
                SKIP_INSTANCE_STATIC,
                f"--instance requires a variable source, but '{rel_user(source)}' "
                "is static",
            )
        instance = resolve_instance(source, FLAGS.instance)
        eprint(
            f"instance {instance.name!r}: fontmake -i {instance.fontmake_arg()} / "
            f"fontc --instance {instance.fontc_arg()}"
        )

    if FLAGS.flavor == FLAVOR_OTF:
        # CFF (flavor otf) is static-only on both sides for now: fontc has no
        # CFF2 writer, so there is nothing to compare a variable build against.
        # Instance mode is exempt: the output is static even though the source
        # is variable, so CFF is exactly as comparable as it is for a static.
        if source is not None and instance is None and source_is_variable(source):
            skip(
                SKIP_OTF_VARIABLE,
                f"--flavor otf requires a static source, but '{rel_user(source)}' is "
                "variable (fontc cannot write CFF2 yet)",
            )
        if FLAGS.compare != "default":
            sys.exit(
                f"--flavor otf is only supported with --compare default, not "
                f"'{FLAGS.compare}'"
            )

    # Check if we're in the fontc repository (optional - allows building binaries)
    cwd = Path(".").resolve()
    fontc_repo_root = None
    if (cwd / "fontc" / "Cargo.toml").is_file():
        fontc_repo_root = cwd
        eprint(f"Detected fontc repository at {rel_user(fontc_repo_root)}")

    # Get binary paths - will look in PATH or build if in repo
    fontc_bin_path = None
    if not has_precompiled_fonts:
        fontc_bin_path = get_crate_path(FLAGS.fontc_path, fontc_repo_root, "fontc")

    otl_bin_path = get_crate_path(
        FLAGS.normalizer_path, fontc_repo_root, "otl-normalizer"
    )

    if not has_precompiled_fonts and shutil.which("fontmake") is None:
        sys.exit("No fontmake")
    if shutil.which("ttx") is None:
        sys.exit("No ttx")

    if FLAGS.outdir is not None:
        out_dir = Path(FLAGS.outdir).resolve()
        if not out_dir.exists():
            sys.exit(f"Specified output directory {out_dir} does not exist")
    elif fontc_repo_root is not None:
        # If in fontc repo, use repo's build directory
        out_dir = fontc_repo_root / "build"
    else:
        # Otherwise use current directory
        out_dir = cwd / "ttx_diff_output"
        eprint(f"No --outdir specified, using {rel_user(out_dir)}")

    diffs = False

    compare = FLAGS.compare
    build_dir = out_dir / compare
    build_dir.mkdir(parents=True, exist_ok=True)
    eprint(f"Compare {compare} in {rel_user(build_dir)}")

    failures = dict()

    fontmake_out = output_font_path(build_dir, "fontmake")
    fontc_out = output_font_path(build_dir, "fontc")

    total_start = time.time()

    if has_precompiled_fonts:
        eprint("Using precompiled fonts:")
        eprint(f"  fontc: {rel_user(FLAGS.fontc_font)}")
        eprint(f"  fontmake: {rel_user(FLAGS.fontmake_font)}")

        fontc_input = Path(FLAGS.fontc_font).resolve()
        fontmake_input = Path(FLAGS.fontmake_font).resolve()

        if not fontc_input.is_file():
            sys.exit(f"fontc font not found: {fontc_input}")
        if not fontmake_input.is_file():
            sys.exit(f"fontmake font not found: {fontmake_input}")

        # When using precompiled fonts, always clean up and rebuild all derived files
        if FLAGS.rebuild != "both":
            eprint(
                "WARN: --rebuild flag ignored with precompiled fonts (always rebuilds derived files)"
            )
        delete_things_we_must_rebuild("both", fontmake_out, fontc_out, skip_fonts=True)

        if fontc_input != fontc_out:
            copy(fontc_input, fontc_out)
        if fontmake_input != fontmake_out:
            copy(fontmake_input, fontmake_out)
    else:
        delete_things_we_must_rebuild(FLAGS.rebuild, fontmake_out, fontc_out)

        with timed("build fontc"):
            try:
                if compare == "default":
                    build_fontc(source, fontc_bin_path, build_dir, instance)
                else:
                    run_gftools(
                        source, FLAGS.config, build_dir, fontc_bin=fontc_bin_path
                    )
            except BuildFail as e:
                failures["fontc"] = {
                    "command": " ".join(e.command),
                    "stderr": e.msg[-MAX_ERR_LEN:],
                }
        with timed("build fontmake"):
            try:
                if compare == "default":
                    build_fontmake(source, build_dir, instance)
                else:
                    run_gftools(source, FLAGS.config, build_dir)
            except BuildFail as e:
                failures["fontmake"] = {
                    "command": " ".join(e.command),
                    "stderr": e.msg[-MAX_ERR_LEN:],
                }

    report_errors_and_exit_if_there_were_any(failures)

    # if compilation completed, these exist
    assert fontmake_out.is_file(), fontmake_out
    assert fontc_out.is_file(), fontc_out

    output = generate_output(build_dir, otl_bin_path, fontmake_out, fontc_out)
    if output["fontc"] == output["fontmake"]:
        eprint("output is identical")
    else:
        diffs = True
        if not FLAGS.json:
            print_output(build_dir, output)
        else:
            output = jsonify_output(output)
            print_json(output)

    def format_time(secs: float) -> str:
        mins = int(secs / 60)
        if mins > 0:
            secs -= mins * 60
            return f"{mins}m{secs:.3f}s"
        return f"{secs:.3f}s"

    if FLAGS.timings:
        total = time.time() - total_start
        eprint("TIMINGS")
        for label, elapsed, depth in _timing_log:
            indent = "  " * depth
            eprint(f"  {indent}{format_time(elapsed)}  {label}")
        eprint(f"  {format_time(total)}  total")

    sys.exit(diffs * 2)  # 0 or 2
