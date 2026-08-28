//! Generates a [CFF](https://learn.microsoft.com/en-us/typography/opentype/spec/cff) table.
//!
//! This is the single-master table; a variable source gets a CFF2 from
//! [`crate::cff2`] instead, built by the same work. Components must have been
//! decomposed by fontir ([`Flags::CFF_OUTLINES`] implies
//! [`Flags::DECOMPOSE_COMPONENTS`] when set via the CLI).

use std::collections::HashMap;

use fontdrasil::{
    coords::NormalizedLocation,
    orchestration::{Access, Work},
    types::GlyphName,
};
use fontir::ir::{self, PostscriptSettings, StaticMetadata};
use kurbo::{BezPath, PathEl, Point};
use ordered_float::OrderedFloat;
use unicode_normalization::UnicodeNormalization;
use write_fonts::{
    OtRound,
    ps::cff::v1::{CffFontBuilder, GlyphData, PrivateDictValues, TopDictValues, charstring},
    tables::glyf::Bbox,
    types::NameId,
};

use crate::{
    error::{Error, GlyphProblem},
    glyphs::CheckedGlyph,
    orchestration::{AnyWorkId, BeWork, CffOutput, Context, WorkId},
    post::final_glyph_names,
};

#[derive(Debug)]
struct CffWork {}

pub fn create_cff_work() -> Box<BeWork> {
    Box::new(CffWork {})
}

/// The US-English entry in the name table with the given id, if any.
///
/// A localized font has several records per id — Gasoek One's family name is
/// there in both Korean and English — and the CFF, which has no notion of
/// language, gets the English one. ufo2ft reads the UFO's own (unlocalized)
/// font info for these, which is what the English record was built from.
fn name(static_metadata: &StaticMetadata, id: NameId) -> Option<String> {
    const US_ENGLISH: u16 = 0x409;
    static_metadata
        .names
        .iter()
        .find(|(key, _)| key.name_id == id && key.lang_id == US_ENGLISH)
        .map(|(_, value)| value.clone())
}

/// The characters ufo2ft's `normalizeStringForPostscript` deletes outright.
const POSTSCRIPT_STRING_EXCEPTIONS: &str = "[](){}<>/%";

/// Whether a character may stand in a PostScript string as-is, which ufo2ft
/// takes to be `chr(33)..=chr(126)`. Note that this excludes the space, so
/// spaces take the decompose-and-fall-back path below (where they survive,
/// being ASCII) rather than the fast path.
fn is_postscript_char(c: char) -> bool {
    matches!(c, '\u{21}'..='\u{7e}')
}

/// Port of ufo2ft's `normalizeStringForPostscript` (with `allowSpaces`, which
/// is what the CFF Notice/Copyright path uses).
///
/// Deletes `[](){}<>/%`, keeps printable ASCII as-is, and puts everything else
/// through NFKD: if the decomposition is entirely printable ASCII it is
/// substituted (so `™` → `TM`, `ﬁ` → `fi`), and otherwise the decomposition is
/// ASCII-encoded with one `?` per character that doesn't fit — so `é`, which
/// decomposes to `e` plus a combining acute, becomes `e?`, not `e`.
fn normalize_string_for_postscript(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if POSTSCRIPT_STRING_EXCEPTIONS.contains(c) {
            continue;
        }
        if is_postscript_char(c) {
            out.push(c);
            continue;
        }
        let decomposed: String = c.nfkd().collect();
        if decomposed.chars().all(is_postscript_char) {
            out.push_str(&decomposed);
        } else {
            out.extend(
                decomposed
                    .chars()
                    .map(|c| if c.is_ascii() { c } else { '?' }),
            );
        }
    }
    out
}

/// The Notice/Copyright string ufo2ft would store for a name table value.
///
/// ufo2ft coerces a missing value to "", replaces the copyright sign, and only
/// then normalizes for PostScript — so a `©` becomes `Copyright`, not `?`.
fn postscript_string(value: Option<String>) -> Option<String> {
    Some(normalize_string_for_postscript(
        &value.unwrap_or_default().replace('\u{00a9}', "Copyright"),
    ))
}

/// The smallest integer box enclosing a charstring's exact bounds.
///
/// fontTools recalculates head, hhea and vhea from the compiled charstrings
/// when it writes a CFF font, and it grows the bounds outward rather than
/// rounding them: `intRect` for the FontBBox head copies, and
/// `ceil(max) - floor(min)` for the bounds width hhea/vhea extend the side
/// bearings by. A glyph whose curve overshoots its on-curve points by a
/// fraction of a unit therefore counts as a whole unit wider than the rounded
/// box in [`CffOutput::glyph_bounds`] says.
pub(crate) fn outer_bounds(bounds: Option<kurbo::Rect>) -> Option<[i32; 4]> {
    bounds.map(|r| {
        [
            r.x0.floor() as i32,
            r.y0.floor() as i32,
            r.x1.ceil() as i32,
            r.y1.ceil() as i32,
        ]
    })
}

/// The bounds the CFF work recorded for a glyph, as a [`Bbox`].
///
/// CFF bounds are i32 but the metrics tables are i16, so an outline that far
/// from the origin is an error rather than a silent wraparound.
pub(crate) fn bbox_from_cff(glyph_name: &GlyphName, bounds: [i32; 4]) -> Result<Bbox, Error> {
    let coord = |value: i32, what: &str| {
        i16::try_from(value).map_err(|_| Error::OutOfBounds {
            what: format!("{glyph_name} bbox {what}"),
            value: value.to_string(),
        })
    };
    Ok(Bbox {
        x_min: coord(bounds[0], "x_min")?,
        y_min: coord(bounds[1], "y_min")?,
        x_max: coord(bounds[2], "x_max")?,
        y_max: coord(bounds[3], "y_max")?,
    })
}

/// The Private DICT ufo2ft would build from the same PostScript fontinfo.
///
/// ufo2ft rounds every array element; writes the blue-related scalars only
/// when at least one blues array is non-empty, using its own fallbacks
/// (BlueFuzz 0 — not the CFF default 1 — and a computed BlueScale); writes
/// stems only when *both* stem-snap lists are non-empty, with StdHW/StdVW
/// taken from the first element *before* sorting; and bypasses width
/// optimization when the source states either width explicitly.
pub(crate) fn private_dict_values(ps: &PostscriptSettings) -> PrivateDictValues {
    fn rounded(values: &[OrderedFloat<f64>]) -> Vec<i32> {
        values.iter().map(|v| v.into_inner().ot_round()).collect()
    }
    let mut private = PrivateDictValues::default();

    let blue_values = rounded(&ps.blue_values);
    let other_blues = rounded(&ps.other_blues);
    let family_blues = rounded(&ps.family_blues);
    let family_other_blues = rounded(&ps.family_other_blues);
    if [
        &blue_values,
        &other_blues,
        &family_blues,
        &family_other_blues,
    ]
    .iter()
    .any(|values| !values.is_empty())
    {
        private.blue_fuzz = Some(
            ps.blue_fuzz
                .map(|v| v.into_inner())
                .unwrap_or(0.0)
                .ot_round(),
        );
        private.blue_shift = Some(
            ps.blue_shift
                .map(|v| v.into_inner())
                .unwrap_or(7.0)
                .ot_round(),
        );
        private.blue_scale = Some(
            ps.blue_scale
                .map(|v| v.into_inner())
                .unwrap_or_else(|| fallback_blue_scale(&ps.blue_values, &ps.other_blues)),
        );
        private.force_bold = Some(ps.force_bold.unwrap_or(false));
        private.blue_values = blue_values;
        private.other_blues = other_blues;
        private.family_blues = family_blues;
        private.family_other_blues = family_other_blues;
    }

    let stem_snap_h = rounded(&ps.stem_snap_h);
    let stem_snap_v = rounded(&ps.stem_snap_v);
    if !stem_snap_h.is_empty() && !stem_snap_v.is_empty() {
        private.std_hw = Some(stem_snap_h[0]);
        private.std_vw = Some(stem_snap_v[0]);
        let mut sorted_h = stem_snap_h;
        sorted_h.sort_unstable();
        let mut sorted_v = stem_snap_v;
        sorted_v.sort_unstable();
        private.stem_snap_h = sorted_h;
        private.stem_snap_v = sorted_v;
    }

    if ps.default_width_x.is_some() || ps.nominal_width_x.is_some() {
        private.default_width_x = Some(
            ps.default_width_x
                .map(|v| v.into_inner())
                .unwrap_or(200.0)
                .ot_round(),
        );
        private.nominal_width_x = Some(
            ps.nominal_width_x
                .map(|v| v.into_inner())
                .unwrap_or(0.0)
                .ot_round(),
        );
    }
    private
}

/// ufo2ft's postscriptBlueScaleFallback: 3/(4 × the tallest zone), measured
/// on the *unrounded* blue values, defaulting to the CFF default.
fn fallback_blue_scale(
    blue_values: &[OrderedFloat<f64>],
    other_blues: &[OrderedFloat<f64>],
) -> f64 {
    let max_zone_height = blue_values
        .chunks_exact(2)
        .chain(other_blues.chunks_exact(2))
        .map(|pair| (pair[1].into_inner() - pair[0].into_inner()).abs())
        .fold(0.0f64, f64::max);
    if max_zone_height != 0.0 {
        3.0 / (4.0 * max_zone_height)
    } else {
        0.039625
    }
}

/// Which explicit closing line segments a glyph's contours can lose.
///
/// fontir spells out the closing segment; in UFO semantics (and hence in
/// fontmake's charstrings, which are drawn through a point pen) the closing
/// line of a closed contour is implicit. Type 2 charstrings close contours
/// implicitly too, so an explicit closing line is pure redundancy.
///
/// Whether the last segment lands on the contour start is decided on rounded
/// coordinates, because that is what the charstring will contain; the points
/// themselves are passed through unrounded, since the pen rounds them (and
/// raises quadratics from the unrounded values).
///
/// The decision is taken **once for all masters**: a CFF2 charstring blends
/// one command list across every master, so a line that survived in one
/// master and vanished in another would leave nothing to blend. The masters
/// are interpolation compatible by this point, so they agree on where the
/// `LineTo`s are; all that can differ is whether one of them happens to round
/// onto its contour start. Returns one flag per path element, true where a
/// `LineTo` is to be dropped.
///
/// fontTools does not face this: `CFF2CharStringMergePen` draws each master
/// as its source says and then refuses to merge, with
/// `VarLibCFFPointTypeMergeError`.
fn droppable_closing_lines<'a>(paths: impl Iterator<Item = &'a BezPath>) -> Vec<bool> {
    fn round(p: Point) -> (i32, i32) {
        (p.x.ot_round(), p.y.ot_round())
    }
    let mut droppable: Option<Vec<bool>> = None;
    for path in paths {
        let els = path.elements();
        let mut mine = vec![false; els.len()];
        let mut contour_start = (0, 0);
        for (idx, el) in els.iter().enumerate() {
            match el {
                PathEl::MoveTo(p) => contour_start = round(*p),
                PathEl::LineTo(p) => {
                    let closes_the_contour = round(*p) == contour_start
                        && matches!(els.get(idx + 1), Some(PathEl::ClosePath));
                    if closes_the_contour {
                        #[allow(clippy::indexing_slicing)] // mine is els-sized
                        {
                            mine[idx] = true;
                        }
                    }
                }
                _ => (),
            }
        }
        droppable = Some(match droppable {
            None => mine,
            // every master has to agree, so this is an intersection
            Some(agreed) => agreed.iter().zip(&mine).map(|(a, b)| *a && *b).collect(),
        });
    }
    droppable.unwrap_or_default()
}

/// The endpoint of a path element, if it has one.
fn end_point(el: &PathEl) -> Option<Point> {
    match el {
        PathEl::MoveTo(p) | PathEl::LineTo(p) | PathEl::QuadTo(_, p) | PathEl::CurveTo(_, _, p) => {
            Some(*p)
        }
        PathEl::ClosePath => None,
    }
}

/// Which of a master's closed contours were drawn with an explicit closing line.
///
/// One flag per contour, in contour order. Port of ufo2ft's
/// `_has_explicit_closing_line`, which asks of the *source point list* whether
/// the contour's first on-curve point is a `line` that repeats the point before
/// it — i.e. the designer drew the closing segment too, collapsing it to zero
/// length.
/// <https://github.com/googlefonts/ufo2ft/blob/2f11b0ff/Lib/ufo2ft/filters/explicitClosingLine.py#L86-L98>
///
/// fontir always spells the closing segment out (its `close_path` is
/// `PointToSegmentPen(outputImpliedClosingLine=True)`), so here that reads as:
/// the element before `ClosePath` is a `LineTo` landing exactly on the endpoint
/// of the element before *that*. Coordinates are compared unrounded, as ufo2ft
/// compares the source points.
fn explicit_closing_lines(path: &BezPath) -> Vec<bool> {
    let els = path.elements();
    els.iter()
        .enumerate()
        .filter(|(_, el)| matches!(el, PathEl::ClosePath))
        .map(|(idx, _)| {
            let closing = idx.checked_sub(1).and_then(|i| els.get(i));
            let before = idx.checked_sub(2).and_then(|i| els.get(i));
            match (closing, before) {
                (Some(PathEl::LineTo(p)), Some(before)) => end_point(before) == Some(*p),
                _ => false,
            }
        })
        .collect()
}

/// Does this glyph need its closing lines made explicit in every master?
///
/// Port of ufo2ft's `ExplicitClosingLineIFilter`, which only the CFF
/// *interpolatable* pre-processor runs — hence this only ever fires for CFF2:
/// if some masters drew a contour's closing line and others left it implied, the
/// whole glyph is marked and every master's charstring gets it written out.
/// <https://github.com/googlefonts/ufo2ft/blob/2f11b0ff/Lib/ufo2ft/filters/explicitClosingLine.py#L34-L59>
///
/// The masters are interpolation compatible by this point, so ufo2ft's "do the
/// point types agree" precondition is already met. A single master cannot
/// disagree with itself, so this is a no-op for CFF1, which fontc only writes
/// for a font with no axes and hence one master.
fn needs_explicit_closing_lines<'a>(paths: impl Iterator<Item = &'a BezPath>) -> bool {
    let mut flags: Option<(Vec<bool>, Vec<bool>)> = None; // (any master, every master)
    let mut num_masters = 0;
    for path in paths {
        num_masters += 1;
        let mine = explicit_closing_lines(path);
        let Some((any, all)) = flags.as_mut() else {
            flags = Some((mine.clone(), mine));
            continue;
        };
        if any.len() != mine.len() {
            return false;
        }
        for (i, explicit) in mine.into_iter().enumerate() {
            #[allow(clippy::indexing_slicing)] // lengths just checked equal
            {
                any[i] |= explicit;
                all[i] &= explicit;
            }
        }
    }
    if num_masters < 2 {
        return false;
    }
    flags.is_some_and(|(any, all)| any.into_iter().zip(all).any(|(any, all)| any && !all))
}

/// Apply [`droppable_closing_lines`] to one master's path.
fn without_closing_lines(path: &BezPath, droppable: &[bool]) -> BezPath {
    path.elements()
        .iter()
        .enumerate()
        .filter(|(idx, _)| !droppable.get(*idx).copied().unwrap_or(false))
        .map(|(_, el)| *el)
        .collect()
}

/// The per-master outlines a PostScript charstring is drawn from.
///
/// The contours of each master are concatenated into one path, checked for
/// interpolation compatibility exactly as the glyf/gvar path checks them (so
/// an incompatible source fails the same way, not with a panic), and stripped
/// of the closing lines every master agrees are redundant.
///
/// `keep_closing_lines` is for the `.notdef` fontir synthesizes: like ufo2ft's
/// stub it draws the closing line of each box explicitly, and ufo2ft keeps
/// those (its `explicitClosingLine` glyph lib key), so we do too. A glyph whose
/// masters disagree about their closing lines keeps them for the same reason,
/// see [`needs_explicit_closing_lines`].
pub(crate) fn postscript_outlines(
    glyph: &ir::Glyph,
    keep_closing_lines: bool,
) -> Result<HashMap<NormalizedLocation, BezPath>, Error> {
    if glyph
        .sources()
        .values()
        .any(|instance| !instance.components.is_empty())
    {
        return Err(Error::CffGlyphHasComponents(glyph.name.clone()));
    }
    let CheckedGlyph::Contour { paths, .. } = CheckedGlyph::new(glyph)? else {
        return Err(Error::CffGlyphHasComponents(glyph.name.clone()));
    };
    if keep_closing_lines || needs_explicit_closing_lines(paths.values()) {
        return Ok(paths);
    }
    let droppable = droppable_closing_lines(paths.values());
    Ok(paths
        .iter()
        .map(|(loc, path)| (loc.clone(), without_closing_lines(path, &droppable)))
        .collect())
}

impl Work<Context, AnyWorkId, Error> for CffWork {
    fn id(&self) -> AnyWorkId {
        WorkId::Cff.into()
    }

    fn read_access(&self) -> Access<AnyWorkId> {
        // We need to read all the glyphs, but we don't know their names until
        // glyph order is final; the workload refines this once it is
        // (same pattern as the glyf/loca and gvar works)
        Access::Unknown
    }

    /// Generate [CFF](https://learn.microsoft.com/en-us/typography/opentype/spec/cff),
    /// or [CFF2](https://learn.microsoft.com/en-us/typography/opentype/spec/cff2)
    /// when the font has axes.
    ///
    /// A font has one PostScript outline table, never both, so one work
    /// produces whichever the source calls for; see [`crate::cff2`].
    fn exec(&self, context: &Context) -> Result<(), Error> {
        let static_metadata = context.ir.static_metadata.get();
        if !static_metadata.axes.is_empty() {
            context.cff.set(crate::cff2::build_cff2(context)?);
            return Ok(());
        }
        let glyph_order = context.ir.glyph_order.get();
        let metrics = context
            .ir
            .global_metrics
            .get()
            .at(static_metadata.default_location());

        let upem = static_metadata.units_per_em as f64;
        // CFF is single-master, so it wants the default master's hints; the IR
        // keeps every master's, for CFF2's benefit
        let postscript = static_metadata.postscript_default();
        let family_name = name(&static_metadata, NameId::TYPOGRAPHIC_FAMILY_NAME)
            .or_else(|| name(&static_metadata, NameId::FAMILY_NAME));
        let top_dict = TopDictValues {
            // like ufo2ft: "{versionMajor}.{versionMinor}", not head.fontRevision
            version: Some(format!(
                "{}.{}",
                static_metadata.misc.version_major, static_metadata.misc.version_minor
            )),
            notice: postscript_string(name(&static_metadata, NameId::TRADEMARK)),
            copyright: postscript_string(name(&static_metadata, NameId::COPYRIGHT_NOTICE)),
            // ufo2ft reads postscriptFullName, whose fallback is
            // "{preferred family} {preferred subfamily}" — *not* the name
            // table's full font name, which for Geo is "Geo Medium" where the
            // source states a postscriptFullName of "Geo-Regular"
            full_name: postscript.full_name.clone().or_else(|| {
                let subfamily = name(&static_metadata, NameId::TYPOGRAPHIC_SUBFAMILY_NAME)
                    .or_else(|| name(&static_metadata, NameId::SUBFAMILY_NAME));
                match (family_name.as_ref(), subfamily) {
                    (Some(family), Some(subfamily)) => Some(format!("{family} {subfamily}")),
                    _ => None,
                }
            }),
            family_name,
            // like ufo2ft: postscriptWeightName, or no Weight entry at all
            weight: postscript.weight_name.clone(),
            is_fixed_pitch: static_metadata.misc.is_fixed_pitch.unwrap_or_default(),
            italic_angle: static_metadata.italic_angle.into_inner(),
            underline_position: metrics.underline_position.into_inner().ot_round(),
            underline_thickness: metrics.underline_thickness.into_inner().ot_round(),
            font_matrix: Some([1.0 / upem, 0.0, 0.0, 1.0 / upem, 0.0, 0.0]),
        };

        let private = private_dict_values(&postscript);

        let postscript_name =
            name(&static_metadata, NameId::POSTSCRIPT_NAME).unwrap_or_else(|| {
                // ufo2ft's fallback: "{family}-{style}" without spaces
                let family = top_dict.family_name.clone().unwrap_or_default();
                let style = name(&static_metadata, NameId::SUBFAMILY_NAME)
                    .unwrap_or_else(|| "Regular".to_string());
                format!("{family}-{style}").replace(' ', "")
            });

        // The CFF charset must use the same names as post 2.0 would
        // (production names applied when enabled)
        let final_names =
            final_glyph_names(&glyph_order, static_metadata.postscript_names.as_ref());

        // fontir synthesizes a .notdef when the source has none. Like ufo2ft's
        // stub it draws the closing line of each box explicitly, and ufo2ft
        // keeps those (its `explicitClosingLine` glyph lib key), so we do too.
        let notdef: GlyphName = ".notdef".into();
        let synthesized_notdef = !context.ir.preliminary_glyph_order.get().contains(&notdef);

        let mut builder = CffFontBuilder::new(postscript_name, top_dict, private);
        let mut glyph_bounds = Vec::with_capacity(glyph_order.len());
        let mut glyph_outer_bounds = Vec::with_capacity(glyph_order.len());
        for (glyph_name, final_name) in glyph_order.names().zip(final_names) {
            let glyph = context.ir.get_glyph(glyph_name.clone());
            let instance = glyph.default_instance();
            let keep_closing_lines = synthesized_notdef && *glyph_name == notdef;
            let outlines = postscript_outlines(&glyph, keep_closing_lines)?;
            let path = outlines
                .get(static_metadata.default_location())
                .ok_or_else(|| {
                    Error::GlyphError(glyph_name.clone(), GlyphProblem::MissingDefault)
                })?;

            let mut pen = charstring::CharstringBuilder::new();
            // CFF keeps the source (counter-clockwise) contour direction, so
            // unlike glyf there is nothing to reverse here
            pen.append_path(path);
            let charstring = pen.build(None, true)?;
            // ufo2ft measures the charstring it just built, rounds each side
            // (roundTolerance 0.5 never reaches the floor/ceil fallback), and
            // treats an all-zero box as no box at all. This is the box the
            // side bearings come from.
            glyph_bounds.push(
                charstring
                    .bounds
                    .map(|r| {
                        [
                            r.x0.ot_round(),
                            r.y0.ot_round(),
                            r.x1.ot_round(),
                            r.y1.ot_round(),
                        ]
                    })
                    .filter(|bounds| *bounds != [0; 4]),
            );
            glyph_outer_bounds.push(outer_bounds(charstring.bounds));
            builder.add_glyph(GlyphData {
                name: final_name,
                advance_width: instance.width,
                charstring: charstring.bytes,
                bounds: charstring.bounds,
            });
        }

        let cff = builder.build()?;
        context.cff.set(CffOutput {
            table: cff.as_bytes().to_vec(),
            cff2: false,
            glyph_bounds,
            glyph_outer_bounds,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(svg: &str) -> BezPath {
        BezPath::from_svg(svg).unwrap()
    }

    fn floats(values: &[f64]) -> Vec<OrderedFloat<f64>> {
        values.iter().copied().map(OrderedFloat).collect()
    }

    /// Afrotype/tac: the parenthesised URL in the copyright loses its
    /// brackets *and* both slashes, which is most of what the corpus hits.
    #[test]
    fn postscript_string_strips_bracketing_and_slashes() {
        assert_eq!(
            postscript_string(Some(
                "Copyright 2024 The Tac One Project Authors \
                 (https://github.com/Afrotype/tac)"
                    .to_string()
            )),
            Some(
                "Copyright 2024 The Tac One Project Authors \
                 https:github.comAfrotypetac"
                    .to_string()
            )
        );
    }

    #[test]
    fn postscript_string_replaces_copyright_sign_before_normalizing() {
        // the © must become "Copyright", not the "?" it would decompose to
        assert_eq!(
            postscript_string(Some("\u{00a9} 2024 Someone".to_string())),
            Some("Copyright 2024 Someone".to_string())
        );
        assert_eq!(postscript_string(None), Some(String::new()));
    }

    #[test]
    fn postscript_string_decomposes_to_ascii() {
        // an all-ASCII decomposition is substituted whole
        assert_eq!(normalize_string_for_postscript("\u{2122}\u{fb01}"), "TMfi");
        // otherwise it is ASCII-encoded with one '?' per stranded character,
        // so an accented letter keeps its base letter *and* gains a '?'
        assert_eq!(normalize_string_for_postscript("caf\u{e9}"), "cafe?");
        assert_eq!(normalize_string_for_postscript("na\u{ef}ve"), "nai?ve");
        // no decomposition at all: one '?' for the character itself
        assert_eq!(normalize_string_for_postscript("\u{65e5}\u{672c}"), "??");
        // curly quotes and dashes are common in copyright strings
        assert_eq!(
            normalize_string_for_postscript("\u{2014}it\u{2019}s"),
            "?it?s"
        );
        // NBSP decomposes to a plain space, and spaces survive
        assert_eq!(normalize_string_for_postscript("a\u{a0}b c"), "a b c");
        // a decomposition may reintroduce exception characters; they stay,
        // because ufo2ft only filters the original character
        assert_eq!(normalize_string_for_postscript("\u{2474}"), "(1)");
    }

    /// A closed contour's final line back to its start is implied by the
    /// contour close, so ufo2ft's point pen never emits it and neither do we.
    #[test]
    fn a_closing_line_is_redundant() {
        let square = path("M0,0 L100,0 L100,100 L0,100 L0,0 Z");
        let droppable = droppable_closing_lines([&square].into_iter());
        assert_eq!(droppable, vec![false, false, false, false, true, false]);
        assert_eq!(
            without_closing_lines(&square, &droppable).to_svg(),
            "M0,0 L100,0 L100,100 L0,100 Z"
        );
    }

    /// But the decision is taken across every master at once. A CFF2
    /// charstring blends one command list, so a line that vanished in one
    /// master and survived in another would leave nothing to blend — here the
    /// second master's last point does not land on its contour start, so both
    /// masters keep the line.
    #[test]
    fn masters_agree_on_which_closing_lines_go() {
        let closes = path("M0,0 L100,0 L100,100 L0,100 L0,0 Z");
        let doesnt = path("M0,0 L120,0 L120,110 L0,110 L5,20 Z");
        let droppable = droppable_closing_lines([&closes, &doesnt].into_iter());
        assert!(!droppable.iter().any(|drop| *drop));
        assert_eq!(
            without_closing_lines(&closes, &droppable).to_svg(),
            closes.to_svg()
        );
        // and the two masters still draw the same sequence of commands
        assert_eq!(
            without_closing_lines(&closes, &droppable).elements().len(),
            without_closing_lines(&doesnt, &droppable).elements().len()
        );
    }

    /// Each contour is decided on its own, and an open contour has no
    /// closing line to lose.
    #[test]
    fn every_contour_is_decided_separately() {
        let two = path("M0,0 L10,0 L0,0 Z M50,0 L60,0 L60,10");
        assert_eq!(
            droppable_closing_lines([&two].into_iter()),
            vec![false, false, true, false, false, false, false]
        );
    }

    /// A master that drew its own closing line has the contour's last two
    /// points in the same place; one that left it implied does not.
    #[test]
    fn an_explicit_closing_line_is_a_repeated_point() {
        // "M0,0 L100,0 L100,100 L0,0 Z": the source's last point is (100,100),
        // and fontir added the L0,0 — the closing line is implied
        assert_eq!(
            explicit_closing_lines(&path("M0,0 L100,0 L100,100 L0,0 Z")),
            vec![false]
        );
        // "…L0,0 L0,0 Z": the source's last point is the contour start too, so
        // the closing line fontir added is zero length — drawn by the designer
        assert_eq!(
            explicit_closing_lines(&path("M0,0 L100,0 L0,0 L0,0 Z")),
            vec![true]
        );
        // a contour closed by a curve has no closing *line* whatever it does
        assert_eq!(
            explicit_closing_lines(&path("M0,0 L100,0 C50,50 20,20 0,0 Z")),
            vec![false]
        );
        // one flag per closed contour, in contour order; an open one has none
        assert_eq!(
            explicit_closing_lines(&path("M0,0 L10,0 L0,0 L0,0 Z M5,5 L6,5 L5,5 Z M9,9 L8,8")),
            vec![true, false]
        );
    }

    /// ufo2ft's `ExplicitClosingLineIFilter`: masters that disagree about a
    /// closing line make it explicit everywhere, so nothing is dropped.
    #[test]
    fn masters_that_disagree_keep_every_closing_line() {
        let implied = path("M0,0 L100,0 L100,100 L0,0 Z");
        let explicit = path("M0,0 L100,0 L0,0 L0,0 Z");
        assert!(needs_explicit_closing_lines(
            [&implied, &explicit].into_iter()
        ));
        // all masters agreeing either way is not a disagreement
        assert!(!needs_explicit_closing_lines(
            [&implied, &implied].into_iter()
        ));
        assert!(!needs_explicit_closing_lines(
            [&explicit, &explicit].into_iter()
        ));
        // and a lone master cannot disagree with itself: this is why the CFF1
        // path, which fontc only takes for a font with no axes, never sees it
        assert!(!needs_explicit_closing_lines([&explicit].into_iter()));
    }

    /// The filter is per glyph, not per contour: one contour's disagreement
    /// keeps the closing lines of all of them.
    #[test]
    fn one_disagreeing_contour_flags_the_whole_glyph() {
        let a = path("M0,0 L10,0 L0,0 L0,0 Z M50,0 L60,0 L60,10 L50,0 Z");
        let b = path("M0,0 L10,0 L0,0 L0,0 Z M50,0 L60,0 L50,0 L50,0 Z");
        assert!(needs_explicit_closing_lines([&a, &b].into_iter()));
    }

    #[test]
    fn no_hints_no_private_entries() {
        let private = private_dict_values(&PostscriptSettings::default());
        assert_eq!(private, PrivateDictValues::default());
    }

    #[test]
    fn blues_bring_scalars_with_ufo2ft_fallbacks() {
        let private = private_dict_values(&PostscriptSettings {
            blue_values: floats(&[-10.5, 0.5, 699.5, 710.4]),
            ..Default::default()
        });
        // otRound: half rounds toward +∞, even for negatives
        assert_eq!(private.blue_values, vec![-10, 1, 700, 710]);
        // ufo2ft's fallback is 0, unlike the CFF default of 1
        assert_eq!(private.blue_fuzz, Some(0));
        assert_eq!(private.blue_shift, Some(7));
        assert_eq!(private.force_bold, Some(false));
        // 3 / (4 × tallest zone), from the unrounded values: zone height 11
        assert_eq!(private.blue_scale, Some(3.0 / 44.0));
    }

    #[test]
    fn explicit_blue_scalars_pass_through() {
        let private = private_dict_values(&PostscriptSettings {
            other_blues: floats(&[-210.0, -200.0]),
            blue_scale: Some(0.05.into()),
            blue_shift: Some(8.0.into()),
            blue_fuzz: Some(2.0.into()),
            force_bold: Some(true),
            ..Default::default()
        });
        assert_eq!(private.blue_values, Vec::<i32>::new());
        assert_eq!(private.other_blues, vec![-210, -200]);
        assert_eq!(private.blue_scale, Some(0.05));
        assert_eq!(private.blue_shift, Some(8));
        assert_eq!(private.blue_fuzz, Some(2));
        assert_eq!(private.force_bold, Some(true));
    }

    #[test]
    fn no_blues_means_no_blue_scalars() {
        // the scalars only appear when a blues array does
        let private = private_dict_values(&PostscriptSettings {
            blue_fuzz: Some(2.0.into()),
            blue_scale: Some(0.05.into()),
            force_bold: Some(true),
            stem_snap_h: floats(&[30.0]),
            stem_snap_v: floats(&[80.0]),
            ..Default::default()
        });
        assert_eq!(private.blue_fuzz, None);
        assert_eq!(private.blue_scale, None);
        assert_eq!(private.blue_shift, None);
        assert_eq!(private.force_bold, None);
    }

    #[test]
    fn stems_require_both_directions() {
        let only_v = private_dict_values(&PostscriptSettings {
            stem_snap_v: floats(&[80.0, 90.0]),
            ..Default::default()
        });
        assert_eq!(only_v.std_vw, None);
        assert!(only_v.stem_snap_v.is_empty());

        let both = private_dict_values(&PostscriptSettings {
            stem_snap_h: floats(&[34.0, 30.0]),
            stem_snap_v: floats(&[90.0, 80.0]),
            ..Default::default()
        });
        // Std is the first element before sorting; the snap lists are sorted
        assert_eq!(both.std_hw, Some(34));
        assert_eq!(both.std_vw, Some(90));
        assert_eq!(both.stem_snap_h, vec![30, 34]);
        assert_eq!(both.stem_snap_v, vec![80, 90]);
    }

    #[test]
    fn either_explicit_width_bypasses_optimization_for_both() {
        let private = private_dict_values(&PostscriptSettings {
            default_width_x: Some(600.0.into()),
            ..Default::default()
        });
        // ufo2ft falls back to 200/0 for the one that isn't stated
        assert_eq!(private.default_width_x, Some(600));
        assert_eq!(private.nominal_width_x, Some(0));
    }
}
