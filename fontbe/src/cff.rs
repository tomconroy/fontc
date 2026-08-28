//! Generates a [CFF](https://learn.microsoft.com/en-us/typography/opentype/spec/cff) table.
//!
//! Only static (single-master) fonts are supported; variable PostScript
//! outlines require CFF2, which is future work. Components must have been
//! decomposed by fontir ([`Flags::CFF_OUTLINES`] implies
//! [`Flags::DECOMPOSE_COMPONENTS`] when set via the CLI).

use fontdrasil::{
    orchestration::{Access, Work},
    types::GlyphName,
};
use fontir::ir::{PostscriptSettings, StaticMetadata};
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
    error::Error,
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
fn outer_bounds(charstring: &charstring::Charstring) -> Option<[i32; 4]> {
    charstring.bounds.map(|r| {
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
fn private_dict_values(ps: &PostscriptSettings) -> PrivateDictValues {
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

/// Drop the explicit closing line segment of each closed contour.
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
fn drop_explicit_closing_lines(path: &BezPath) -> BezPath {
    fn round(p: Point) -> (i32, i32) {
        (p.x.ot_round(), p.y.ot_round())
    }
    let mut out: Vec<PathEl> = Vec::with_capacity(path.elements().len());
    let mut contour_start = (0, 0);
    for el in path.elements() {
        match el {
            PathEl::MoveTo(p) => contour_start = round(*p),
            PathEl::ClosePath => {
                if matches!(out.last(), Some(PathEl::LineTo(p)) if round(*p) == contour_start) {
                    out.pop();
                }
            }
            _ => (),
        }
        out.push(*el);
    }
    out.into_iter().collect()
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

    /// Generate [CFF](https://learn.microsoft.com/en-us/typography/opentype/spec/cff)
    fn exec(&self, context: &Context) -> Result<(), Error> {
        let static_metadata = context.ir.static_metadata.get();
        if !static_metadata.axes.is_empty() {
            return Err(Error::CffNotStatic);
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
            if !instance.components.is_empty() {
                return Err(Error::CffGlyphHasComponents(glyph_name.clone()));
            }
            let keep_closing_lines = synthesized_notdef && *glyph_name == notdef;

            let mut pen = charstring::CharstringBuilder::new();
            for contour in &instance.contours {
                // CFF keeps the source (counter-clockwise) contour direction,
                // so unlike glyf there is nothing to reverse here
                if keep_closing_lines {
                    pen.append_path(contour);
                } else {
                    pen.append_path(&drop_explicit_closing_lines(contour));
                }
            }
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
            glyph_outer_bounds.push(outer_bounds(&charstring));
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
            glyph_bounds,
            glyph_outer_bounds,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
