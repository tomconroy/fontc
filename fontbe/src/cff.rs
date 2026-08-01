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
use fontir::ir::StaticMetadata;
use kurbo::{BezPath, PathEl, Point};
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

/// The first entry in the name table with the given id, if any.
fn name(static_metadata: &StaticMetadata, id: NameId) -> Option<String> {
    static_metadata
        .names
        .iter()
        .find(|(key, _)| key.name_id == id)
        .map(|(_, value)| value.clone())
}

/// ufo2ft coerces missing notice/copyright to "" and replaces the copyright
/// sign; we don't (yet) apply its full PostScript string normalization.
fn postscript_string(value: Option<String>) -> Option<String> {
    Some(value.unwrap_or_default().replace('\u{00a9}', "Copyright"))
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
            full_name: name(&static_metadata, NameId::FULL_NAME),
            family_name,
            // ufo2ft takes this from postscriptWeightName, which has no IR
            // equivalent; fonts without it don't get a Weight entry
            weight: None,
            is_fixed_pitch: static_metadata.misc.is_fixed_pitch.unwrap_or_default(),
            italic_angle: static_metadata.italic_angle.into_inner(),
            underline_position: metrics.underline_position.into_inner().ot_round(),
            underline_thickness: metrics.underline_thickness.into_inner().ot_round(),
            font_matrix: Some([1.0 / upem, 0.0, 0.0, 1.0 / upem, 0.0, 0.0]),
        };

        // TODO: populate blue zones and stem widths once fontir carries
        // PostScript hinting metadata; ufo2ft fills these from fontinfo
        let private = PrivateDictValues::default();

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
            // treats an all-zero box as no box at all
            let bounds = charstring
                .bounds
                .map(|r| {
                    [
                        r.x0.ot_round(),
                        r.y0.ot_round(),
                        r.x1.ot_round(),
                        r.y1.ot_round(),
                    ]
                })
                .filter(|bounds| *bounds != [0; 4]);
            glyph_bounds.push(bounds);
            builder.add_glyph(GlyphData {
                name: final_name,
                advance_width: instance.width,
                charstring: charstring.bytes,
                bounds,
            });
        }

        let cff = builder.build()?;
        context.cff.set(CffOutput {
            table: cff.as_bytes().to_vec(),
            glyph_bounds,
        });
        Ok(())
    }
}
