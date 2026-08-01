//! Generates a [CFF](https://learn.microsoft.com/en-us/typography/opentype/spec/cff) table.
//!
//! Only static (single-master) fonts are supported; variable PostScript
//! outlines require CFF2, which is future work. Components must have been
//! decomposed by fontir ([`Flags::CFF_OUTLINES`] implies
//! [`Flags::DECOMPOSE_COMPONENTS`] when set via the CLI).

use fontdrasil::orchestration::{Access, Work};
use fontir::ir::StaticMetadata;
use kurbo::{BezPath, PathEl, Point, Rect, Shape};
use write_fonts::{
    OtRound,
    ps::cff::v1::{CffFontBuilder, GlyphData, PrivateDictValues, TopDictValues, charstring},
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

/// `path` with every point rounded to integer coordinates, the way
/// coordinates round when written to a charstring.
fn rounded_path(path: &BezPath) -> BezPath {
    fn round(p: Point) -> Point {
        let (x, y): (f64, f64) = (p.x.ot_round(), p.y.ot_round());
        Point::new(x, y)
    }
    path.elements()
        .iter()
        .map(|el| match *el {
            PathEl::MoveTo(p) => PathEl::MoveTo(round(p)),
            PathEl::LineTo(p) => PathEl::LineTo(round(p)),
            PathEl::QuadTo(p1, p2) => PathEl::QuadTo(round(p1), round(p2)),
            PathEl::CurveTo(p1, p2, p3) => PathEl::CurveTo(round(p1), round(p2), round(p3)),
            PathEl::ClosePath => PathEl::ClosePath,
        })
        .collect()
}

/// Drop the explicit closing line segment of each closed contour.
///
/// fontir spells out the closing segment; in UFO semantics (and hence in
/// fontmake's charstrings, which are drawn through a point pen) the closing
/// line of a closed contour is implicit. Type 2 charstrings close contours
/// implicitly too, so an explicit closing line is pure redundancy.
///
/// Expects a path whose points are already rounded, so that "lands on the
/// contour start" means what it will mean in the emitted charstring.
fn drop_explicit_closing_lines(path: &BezPath) -> BezPath {
    let mut out: Vec<PathEl> = Vec::with_capacity(path.elements().len());
    let mut contour_start = Point::ZERO;
    for el in path.elements() {
        match el {
            PathEl::MoveTo(p) => contour_start = *p,
            PathEl::ClosePath => {
                if matches!(out.last(), Some(PathEl::LineTo(p)) if *p == contour_start) {
                    out.pop();
                }
            }
            _ => (),
        }
        out.push(*el);
    }
    out.into_iter().collect()
}

/// Exact bounds of the rounded path, like fontTools' BoundsPen: curve-exact,
/// and a contour consisting of a lone moveto still contributes its point.
fn path_bounds(path: &BezPath) -> Option<Rect> {
    if path.elements().is_empty() {
        return None;
    }
    if path.segments().next().is_some() {
        return Some(path.bounding_box());
    }
    match path.elements().first() {
        Some(PathEl::MoveTo(p)) => Some(Rect::from_points(*p, *p)),
        _ => None,
    }
}

fn union(a: Option<Rect>, b: Option<Rect>) -> Option<Rect> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.union(b)),
        (a, b) => a.or(b),
    }
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

        let mut builder = CffFontBuilder::new(postscript_name, top_dict, private);
        let mut glyph_bounds = Vec::with_capacity(glyph_order.len());
        for (glyph_name, final_name) in glyph_order.names().zip(final_names) {
            let glyph = context.ir.get_glyph(glyph_name.clone());
            let instance = glyph.default_instance();
            if !instance.components.is_empty() {
                return Err(Error::CffGlyphHasComponents(glyph_name.clone()));
            }

            let mut pen = charstring::CharstringBuilder::new();
            let mut bounds = None;
            for contour in &instance.contours {
                // CFF keeps the source (counter-clockwise) contour direction,
                // so unlike glyf there is nothing to reverse here
                let rounded = drop_explicit_closing_lines(&rounded_path(contour));
                bounds = union(bounds, path_bounds(&rounded));
                pen.append_path(&rounded);
            }
            // fontTools floors mins and ceils maxes (intRect)
            let bounds = bounds.map(|r| {
                [
                    r.x0.floor() as i32,
                    r.y0.floor() as i32,
                    r.x1.ceil() as i32,
                    r.y1.ceil() as i32,
                ]
            });
            glyph_bounds.push(bounds);
            builder.add_glyph(GlyphData {
                name: final_name,
                advance_width: instance.width,
                charstring: pen.build(None, true),
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
