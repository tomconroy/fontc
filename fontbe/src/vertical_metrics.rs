//! Generates the [vmtx](https://learn.microsoft.com/en-us/typography/opentype/spec/vmtx),
//! [vhea](https://learn.microsoft.com/en-us/typography/opentype/spec/vhea) and
//! [VORG](https://learn.microsoft.com/en-us/typography/opentype/spec/vorg) tables.

use std::collections::HashMap;

use fontdrasil::orchestration::{Access, AccessBuilder, Work};
use fontir::orchestration::{Flags, WorkId as FeWorkId};
use log::trace;
use write_fonts::{
    OtRound, dump_table,
    tables::{
        vhea::Vhea,
        vmtx::Vmtx,
        vorg::{VertOriginYMetrics, Vorg},
    },
    types::{FWord, GlyphId16, Version16Dot16},
};

use crate::{
    cff::bbox_from_cff,
    error::Error,
    metrics_and_limits::MetricsBuilder,
    orchestration::{AnyWorkId, BeWork, Context, WorkId},
};

#[derive(Debug)]
struct VerticalMetricsWork {}

pub fn create_vertical_metrics_work() -> Box<BeWork> {
    Box::new(VerticalMetricsWork {})
}

impl Work<Context, AnyWorkId, Error> for VerticalMetricsWork {
    fn id(&self) -> AnyWorkId {
        WorkId::Vmtx.into()
    }

    fn read_access(&self) -> Access<AnyWorkId> {
        AccessBuilder::new()
            .variant(FeWorkId::GlobalMetrics)
            .variant(FeWorkId::GlyphOrder)
            .variant(FeWorkId::ALL_GLYPHS)
            .variant(WorkId::ALL_GLYF_FRAGMENTS)
            // We need composite bboxes to be calculated:
            .variant(WorkId::Glyf)
            // For CFF flavor builds bounds come from the CFF work instead;
            // whichever isn't scheduled is trivially fulfilled
            .variant(WorkId::Cff)
            .variant(WorkId::ExtraFeaTables)
            .build()
    }

    fn write_access(&self) -> Access<AnyWorkId> {
        AccessBuilder::new()
            .variant(WorkId::Vmtx)
            .variant(WorkId::Vhea)
            .variant(WorkId::Vorg)
            .build()
    }

    fn also_completes(&self) -> Vec<AnyWorkId> {
        vec![WorkId::Vhea.into(), WorkId::Vorg.into()]
    }

    /// Generate:
    ///
    /// * [vmtx](https://learn.microsoft.com/en-us/typography/opentype/spec/vmtx)
    /// * [vhea](https://learn.microsoft.com/en-us/typography/opentype/spec/vhea)
    /// * [VORG](https://learn.microsoft.com/en-us/typography/opentype/spec/vorg), for CFF builds
    fn exec(&self, context: &Context) -> Result<(), Error> {
        let static_metadata = context.ir.static_metadata.get();

        if !static_metadata.build_vertical {
            trace!("Skip vmtx and vhea; this is not a vertical font");
            return Ok(());
        }

        let glyph_order = context.ir.glyph_order.get();
        let default_metrics = context
            .ir
            .global_metrics
            .get()
            .at(static_metadata.default_location());

        // In a CFF build bounds come from the CFF work; there are no glyf fragments
        let cff = context
            .flags
            .contains(Flags::CFF_OUTLINES)
            .then(|| context.cff.get());

        // Collate vertical metrics
        let mut builder = MetricsBuilder::default();
        // VORG is a CFF-only table; ufo2ft only lists it in OutlineOTFCompiler.tables
        // https://github.com/googlefonts/ufo2ft/blob/2f11b0ff/Lib/ufo2ft/outlineCompiler.py#L1321
        let build_vorg = context.flags.contains(Flags::CFF_OUTLINES);
        let mut vertical_origins = Vec::new();
        for (gid, gn) in glyph_order.iter() {
            let glyph = context.ir.get_glyph(gn.clone());
            let instance = glyph.default_instance();

            // https://github.com/googlefonts/ufo2ft/blob/2f11b0ff/Lib/ufo2ft/outlineCompiler.py#L882-L890
            let advance = instance.height(&default_metrics);
            let vertical_origin = instance.vertical_origin(&default_metrics);
            if build_vorg {
                vertical_origins.push((gid, vertical_origin));
            }

            let glyf_bbox = || {
                context
                    .glyphs
                    .get(&WorkId::GlyfFragment(gn.clone()).into())
                    .data
                    .bbox()
            };
            let cff_bbox = |bounds: &[Option<[i32; 4]>]| {
                bounds
                    .get(gid.to_u16() as usize)
                    .copied()
                    .flatten()
                    .map(|bounds| bbox_from_cff(gn, bounds))
                    .transpose()
            };
            let bbox = match &cff {
                Some(cff) => cff_bbox(&cff.glyph_bounds)?,
                None => glyf_bbox(),
            };
            // like hhea: the side bearing comes from the rounded box, but the
            // height it spans comes from the outer one, because fontTools
            // recalculates vhea from the charstrings with `ceil(yMax)` and
            // `floor(yMin)`
            let outer_bbox = match &cff {
                Some(cff) => cff_bbox(&cff.glyph_outer_bounds)?,
                None => bbox,
            };
            let side_bearing = vertical_origin - bbox.map(|bbox| bbox.y_max).unwrap_or_default();
            let bounds_advance = outer_bbox.map(|bbox| bbox.y_max as i32 - bbox.y_min as i32);

            builder.update(advance, side_bearing, bounds_advance);
        }

        let metrics = builder.build();

        // Build and send vertical metrics tables out into the world
        let mut vhea = Vhea {
            version: Version16Dot16::VERSION_1_1,
            ascender: FWord::new(default_metrics.vhea_ascender.into_inner().ot_round()),
            descender: FWord::new(default_metrics.vhea_descender.into_inner().ot_round()),
            line_gap: FWord::new(default_metrics.vhea_line_gap.into_inner().ot_round()),
            advance_height_max: metrics.advance_max,
            min_top_side_bearing: metrics.min_first_side_bearing,
            min_bottom_side_bearing: metrics.min_second_side_bearing,
            y_max_extent: metrics.max_extent,
            caret_slope_rise: default_metrics
                .vhea_caret_slope_rise
                .into_inner()
                .ot_round(),
            caret_slope_run: default_metrics.vhea_caret_slope_run.into_inner().ot_round(),
            caret_offset: default_metrics.vhea_caret_offset.into_inner().ot_round(),
            number_of_long_ver_metrics: metrics.long_metrics.len().try_into().map_err(|_| {
                Error::OutOfBounds {
                    what: "number_of_long_metrics".into(),
                    value: format!("{}", metrics.long_metrics.len()),
                }
            })?,
        };
        // Apply FEA `table vhea { ... }` overrides.
        // FEA can only set VertTypoAscender, VertTypoDescender, VertTypoLineGap.
        if let Some(extra_tables) = context.extra_fea_tables.try_get()
            && let Some(fea_vhea) = extra_tables.vhea.as_ref()
        {
            vhea.ascender = fea_vhea.ascender;
            vhea.descender = fea_vhea.descender;
            vhea.line_gap = fea_vhea.line_gap;
        }
        context.vhea.set(vhea);

        let vmtx = Vmtx::new(metrics.long_metrics, metrics.first_side_bearings);
        let raw_vmtx = dump_table(&vmtx)
            .map_err(|e| Error::DumpTableError {
                e,
                context: "vmtx".into(),
            })?
            .into();
        context.vmtx.set(raw_vmtx);

        if build_vorg && let Some(vorg) = build_vorg_table(&vertical_origins) {
            context.vorg.set(vorg);
        }

        Ok(())
    }
}

/// Build VORG from the vertical origin of every glyph, in glyph order.
///
/// Port of ufo2ft's `setupTable_VORG`:
/// <https://github.com/googlefonts/ufo2ft/blob/2f11b0ff/Lib/ufo2ft/outlineCompiler.py#L912-L936>
///
/// The default is the most common vertical origin, ties broken in favour of the
/// first one seen (`Counter.most_common(1)` returns the first maximum). Records
/// are only written when more than one distinct origin exists, and only for the
/// glyphs that differ from the default; fontTools sorts them by glyph id
/// (`fontTools/ttLib/tables/V_O_R_G_.py`, `compile`), which is the order we
/// collect them in.
fn build_vorg_table(vertical_origins: &[(GlyphId16, i16)]) -> Option<Vorg> {
    if vertical_origins.is_empty() {
        return None;
    }
    // insertion-ordered so first-seen wins ties, like Python's Counter
    let mut counts: Vec<(i16, usize)> = Vec::new();
    let mut index: HashMap<i16, usize> = HashMap::new();
    for (_, origin) in vertical_origins {
        match index.get(origin) {
            Some(i) => counts[*i].1 += 1,
            None => {
                index.insert(*origin, counts.len());
                counts.push((*origin, 1));
            }
        }
    }
    // NOT max_by_key: that returns the *last* maximum, Python's max the first
    let default_vert_origin_y = counts
        .iter()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })
        .map(|(origin, _)| *origin)
        .unwrap();

    let metrics = if counts.len() > 1 {
        vertical_origins
            .iter()
            .filter(|(_, origin)| *origin != default_vert_origin_y)
            .map(|(gid, origin)| VertOriginYMetrics::new(*gid, *origin))
            .collect()
    } else {
        Vec::new()
    };
    Some(Vorg::new(default_vert_origin_y, metrics))
}
