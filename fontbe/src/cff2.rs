//! Generates a [CFF2](https://learn.microsoft.com/en-us/typography/opentype/spec/cff2) table.
//!
//! This is the variable counterpart of [`crate::cff`]: where a `CFF ` table
//! draws one master, a `CFF2` draws every master at once, with the values that
//! differ spelled out as a default plus one delta per variation region. Both
//! are produced by the same work (there is one PostScript outline table in a
//! font, never both), which picks by whether the font has axes.
//!
//! It is a port of `fontTools.varLib.cff`'s `merge_region_fonts`, so that
//! `fontc --flavor otf` on a variable source matches
//! `fontmake -o variable-cff2 --optimize-cff 1`. The charstring merging,
//! variation store and DICT encoding live in write-fonts
//! (`write_fonts::ps::cff::v2`); what is here is the bookkeeping varLib does
//! around them: which masters a glyph participates in, which variation store
//! index that participation earns, and what the Private DICT blends against.

use std::collections::{HashMap, HashSet};

use fontdrasil::{
    coords::NormalizedLocation,
    types::GlyphName,
    variations::{RoundingBehaviour, VariationModel, VariationRegion},
};
use fontir::ir::{PostscriptSettings, StaticMetadata};
use log::warn;
use ordered_float::OrderedFloat;
use write_fonts::{
    OtRound,
    ps::cff::{
        dict::Number,
        v1::PrivateDictValues,
        v2::{
            Blend, Cff2CharstringBuilder, Cff2FontBuilder, Cff2GlyphData, Cff2PrivateDictValues,
            Cff2TopDictValues, varstore::build_var_store,
        },
    },
};

use crate::{
    cff::{outer_bounds, postscript_outlines, private_dict_values},
    error::{Error, GlyphProblem},
    orchestration::{CffOutput, Context},
};

/// How charstring blend deltas are rounded.
///
/// `CFF2CharStringMergePen` rounds with a tolerance of 0.01
/// (`varLib/cff.py:469,482`), so a delta only snaps to an integer when it is
/// already within 0.01 of one; a genuinely fractional delta survives and is
/// written as a 16.16 fixed operand. This is neither `otRound` nor gvar's
/// all-integer rule.
const CHARSTRING_DELTA_ROUNDING: RoundingBehaviour =
    RoundingBehaviour::RoundTiesEvenWithin(OrderedFloat(0.01));

/// `cffLib`'s Private DICT defaults, which decide two things: which keys the
/// default master's DICT holds at all, and what a master that states nothing
/// contributes to a blend.
mod cff_default {
    pub(super) const BLUE_SCALE: f64 = 0.039625;
    pub(super) const BLUE_SHIFT: i32 = 7;
    pub(super) const BLUE_FUZZ: i32 = 1;
}

/// Build the `CFF2` table of a variable font.
pub(crate) fn build_cff2(context: &Context) -> Result<CffOutput, Error> {
    let static_metadata = context.ir.static_metadata.get();
    let glyph_order = context.ir.glyph_order.get();
    let axis_order = static_metadata.axes.axis_order();

    // varLib's masters are the designspace's sources, every one of them.
    // fontc's global model covers only the locations global metadata is
    // defined at, so a sparse source — a brace layer, a designspace layer
    // source — that only some glyphs use is missing from it; the glyphs are
    // where the rest are.
    let glyphs: Vec<_> = glyph_order
        .names()
        .map(|glyph_name| context.ir.get_glyph(glyph_name.clone()))
        .collect();
    let mut locations: HashSet<NormalizedLocation> = static_metadata
        .variation_model
        .locations()
        .cloned()
        .collect();
    for glyph in glyphs.iter() {
        for location in glyph.sources().keys() {
            let mut location = location.clone();
            location.fit_to_axes(&axis_order);
            locations.insert(location);
        }
    }

    // fontTools stops a region tent at the end of normalized space; fontc's
    // usual model stops it at the most extreme master it can see. The two
    // agree for the model over every master, but not for a sub-model over the
    // masters one sparse glyph has — and CFF2 writes those tents into its
    // variation store verbatim. So this path gets its own model.
    let global_model = VariationModel::new_full_axis_ranges(locations, axis_order.clone());
    let masters: Vec<NormalizedLocation> = global_model.locations().cloned().collect();
    let default_index = masters
        .iter()
        .position(|loc| *loc == global_model.default)
        .ok_or_else(|| Error::NoVariationModel(global_model.default.clone()))?;

    let mut models = SubModels::new(&global_model, &masters, axis_order.clone());
    let mut plan = VarStorePlan::default();

    // fontir synthesizes a .notdef when the source has none; like ufo2ft's
    // stub it draws the closing line of each box explicitly, and ufo2ft keeps
    // those, so we do too
    let notdef: GlyphName = ".notdef".into();
    let synthesized_notdef = !context.ir.preliminary_glyph_order.get().contains(&notdef);

    let mut glyph_bounds = Vec::with_capacity(glyph_order.len());
    let mut glyph_outer_bounds = Vec::with_capacity(glyph_order.len());
    let mut glyph_data = Vec::with_capacity(glyph_order.len());
    #[allow(clippy::indexing_slicing)] // default_index is an index into masters
    let default_location = &masters[default_index];
    for (glyph_name, glyph) in glyph_order.names().zip(&glyphs) {
        let is_synthesized_notdef = synthesized_notdef && *glyph_name == notdef;
        let outlines = postscript_outlines(glyph, is_synthesized_notdef)?;
        // glyph sources are keyed by the source's own location, which may not
        // name every axis; the model's are fitted
        let mut outlines: HashMap<_, _> = outlines
            .into_iter()
            .map(|(mut loc, path)| {
                loc.fit_to_axes(&axis_order);
                (loc, path)
            })
            .collect();
        let Some(default_outline) = outlines.get(default_location).cloned() else {
            return Err(Error::GlyphError(
                glyph_name.clone(),
                GlyphProblem::MissingDefault,
            ));
        };
        if is_synthesized_notdef {
            // ufo2ft compiles a .notdef into *every* master, a sparse layer
            // source included, so a synthesized one takes part in masters
            // where no other glyph does — which, since it is glyph 0, is what
            // decides the whole font's variation store index 0. A layer
            // source draws its stub from the font info of the UFO it shares
            // with its parent master, which fontc does not track; the default
            // master's stub is the closest thing we have.
            for location in masters.iter() {
                outlines
                    .entry(location.clone())
                    .or_insert_with(|| default_outline.clone());
            }
        }

        // Which masters this glyph varies across. A region master in which
        // the glyph is missing *or draws nothing* drops out of the model:
        // `_get_cs(.., filterEmpty=True)` (`varLib/cff.py:272-288`) cannot
        // tell a blank glyph from an absent one, and neither do we. The
        // default master is always in, blank or not.
        let mask: Vec<bool> = masters
            .iter()
            .enumerate()
            .map(|(index, location)| {
                index == default_index
                    || outlines
                        .get(location)
                        .is_some_and(|path| !path.elements().is_empty())
            })
            .collect();
        let participants = mask.iter().filter(|drawn| **drawn).count();

        let model = models.sub_model(&mask);
        let mut pen = Cff2CharstringBuilder::new(participants);
        for (index, location) in model.locations().enumerate() {
            let path = outlines.get(location).ok_or_else(|| {
                Error::GlyphError(glyph_name.clone(), GlyphProblem::MissingDefault)
            })?;
            pen.master(index).append_path(path);
        }
        let charstring =
            pen.build(|values| model.deltas_for_masters(values, CHARSTRING_DELTA_ROUNDING))?;

        // A glyph that only the default master draws has nothing to blend; a
        // glyph that does not draw, or whose masters all agree, blends
        // nothing. None of them claim a variation store index
        // (`varLib/cff.py:345-353`).
        let charstring = if participants > 1 && charstring.blends && charstring.marking {
            let vsindex = plan.vsindex_for(&mask, model);
            charstring.with_vsindex(vsindex)?
        } else {
            charstring
        };

        // ufo2ft measures the charstring it just built, rounds each side, and
        // treats an all-zero box as no box at all; side bearings come from
        // there. The default master's outline is what the default instance's
        // metrics are measured from.
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
        glyph_data.push(Cff2GlyphData {
            charstring: charstring.bytes,
            bounds: charstring.bounds,
        });
    }

    // Even a font in which nothing blends gets one variation store index, so
    // that the Private DICT has a model to blend against
    // (`varLib/cff.py:369-376`).
    if plan.subtables.is_empty() {
        let mask = vec![true; masters.len()];
        let model = models.sub_model(&mask);
        plan.vsindex_for(&mask, model);
    }

    // The Private DICT blends against variation store index 0's model —
    // whatever glyph happened to claim it first — so a master that model
    // excludes has its hints silently ignored (`varLib/cff.py:117-136`).
    #[allow(clippy::indexing_slicing)] // just ensured there is a vsindex 0
    let vsindex_zero = plan.masks[0].clone();
    let model = models.sub_model(&vsindex_zero);
    let blending_masters: Vec<&NormalizedLocation> = model.locations().collect();
    let private = private_dict(&static_metadata, &blending_masters, model);

    let upem = static_metadata.units_per_em as f64;
    let mut builder = Cff2FontBuilder::new(
        Cff2TopDictValues {
            font_matrix: Some([1.0 / upem, 0.0, 0.0, 1.0 / upem, 0.0, 0.0]),
            // fontmake never writes maxstack; readers assume the default
            max_stack: None,
        },
        private,
    );
    for glyph in glyph_data {
        builder.add_glyph(glyph);
    }
    let regions = plan
        .regions
        .iter()
        .map(|region| region.to_write_fonts_variation_region(&static_metadata.axes))
        .collect();
    builder.set_var_store(build_var_store(
        static_metadata.axes.len() as u16,
        regions,
        &plan.subtables,
    )?);

    Ok(CffOutput {
        table: builder.build()?.as_bytes().to_vec(),
        cff2: true,
        glyph_bounds,
        glyph_outer_bounds,
    })
}

/// The variation store a CFF2 needs, built the way `merge_charstrings` builds
/// it: an index per distinct set of participating masters, numbered by first
/// encounter in glyph order.
#[derive(Debug, Default)]
struct VarStorePlan {
    /// `masterSupports`: the region list, in the order regions are first
    /// seen across all the sub-models, deduplicated by equality.
    regions: Vec<VariationRegion>,
    /// `varDataList`: one region-index list per variation store index, each
    /// in its own sub-model's order rather than sorted.
    subtables: Vec<Vec<u16>>,
    by_mask: HashMap<Vec<bool>, u16>,
    /// The participation mask each index was allocated for, so the Private
    /// DICT can find index 0's model again.
    masks: Vec<Vec<bool>>,
}

impl VarStorePlan {
    fn vsindex_for(&mut self, mask: &[bool], model: &VariationModel) -> u16 {
        if let Some(vsindex) = self.by_mask.get(mask) {
            return *vsindex;
        }
        let mut indices = Vec::with_capacity(model.regions().len().saturating_sub(1));
        for support in model.regions().iter().skip(1) {
            let index = match self.regions.iter().position(|known| known == support) {
                Some(index) => index,
                None => {
                    self.regions.push(support.clone());
                    self.regions.len() - 1
                }
            };
            indices.push(index as u16);
        }
        let vsindex = self.subtables.len() as u16;
        self.subtables.push(indices);
        self.by_mask.insert(mask.to_vec(), vsindex);
        self.masks.push(mask.to_vec());
        vsindex
    }
}

/// The variation models of the master subsets a font's glyphs use.
///
/// fontTools caches these on the parent model and keys them the same way we
/// do, on which masters take part (`VariationModel.getSubModel`,
/// `models.py:305-324`). A glyph present in every master gets the parent
/// model itself.
struct SubModels<'a> {
    global: &'a VariationModel,
    masters: &'a [NormalizedLocation],
    axis_order: Vec<write_fonts::types::Tag>,
    cache: HashMap<Vec<bool>, VariationModel>,
}

impl<'a> SubModels<'a> {
    fn new(
        global: &'a VariationModel,
        masters: &'a [NormalizedLocation],
        axis_order: Vec<write_fonts::types::Tag>,
    ) -> Self {
        SubModels {
            global,
            masters,
            axis_order,
            cache: HashMap::new(),
        }
    }

    fn sub_model(&mut self, mask: &[bool]) -> &VariationModel {
        if !self.cache.contains_key(mask) {
            let model = if mask.iter().all(|taking_part| *taking_part) {
                self.global.clone()
            } else {
                let locations = self
                    .masters
                    .iter()
                    .zip(mask)
                    .filter(|(_, taking_part)| **taking_part)
                    .map(|(location, _)| location.clone())
                    .collect();
                VariationModel::new_full_axis_ranges(locations, self.axis_order.clone())
            };
            self.cache.insert(mask.to_vec(), model);
        }
        #[allow(clippy::unwrap_used)] // just inserted
        self.cache.get(mask).unwrap()
    }
}

/// The Private DICT values of one master, as `cffLib` holds them.
///
/// A region master's DICT is the one ufo2ft compiled from that master's own
/// font info, and `cffLib` fills in its defaults, so `BlueScale`,
/// `BlueShift` and `BlueFuzz` are always there even for a master that states
/// no hints at all. The keys with no default — the arrays, `StdHW` and
/// `StdVW` — are there only if the master stated them.
struct MasterPrivate {
    ufo2ft: PrivateDictValues,
    /// Whether this is the default master, whose DICT has already been
    /// through the CFF encoder; see [`MasterPrivate::blue_scale`].
    is_default: bool,
}

impl MasterPrivate {
    fn new(postscript: &PostscriptSettings, is_default: bool) -> Self {
        MasterPrivate {
            ufo2ft: private_dict_values(postscript),
            is_default,
        }
    }

    /// This master's `BlueScale`.
    ///
    /// The default master's has been through the CFF real encoder and back —
    /// `_add_CFF2` upgrades that master's `CFF ` table in place, and
    /// `convertCFFToCFF2` re-reads it — so it keeps only eight significant
    /// digits where every other master's is the raw value ufo2ft computed.
    /// On Cormorant, whose masters state no `postscriptBlueScale` and so both
    /// get ufo2ft's 3/(4×19) fallback, that difference *alone* is what makes
    /// `BlueScale` blend at all: `0.039473684` against
    /// `0.039473684210526314`.
    fn blue_scale(&self) -> f64 {
        let value = self.ufo2ft.blue_scale.unwrap_or(cff_default::BLUE_SCALE);
        if self.is_default {
            write_fonts::ps::cff::dict::round_trip_real(value)
        } else {
            value
        }
    }

    fn blue_shift(&self) -> i32 {
        self.ufo2ft.blue_shift.unwrap_or(cff_default::BLUE_SHIFT)
    }

    fn blue_fuzz(&self) -> i32 {
        self.ufo2ft.blue_fuzz.unwrap_or(cff_default::BLUE_FUZZ)
    }
}

/// The blended Private DICT.
///
/// Only the keys the *default* master's DICT holds are blended at all, and it
/// holds a key only when ufo2ft wrote it and it differs from the CFF default
/// — which is why a family whose default master has the default `BlueShift`
/// of 7 has no `BlueShift` in its CFF2 whatever the other masters say
/// (`varLib/cff.py:138-151` over a DICT `cffLib` built without defaults).
fn private_dict(
    static_metadata: &StaticMetadata,
    blending_masters: &[&NormalizedLocation],
    model: &VariationModel,
) -> Cff2PrivateDictValues {
    let axis_order = static_metadata.axes.axis_order();
    let postscript: HashMap<NormalizedLocation, &PostscriptSettings> = static_metadata
        .postscript
        .iter()
        .map(|(location, settings)| {
            let mut location = location.clone();
            location.fit_to_axes(&axis_order);
            (location, settings)
        })
        .collect();
    // A designspace layer source (a Glyphs brace layer) has no font info of
    // its own: ufo2ft compiles its Private DICT from the UFO it shares with
    // its parent master. fontc does not track which master that is, so the
    // default master's hints stand in — right whenever the sparse sources
    // hang off the default master, which is the common shape.
    let fallback = static_metadata.postscript_default();
    let masters: Vec<MasterPrivate> = blending_masters
        .iter()
        .enumerate()
        .map(|(index, location)| {
            MasterPrivate::new(
                postscript.get(*location).copied().unwrap_or(&fallback),
                index == 0,
            )
        })
        .collect();
    #[allow(clippy::indexing_slicing)] // there is always a default master
    let default = &masters[0];

    let mut private = Cff2PrivateDictValues::default();
    let list = |key: &str, get: fn(&MasterPrivate) -> &Vec<i32>| {
        blend_list(key, get(default), masters.iter().map(get), model)
    };
    private.blue_values = list("BlueValues", |m| &m.ufo2ft.blue_values);
    private.other_blues = list("OtherBlues", |m| &m.ufo2ft.other_blues);
    private.family_blues = list("FamilyBlues", |m| &m.ufo2ft.family_blues);
    private.family_other_blues = list("FamilyOtherBlues", |m| &m.ufo2ft.family_other_blues);
    private.stem_snap_h = list("StemSnapH", |m| &m.ufo2ft.stem_snap_h);
    private.stem_snap_v = list("StemSnapV", |m| &m.ufo2ft.stem_snap_v);

    // The scalars with a CFF default are blended only when the default master
    // states something else; the ones without are blended whenever the
    // default master states them at all.
    if default
        .ufo2ft
        .blue_scale
        .is_some_and(|_| default.blue_scale() != cff_default::BLUE_SCALE)
    {
        private.blue_scale = Some(blend_scalar(
            masters.iter().map(|m| m.blue_scale()).collect(),
            model,
        ));
    }
    if default
        .ufo2ft
        .blue_shift
        .is_some_and(|value| value != cff_default::BLUE_SHIFT)
    {
        private.blue_shift = Some(blend_scalar(
            masters.iter().map(|m| m.blue_shift() as f64).collect(),
            model,
        ));
    }
    if default
        .ufo2ft
        .blue_fuzz
        .is_some_and(|value| value != cff_default::BLUE_FUZZ)
    {
        private.blue_fuzz = Some(blend_scalar(
            masters.iter().map(|m| m.blue_fuzz() as f64).collect(),
            model,
        ));
    }
    private.std_hw = blend_optional_scalar("StdHW", masters.iter().map(|m| m.ufo2ft.std_hw), model);
    private.std_vw = blend_optional_scalar("StdVW", masters.iter().map(|m| m.ufo2ft.std_vw), model);
    private
}

/// One blended array entry per row, the way `merge_PrivateDicts` builds them.
///
/// Rows are converted to values relative to the previous row before the
/// deltas are taken, but the stored default is the row's *absolute* value —
/// so a row reads `[absolute default, delta of relative value, …]`. If any
/// row of any master varies, every row is emitted as a blend, all-zero deltas
/// included; if none does, the whole key collapses back to a plain array.
fn blend_list<'a>(
    key: &str,
    default: &[i32],
    masters: impl Iterator<Item = &'a Vec<i32>>,
    model: &VariationModel,
) -> Vec<Blend> {
    if default.is_empty() {
        return Vec::new();
    }
    let masters: Vec<&Vec<i32>> = masters.collect();
    // A key the default master has and a region master does not is discarded,
    // leaving the default master's own array unblended
    // (`varLib/cff.py:146-152`).
    if masters.iter().any(|values| values.is_empty()) {
        warn!(
            "{key} is in the default master's Private DICT but not every other master's; it will not blend"
        );
        return default.iter().map(|value| Blend::scalar(*value)).collect();
    }
    // `zip(*values)` stops at the shortest master, silently dropping the
    // trailing zones of any master that states more (`varLib/cff.py:153`).
    #[allow(clippy::unwrap_used)] // masters is non-empty
    let rows = masters.iter().map(|values| values.len()).min().unwrap();

    let mut previous = vec![0.0f64; masters.len()];
    let mut any_row_varies = false;
    let mut blends: Vec<(f64, Vec<f64>)> = Vec::with_capacity(rows);
    for row in 0..rows {
        #[allow(clippy::indexing_slicing)] // row < the shortest master's len
        let values: Vec<f64> = masters.iter().map(|values| values[row] as f64).collect();
        let relative: Vec<f64> = values
            .iter()
            .zip(&previous)
            .map(|(value, previous)| value - previous)
            .collect();
        any_row_varies |= !all_equal(&relative);
        previous = values.clone();
        let mut deltas = model.deltas_for_masters(&relative, RoundingBehaviour::None);
        #[allow(clippy::indexing_slicing)] // one delta per master, always >= 1
        {
            deltas[0] = values[0];
        }
        blends.push((deltas.remove(0), deltas));
    }
    blends
        .into_iter()
        .map(|(default, deltas)| {
            if any_row_varies {
                Blend::new(
                    Number::conv_to_int(default),
                    deltas.into_iter().map(Number::conv_to_int).collect(),
                )
            } else {
                Blend::scalar(Number::conv_to_int(default))
            }
        })
        .collect()
}

/// A blended scalar: the masters' values, or just the default's when they all
/// agree. Deltas here are *not* rounded — `merge_PrivateDicts` calls
/// `getDeltas` with no `round` argument (`varLib/cff.py:181,190`), so
/// floating point noise survives into the DICT.
fn blend_scalar(values: Vec<f64>, model: &VariationModel) -> Blend {
    #[allow(clippy::indexing_slicing)] // there is always a default master
    if all_equal(&values) {
        return Blend::scalar(Number::conv_to_int(values[0]));
    }
    let mut deltas = model.deltas_for_masters(&values, RoundingBehaviour::None);
    Blend::new(
        Number::conv_to_int(deltas.remove(0)),
        deltas.into_iter().map(Number::conv_to_int).collect(),
    )
}

/// [`blend_scalar`] for `StdHW`/`StdVW`, which have no CFF default and so can
/// genuinely be missing from a master.
fn blend_optional_scalar(
    key: &str,
    values: impl Iterator<Item = Option<i32>>,
    model: &VariationModel,
) -> Option<Blend> {
    let values: Vec<Option<i32>> = values.collect();
    let default = (*values.first()?)?;
    if values.iter().any(|value| value.is_none()) {
        // fontTools raises a KeyError here and the build dies; keeping the
        // default master's value costs no parity (there is no oracle output
        // to differ from) and produces a usable font
        warn!(
            "{key} is in the default master's Private DICT but not every other master's; it will not blend"
        );
        return Some(Blend::scalar(default));
    }
    Some(blend_scalar(
        values.into_iter().flatten().map(|v| v as f64).collect(),
        model,
    ))
}

fn all_equal(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[0] == pair[1])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)] // test code

    use std::collections::HashSet;

    use write_fonts::types::Tag;

    use super::*;

    fn wght(value: f64) -> NormalizedLocation {
        NormalizedLocation::for_pos(&[("wght", value)])
    }

    /// A model over the given `wght` positions, built the way the CFF2 work
    /// builds one.
    fn model(positions: &[f64]) -> VariationModel {
        VariationModel::new_full_axis_ranges(
            positions.iter().copied().map(wght).collect::<HashSet<_>>(),
            vec![Tag::new(b"wght")],
        )
    }

    fn blend(default: i32, deltas: &[i32]) -> Blend {
        Blend::new(default, deltas.iter().map(|d| Number::Int(*d)).collect())
    }

    fn list(masters: &[&[i32]], model: &VariationModel) -> Vec<Blend> {
        let masters: Vec<Vec<i32>> = masters.iter().map(|m| m.to_vec()).collect();
        blend_list("BlueValues", &masters[0], masters.iter(), model)
    }

    /// The Probe fixture's blue zones, whose masters state *different numbers*
    /// of them: `zip(*values)` stops at the shortest, so the Bold master's
    /// last two zones vanish without a word.
    ///
    /// Each row stores the default master's absolute value but a delta of the
    /// row's value *relative to the previous row*, which is why the second row
    /// blends `[0, 4]` and not `[0, 0]`. Asserted against fontmake's Private
    /// DICT for `fx/Probe.designspace`.
    #[test]
    fn blue_values_truncate_to_the_shortest_master() {
        let blends = list(
            &[
                &[-10, 0, 500, 512, 700, 712],
                &[-14, 0, 520, 534, 720, 731, 800, 810],
            ],
            &model(&[0.0, 1.0]),
        );
        assert_eq!(
            blends,
            vec![
                blend(-10, &[-4]),
                blend(0, &[4]),
                blend(500, &[20]),
                blend(512, &[2]),
                blend(700, &[-2]),
                blend(712, &[-1]),
            ]
        );
    }

    /// `any_points_differ` is decided per key, not per row: one varying row
    /// makes every row a blend, all-zero deltas included. Probe's OtherBlues
    /// are `[-12, 0]` and `[-15, -3]`, and the second row's relative values
    /// are the same in both masters.
    #[test]
    fn one_varying_row_blends_the_whole_key() {
        assert_eq!(
            list(&[&[-12, 0], &[-15, -3]], &model(&[0.0, 1.0])),
            vec![blend(-12, &[-3]), blend(0, &[0])]
        );
    }

    /// ... and a key no master varies collapses back to a plain array.
    #[test]
    fn an_unvarying_key_is_not_a_blend() {
        let blends = list(&[&[-12, 0], &[-12, 0]], &model(&[0.0, 1.0]));
        assert_eq!(
            blends,
            vec![Blend::scalar(-12), Blend::scalar(0)],
            "no deltas at all"
        );
        assert!(blends.iter().all(|value| !value.varies()));
    }

    /// A key the default master has and a region master does not is thrown
    /// away, leaving the default master's own array unblended — this is how
    /// `glyphs3/WghtVar.glyphs`, whose Bold states no zones, keeps Regular's.
    #[test]
    fn a_key_missing_from_a_region_master_does_not_blend() {
        assert_eq!(
            list(&[&[-16, 0, 737, 753], &[]], &model(&[0.0, 1.0])),
            vec![
                Blend::scalar(-16),
                Blend::scalar(0),
                Blend::scalar(737),
                Blend::scalar(753)
            ]
        );
    }

    /// Scalars blend as plain deltas, unrounded, and collapse when the
    /// masters agree. The three-master case checks that the deltas come back
    /// in the model's own order.
    #[test]
    fn scalars_blend_unrounded() {
        let three = model(&[0.0, 0.4, 1.0]);
        assert_eq!(
            blend_scalar(vec![20.0, 24.0, 30.0], &three),
            blend(20, &[4, 10])
        );
        assert_eq!(
            blend_scalar(vec![7.0, 7.0, 7.0], &three),
            Blend::scalar(7),
            "identical masters do not blend"
        );
        // Cormorant's BlueScale: the default master's value has been through
        // the CFF encoder and the region master's has not, and the difference
        // survives as a delta because it is not exactly zero
        let blended = blend_scalar(vec![0.039473684, 0.039473684210526314], &model(&[0.0, 1.0]));
        assert_eq!(
            blended,
            Blend::new(
                Number::Real(0.039473684),
                vec![Number::Real(2.1052631166140756e-10)]
            )
        );
    }

    /// `StdHW` and `StdVW` have no CFF default, so a master really can be
    /// missing one. fontTools raises a `KeyError` and the build dies; we warn
    /// and keep the default master's value.
    #[test]
    fn a_scalar_missing_from_a_region_master_keeps_the_default() {
        let two = model(&[0.0, 1.0]);
        assert_eq!(
            blend_optional_scalar("StdHW", [Some(20), None].into_iter(), &two),
            Some(Blend::scalar(20))
        );
        assert_eq!(
            blend_optional_scalar("StdHW", [None, Some(20)].into_iter(), &two),
            None,
            "and a key the default master lacks is not written at all"
        );
    }

    /// A variation store index is minted per distinct participation set, in
    /// the order the sets are first seen, and the region list accumulates
    /// across sub-models without reordering or deduplicating subtables.
    ///
    /// This is the Vf3 fixture's shape: a sparse glyph comes first in glyph
    /// order, so *its* two-master model is index 0 and the full model is
    /// index 1.
    #[test]
    fn vsindexes_are_minted_in_first_seen_order() {
        let full = model(&[0.0, 0.4, 1.0]);
        let thin_bold = model(&[0.0, 1.0]);
        let thin_mid = model(&[0.0, 0.4]);
        let mut plan = VarStorePlan::default();

        assert_eq!(plan.vsindex_for(&[true, false, true], &thin_bold), 0);
        assert_eq!(plan.vsindex_for(&[true, true, true], &full), 1);
        assert_eq!(plan.vsindex_for(&[true, true, false], &thin_mid), 2);
        // the same set again is the same index
        assert_eq!(plan.vsindex_for(&[true, true, true], &full), 1);

        let tents: Vec<_> = plan
            .regions
            .iter()
            .map(|region| {
                let tent = region.get(&Tag::new(b"wght")).unwrap();
                (
                    tent.min.into_inner().into_inner(),
                    tent.peak.into_inner().into_inner(),
                    tent.max.into_inner().into_inner(),
                )
            })
            .collect();
        assert_eq!(
            tents,
            vec![(0.0, 1.0, 1.0), (0.0, 0.4, 1.0), (0.4, 1.0, 1.0)],
            "the sparse glyph's region is region 0"
        );
        assert_eq!(plan.subtables, vec![vec![0], vec![1, 2], vec![1]]);
    }
}
