//! Pinning variable IR at a single location.
//!
//! These are the primitives an instancer needs: given IR that varies across a
//! design space, produce the equivalent IR defined at exactly one location, so
//! that the ordinary static backends compile a static instance. Nothing here is
//! wired into orchestration; each function is pure.
//!
//! # Everything stays unrounded
//!
//! `fontmake -i` does not round: `--round-instances` is opt-in and off by
//! default, so the interpolated instance carries float coordinates, advances,
//! kerning and metrics into the compiler and rounding happens exactly once, at
//! table-build time (`otRound` in glyf/hmtx/CFF). Rounding here would not only
//! shift values, it would feed `cu2qu` different inputs and change point
//! *counts*. Nothing below rounds.
//!
//! # Everything combines masters, not deltas
//!
//! fontmake's instancer interpolates with [master scalars]: it asks the model
//! for one multiplier per master and computes `sum(scalar_i * master_i)`,
//! skipping masters whose scalar is zero. fontc's usual delta path
//! (`deltas` + `interpolate_from_deltas`) is mathematically the same number but
//! not the same f64, and — more importantly — the zero-scalar skip is
//! *semantic*, not an optimisation:
//!
//! - a master that does not participate at the pin contributes no kerning
//!   *keys*, so which pairs the instance even has depends on it
//!   ([`pin_kerning`]);
//! - a pin that lands exactly on a master has exactly one contributing term,
//!   so fontMath's binary rules (drop-on-length-mismatch, absent-term identity)
//!   never fire and that master's values come through verbatim
//!   ([`pin_postscript`]).
//!
//! [master scalars]: VariationModel::master_scalars
//!
//! # Location keys
//!
//! The pinned IR is keyed by [`StaticMetadata::default_location`] **as-is** —
//! the all-source-axes-at-zero location, including point axes. That is the
//! shape a `.designspace` with one point axis, or a single-master Glyphs v2
//! file, already produces, and the backend looks glyph sources up by exactly
//! that key (see `fontbe`'s `glyphs` work). Keying by the empty location
//! instead would require rewriting `all_source_axes` and the private
//! `default_location` field in lockstep, and getting half of that right breaks
//! every glyph in the font.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use fontdrasil::{
    coords::{NormalizedLocation, UserCoord, UserLocation},
    types::{Axes, GlyphName},
    variations::{DeltaError, VariationModel},
};
use kurbo::Point;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use write_fonts::types::{InvalidTag, Tag};

use crate::{
    error::{BadGlyph, Error},
    ir::{
        Anchor, GlobalMetrics, GlobalMetricsBuilder, Glyph, GlyphAnchors, GlyphInstance, KernGroup,
        KernPair, KernSide, KerningInstance, PostscriptSettings, StaticMetadata,
    },
};

/// Which instance of a variable source to build, as the user asked for it.
///
/// Deliberately un-resolved: telling the two forms apart needs no knowledge of
/// the source, and everything that *does* — which axes exist, what they range
/// over, what the named instances are called — only becomes knowable once the
/// frontend has produced [`StaticMetadata`]. So parsing gets this far and no
/// further; the pin resolves it.
///
/// This lives in fontir rather than beside the CLI so that
/// [`Context::instance`](crate::orchestration::Context) can carry it without
/// fontir knowing anything about the compiler's arguments.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum InstanceSpec {
    /// An explicit position, in **user** space, on some subset of the axes.
    ///
    /// Axes the user didn't mention take their default.
    Location(UserLocation),
    /// The style name of one of the source's named instances.
    Named(String),
}

/// What was wrong with an `--instance` argument, on its face.
///
/// Only syntax: a spec that parses can still name an axis the source doesn't
/// have, or a position outside that axis's range, and those are the pin's to
/// report.
#[derive(Debug, thiserror::Error)]
pub enum InstanceSpecError {
    #[error("an instance must be a style name or an axis position, not empty")]
    Empty,
    #[error("'{0}' is not 'axis=position'")]
    NotAnAxisPosition(String),
    #[error("'{raw}' is not an axis tag: {cause}")]
    InvalidTag { raw: String, cause: InvalidTag },
    #[error("'{value}' is not a position on axis '{tag}'")]
    InvalidPosition { tag: Tag, value: String },
    #[error("axis '{0}' is given more than once")]
    DuplicateAxis(Tag),
}

impl FromStr for InstanceSpec {
    type Err = InstanceSpecError;

    /// `wght=700,wdth=87.5` is a location; anything else is a style name.
    ///
    /// An `=` is what makes it a location: no style name has one, and a
    /// location cannot lack one. That is the whole discrimination — a name is
    /// otherwise unconstrained, so we cannot be pickier without rejecting
    /// legitimate names.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(InstanceSpecError::Empty);
        }
        if !s.contains('=') {
            return Ok(InstanceSpec::Named(s.to_string()));
        }

        let mut coords: Vec<(Tag, UserCoord)> = Vec::new();
        for term in s.split(',') {
            let term = term.trim();
            let Some((raw_tag, raw_pos)) = term.split_once('=') else {
                return Err(InstanceSpecError::NotAnAxisPosition(term.to_string()));
            };
            let (raw_tag, raw_pos) = (raw_tag.trim(), raw_pos.trim());
            let tag = Tag::from_str(raw_tag).map_err(|cause| InstanceSpecError::InvalidTag {
                raw: raw_tag.to_string(),
                cause,
            })?;
            let pos = raw_pos
                .parse::<f64>()
                .ok()
                .filter(|pos| pos.is_finite())
                .ok_or_else(|| InstanceSpecError::InvalidPosition {
                    tag,
                    value: raw_pos.to_string(),
                })?;
            if coords.iter().any(|(seen, _)| *seen == tag) {
                return Err(InstanceSpecError::DuplicateAxis(tag));
            }
            coords.push((tag, UserCoord::new(pos)));
        }
        Ok(InstanceSpec::Location(coords.into()))
    }
}

/// The model to interpolate `glyph` with.
///
/// The global model when the glyph is defined at exactly the same locations,
/// otherwise a model of just this glyph's locations, so sparse masters
/// interpolate from the masters they actually have.
pub(crate) fn variation_model_for_glyph<'a>(
    static_metadata: &'a StaticMetadata,
    glyph: &Glyph,
) -> Cow<'a, VariationModel> {
    if static_metadata
        .variation_model
        .locations()
        .all(|loc| glyph.sources().contains_key(loc))
        && static_metadata.variation_model.num_locations() == glyph.sources().len()
    {
        // great, we have the same model
        return Cow::Borrowed(&static_metadata.variation_model);
    }

    // otherwise we need a special model for this glyph.
    // This code is duplicated in various places (hvar, e.g.)
    // and maybe we can share it? or cache these models more globally?
    Cow::Owned(VariationModel::new(
        glyph.sources().keys().cloned().collect(),
        static_metadata.axes.iter().map(|ax| ax.tag).collect(),
    ))
}

/// Interpolate a new instance of `glyph` at `loc`.
///
/// The result takes its *structure* — contour element types, component names
/// and order — from the default instance, and its numbers from interpolating
/// every master. Unrounded, and combined from the masters themselves, see the
/// module docs.
///
/// Rounding here would be a significant problem for component transforms,
/// whose fractional bits matter, and would feed `cu2qu` different inputs and so
/// change point *counts*. ufo2ft does not round either, see
/// <https://github.com/googlefonts/ufo2ft/blob/01d3faee/Lib/ufo2ft/_compilers/baseCompiler.py#L266>.
pub fn interpolate_glyph_instance(
    static_metadata: &StaticMetadata,
    glyph: &Glyph,
    loc: &NormalizedLocation,
) -> Result<GlyphInstance, BadGlyph> {
    log::debug!("instantiating '{}' at {loc:?}", glyph.name);
    let model = variation_model_for_glyph(static_metadata, glyph);
    let point_seqs = glyph
        .sources()
        .iter()
        .map(|(loc, instance)| (loc.clone(), instance.values_for_interpolation()))
        .collect();
    let points = model
        .interpolate_from_masters(loc, &point_seqs)
        .map_err(|e| BadGlyph::new(&glyph.name, e))?;
    Ok(glyph
        .default_instance()
        .new_with_interpolated_values(&points))
}

/// `glyph` reduced to its interpolated value at `pin`.
///
/// The result has exactly one source, at [`StaticMetadata::default_location`],
/// which is what a genuinely static source produces.
pub fn pin_glyph(
    static_metadata: &StaticMetadata,
    glyph: &Glyph,
    pin: &NormalizedLocation,
) -> Result<Glyph, BadGlyph> {
    let instance = interpolate_glyph_instance(static_metadata, glyph, pin)?;
    Glyph::new(
        glyph.name.clone(),
        glyph.emit_to_binary,
        glyph.codepoints.clone(),
        HashMap::from([(static_metadata.default_location().clone(), instance)]),
    )
}

/// `anchors` reduced to their interpolated positions at `pin`.
///
/// Each anchor interpolates from its own locations, like anchor propagation
/// does, so an anchor defined at only some masters still resolves.
pub fn pin_anchors(
    static_metadata: &StaticMetadata,
    anchors: &GlyphAnchors,
    pin: &NormalizedLocation,
) -> Result<GlyphAnchors, BadGlyph> {
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    let key = static_metadata.default_location();

    let pinned = anchors
        .anchors
        .iter()
        .map(|anchor| {
            let model = VariationModel::new(
                anchor.positions.keys().cloned().collect(),
                axis_order.clone(),
            );
            let point_seqs: HashMap<_, _> = anchor
                .positions
                .iter()
                .map(|(loc, pos)| (loc.clone(), vec![*pos]))
                .collect();
            let pos = model
                .interpolate_from_masters(pin, &point_seqs)
                .map_err(|e| BadGlyph::new(anchors.glyph_name.clone(), e))?
                .first()
                .map(|v: &kurbo::Vec2| v.to_point())
                .unwrap_or(Point::ZERO);
            Ok(Anchor {
                kind: anchor.kind.clone(),
                original_name: anchor.original_name.clone(),
                positions: HashMap::from([(key.clone(), pos)]),
            })
        })
        .collect::<Result<Vec<_>, BadGlyph>>()?;

    Ok(GlyphAnchors::new(anchors.glyph_name.clone(), pinned))
}

/// Every source's kerning reduced to one [`KerningInstance`] at `pin`.
///
/// Pass the instances at the locations `KerningLocations` lists; the result is
/// keyed at the pin so that listing only the pin there is enough to make the
/// backend ignore whatever else is still lying around. Layer-only (sparse)
/// sources contribute no kerning at all in ufo2ft, so don't pass them.
///
/// This is `fontMath`'s `MathKerning`, which is neither "interpolate the raw
/// plist" nor "interpolate the resolved kerning" but a specific mixture of the
/// two:
///
/// - **Groups** are the default master's, and *only* the default master's.
///   ufo2ft builds every master's `MathKerning` with the default source's
///   groups precisely so that kerning math never has to union group
///   definitions that disagree; a master whose groups differ gets a warning and
///   is otherwise ignored.
/// - **Which pairs exist** is the union of the *literal* keys of the masters
///   that actually contribute — i.e. those whose
///   [master scalar](VariationModel::master_scalars) is non-zero. A master the
///   pin does not reach contributes no keys, so the instance's pair list
///   genuinely depends on where the pin is.
/// - **What each master contributes** for a pair is *resolved*, through
///   `exact -> (group1, glyph2) -> (glyph1, group2) -> (group1, group2) -> 0`
///   against the default master's groups. A pair one master omits is therefore
///   whatever a group pair covering it says, and 0 only when nothing covers it.
/// - **Zero pairs are then dropped**, unless either side is a plain glyph that
///   belongs to a kern group — that is `MathKerning.cleanup` keeping what
///   `guessPairType` calls an exception, because a group-level value would
///   otherwise leak back onto a pair the designer explicitly zeroed.
///
/// Values stay f64: `MathKerning.round` fires only under `--round-instances`,
/// and the kern feature writer `otRound`s later.
pub fn pin_kerning<'a>(
    static_metadata: &StaticMetadata,
    instances: impl IntoIterator<Item = &'a KerningInstance>,
    pin: &NormalizedLocation,
) -> Result<KerningInstance, DeltaError> {
    let instances: Vec<_> = instances.into_iter().collect();
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    let key = static_metadata.default_location();

    // ufo2ft: the instance gets the default source's groups
    let groups = instances
        .iter()
        .find(|instance| instance.location.is_default())
        .map(|instance| instance.groups.clone())
        .unwrap_or_default();
    let side1_of = glyph_to_group(&groups, |group| matches!(group, KernGroup::Side1(..)));
    let side2_of = glyph_to_group(&groups, |group| matches!(group, KernGroup::Side2(..)));

    let by_location: HashMap<NormalizedLocation, &KerningInstance> = instances
        .iter()
        .map(|instance| (fit(&instance.location, &axis_order), *instance))
        .collect();
    let model = VariationModel::new(by_location.keys().cloned().collect(), axis_order);

    // Only masters the pin actually reaches, in model order; a zero scalar
    // means the master is not part of this instance at all.
    let contributors: Vec<(&KerningInstance, f64)> = model
        .locations()
        .zip(model.master_scalars(pin))
        .filter(|(_, scalar)| *scalar != 0.0)
        .filter_map(|(loc, scalar)| by_location.get(loc).map(|instance| (*instance, scalar)))
        .collect();

    let pairs: BTreeSet<&KernPair> = contributors
        .iter()
        .flat_map(|(instance, _)| instance.kerns.keys())
        .collect();

    let mut kerns = BTreeMap::new();
    for pair in pairs {
        let value: f64 = contributors
            .iter()
            .map(|(instance, scalar)| resolve_kern(instance, pair, &side1_of, &side2_of) * scalar)
            .sum();
        // MathKerning.cleanup: a zero pair only earns its keep as an exception
        if value == 0.0
            && !is_group_member(&pair.0, &side1_of)
            && !is_group_member(&pair.1, &side2_of)
        {
            continue;
        }
        kerns.insert(pair.clone(), OrderedFloat(value));
    }

    Ok(KerningInstance {
        location: key.clone(),
        kerns,
        groups,
    })
}

/// `glyph -> the group it belongs to`, for the groups `keep` accepts.
///
/// fontMath's `updateGroups`: a glyph in two groups on the same side takes the
/// last one. Ours is last in [`KernGroup`] order rather than in source order,
/// which is a distinction without a difference for well-formed sources — a
/// glyph belongs to at most one kern group per side.
fn glyph_to_group(
    groups: &BTreeMap<KernGroup, BTreeSet<GlyphName>>,
    keep: impl Fn(&KernGroup) -> bool,
) -> HashMap<GlyphName, KernGroup> {
    groups
        .iter()
        .filter(|(group, _)| keep(group))
        .flat_map(|(group, members)| {
            members
                .iter()
                .map(move |member| (member.clone(), group.clone()))
        })
        .collect()
}

/// What one master says about `pair`, through fontMath's fallback chain.
///
/// `MathKerning.__getitem__`: the literal pair, else the class pair covering
/// each side in turn, else 0. Note that a side which *is* a group name has no
/// glyph form, so the probes that need one are skipped rather than falling back
/// to the glyph's own group.
fn resolve_kern(
    instance: &KerningInstance,
    pair: &KernPair,
    side1_of: &HashMap<GlyphName, KernGroup>,
    side2_of: &HashMap<GlyphName, KernGroup>,
) -> f64 {
    if let Some(value) = instance.kerns.get(pair) {
        return value.into_inner();
    }

    let group_side = |side: &KernSide, of: &HashMap<GlyphName, KernGroup>| match side {
        KernSide::Group(..) => Some(side.clone()),
        KernSide::Glyph(name) => of.get(name).cloned().map(KernSide::Group),
    };
    fn glyph_side(side: &KernSide) -> Option<&KernSide> {
        matches!(side, KernSide::Glyph(..)).then_some(side)
    }

    let (group1, group2) = (group_side(&pair.0, side1_of), group_side(&pair.1, side2_of));
    let (glyph1, glyph2) = (glyph_side(&pair.0), glyph_side(&pair.1));

    let probe = |one: Option<&KernSide>, two: Option<&KernSide>| -> Option<f64> {
        let pair = (one?.clone(), two?.clone());
        instance.kerns.get(&pair).map(|value| value.into_inner())
    };

    probe(group1.as_ref(), glyph2)
        .or_else(|| probe(glyph1, group2.as_ref()))
        .or_else(|| probe(group1.as_ref(), group2.as_ref()))
        .unwrap_or_default()
}

/// Is this side a plain glyph that belongs to a kern group?
///
/// fontMath calls such a side an "exception" and never cleans up a zero pair
/// that has one.
fn is_group_member(side: &KernSide, of: &HashMap<GlyphName, KernGroup>) -> bool {
    match side {
        KernSide::Glyph(name) => of.contains_key(name),
        KernSide::Group(..) => false,
    }
}

/// `loc` restricted to exactly `axis_order`, the way the variation model keys.
fn fit(loc: &NormalizedLocation, axis_order: &[Tag]) -> NormalizedLocation {
    let mut loc = loc.clone();
    loc.fit_to_axes(axis_order);
    loc
}

/// Glyphs.app per-master numbers reduced to their values at `pin`.
///
/// Each named number interpolates from the masters that define it, and a master
/// that doesn't define it simply isn't in that number's model. It is genuinely
/// sparse, not fontMath's partial sum: the model is rebuilt per name, so the
/// scalars renormalise over the masters that have it. There is no fontmake
/// behaviour to match either way — Glyphs number values have no UFO equivalent
/// and never reach `MathInfo`.
pub fn pin_number_values(
    static_metadata: &StaticMetadata,
    pin: &NormalizedLocation,
) -> Result<HashMap<NormalizedLocation, BTreeMap<SmolStr, OrderedFloat<f64>>>, DeltaError> {
    if static_metadata.number_values.is_empty() {
        return Ok(Default::default());
    }
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    let key = static_metadata.default_location();

    let names: HashSet<&SmolStr> = static_metadata
        .number_values
        .values()
        .flat_map(|values| values.keys())
        .collect();

    let mut pinned = BTreeMap::new();
    for name in names {
        let point_seqs: HashMap<_, _> = static_metadata
            .number_values
            .iter()
            .filter_map(|(loc, values)| {
                values
                    .get(name)
                    .map(|value| (fit(loc, &axis_order), vec![value.into_inner()]))
            })
            .collect();
        let model = VariationModel::new(point_seqs.keys().cloned().collect(), axis_order.clone());
        let value = model
            .interpolate_from_masters(pin, &point_seqs)?
            .first()
            .copied()
            .unwrap_or(0.0);
        pinned.insert(name.clone(), OrderedFloat(value));
    }

    Ok(HashMap::from([(key.clone(), pinned)]))
}

/// The masters' PostScript settings reduced to the one set the pin wants.
///
/// fontMath treats every numeric `postscript*` fontinfo key as an ordinary
/// interpolating attribute — there is no "nearest master" anywhere in
/// `fontmake -i` — but its rules for a *missing* term are unusual, and they are
/// what makes this more than a loop over [`VariationModel::master_scalars`]:
///
/// - A master that doesn't state an attribute is not sparse and is not
///   renormalised. It simply drops out of the sum, so the instance gets
///   `sum(scalar_i * value_i)` over only the masters that *do* state it: one
///   master saying `blue_shift = 8` and another saying nothing gives **4** at
///   the midpoint, not 8. (`_processMathOne`, where the present term is the
///   whole answer when the other is `None`.)
/// - Two contributing masters whose lists have **different lengths** annihilate
///   the attribute — it becomes absent, *not* the default master's value, and
///   ufo2ft's CFF writer then drops `StemSnapH` *and* `StemSnapV` when either
///   is empty. (`_processMathOneNumberList`.)
/// - Both rules are applied left to right, so `[len4, len6, len4]` yields only
///   the third master's scaled list while `[len4, len4, len6]` yields nothing.
///   fontMath accumulates in designspace source order; IR has no source order,
///   so we accumulate in model order. Only a font with three or more
///   contributing masters *and* disagreeing list lengths can tell the
///   difference.
/// - A pin that lands on a master has one contributing term, so none of the
///   binary rules fire and that master's values survive verbatim — which is
///   also what ufo2ft's exact-master short circuit does.
///
/// Everything stays unrounded; the CFF writer does the `otRound`ing, and
/// `blue_scale` is never rounded at all.
///
/// [`PostscriptSettings::force_bold`] is not interpolated: it is on ufo2ft's
/// copy-from-the-default-master whitelist. [`PostscriptSettings::full_name`] is
/// not a `MathInfo` attribute and not on that whitelist, so an interpolated
/// instance never has one however many masters set it.
pub fn pin_postscript(
    static_metadata: &StaticMetadata,
    pin: &NormalizedLocation,
) -> PostscriptSettings {
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    let by_location: HashMap<NormalizedLocation, &PostscriptSettings> = static_metadata
        .postscript
        .iter()
        .map(|(loc, settings)| (fit(loc, &axis_order), settings))
        .collect();
    let model = VariationModel::new(by_location.keys().cloned().collect(), axis_order);

    let contributors: Vec<(&PostscriptSettings, f64)> = model
        .locations()
        .zip(model.master_scalars(pin))
        .filter(|(_, scalar)| *scalar != 0.0)
        .filter_map(|(loc, scalar)| by_location.get(loc).map(|settings| (*settings, scalar)))
        .collect();

    let list = |get: fn(&PostscriptSettings) -> &Vec<OrderedFloat<f64>>| -> Vec<OrderedFloat<f64>> {
        contributors
            .iter()
            .fold(None, |acc, (settings, scalar)| {
                accumulate_list(acc, get(settings), *scalar)
            })
            .unwrap_or_default()
            .into_iter()
            .map(OrderedFloat)
            .collect()
    };
    let number = |get: fn(&PostscriptSettings) -> Option<OrderedFloat<f64>>| {
        contributors
            .iter()
            .fold(None, |acc, (settings, scalar)| {
                accumulate_number(acc, get(settings), *scalar)
            })
            .map(OrderedFloat)
    };

    PostscriptSettings {
        blue_values: list(|ps| &ps.blue_values),
        other_blues: list(|ps| &ps.other_blues),
        family_blues: list(|ps| &ps.family_blues),
        family_other_blues: list(|ps| &ps.family_other_blues),
        blue_scale: number(|ps| ps.blue_scale),
        blue_shift: number(|ps| ps.blue_shift),
        blue_fuzz: number(|ps| ps.blue_fuzz),
        stem_snap_h: list(|ps| &ps.stem_snap_h),
        stem_snap_v: list(|ps| &ps.stem_snap_v),
        // copied from the default master, not interpolated
        force_bold: static_metadata.postscript_default().force_bold,
        // Derived, in fontMath, from the *interpolated* OS/2 weight class, which
        // is why an instance's CFF `Weight` and its `OS/2.usWeightClass` can
        // legitimately disagree. IR has one `misc.us_weight_class` for the whole
        // font rather than one per master, so there is nothing here to
        // interpolate; this waits on the commit that makes the OS/2 classes
        // instance-aware. Absent is at least what fontmake produces for a pin
        // that lands on a master, where no math runs and `postscriptWeightName`
        // is never set.
        weight_name: None,
        full_name: None,
        default_width_x: number(|ps| ps.default_width_x),
        nominal_width_x: number(|ps| ps.nominal_width_x),
    }
}

/// fontMath's `_processMathOne` for one number, pre-scaled by `scalar`.
///
/// An absent term is the identity, so an attribute only some masters state
/// accumulates a partial sum rather than being renormalised.
fn accumulate_number(
    acc: Option<f64>,
    value: Option<OrderedFloat<f64>>,
    scalar: f64,
) -> Option<f64> {
    match (acc, value) {
        (Some(acc), Some(value)) => Some(acc + value.into_inner() * scalar),
        (Some(acc), None) => Some(acc),
        (None, value) => value.map(|value| value.into_inner() * scalar),
    }
}

/// fontMath's `_processMathOneNumberList` for one list, pre-scaled by `scalar`.
///
/// Element-wise, except that a length mismatch wipes the attribute out. `None`
/// is both "nothing yet" and "wiped out" — the two are indistinguishable to
/// fontMath, which lets the *next* master resurrect a wiped-out attribute.
///
/// An empty list is IR's way of saying the master didn't state the attribute.
/// A UFO can distinguish an absent key from an explicitly empty array, and
/// fontMath would treat the latter as a length mismatch; IR cannot, and treats
/// it as absent, which is what every real source means.
fn accumulate_list(
    acc: Option<Vec<f64>>,
    values: &[OrderedFloat<f64>],
    scalar: f64,
) -> Option<Vec<f64>> {
    if values.is_empty() {
        return acc;
    }
    let term = values.iter().map(|value| value.into_inner() * scalar);
    match acc {
        None => Some(term.collect()),
        Some(acc) if acc.len() == values.len() => {
            Some(acc.iter().zip(term).map(|(acc, term)| acc + term).collect())
        }
        Some(_) => None,
    }
}

/// Global metrics evaluated at `pin` and re-published as a static space.
///
/// Every metric the input defines gets an entry, because `GlobalMetrics::deltas`
/// assumes the map is complete.
///
/// Note this is the one pin that is not unrounded, and cannot be:
/// [`GlobalMetricsBuilder::build`] rounds each master value before computing
/// deltas (deliberately, so that instancing a *variable* font at a master
/// matches building that master), so by the time [`GlobalMetrics`] exists the
/// unrounded master values are gone. What we interpolate is therefore
/// `interp(round(master))` where fontmake computes `round(interp(master))`,
/// and re-publishing through the builder rounds the result again. That second
/// round costs nothing today — every consumer `ot_round`s the value it reads,
/// and `ot_round` is idempotent — but the first one can be off by a unit.
/// Interpolating the builder's unrounded values instead is a separate change.
pub fn pin_global_metrics(
    static_metadata: &StaticMetadata,
    metrics: &GlobalMetrics,
    pin: &NormalizedLocation,
) -> Result<GlobalMetrics, Error> {
    let key = static_metadata.default_location();
    let mut builder = GlobalMetricsBuilder::new();
    for (metric, _) in metrics.iter() {
        builder.set(*metric, key.clone(), metrics.get(*metric, pin));
    }
    builder.build(&Axes::default())
}

/// `static_metadata` rewritten as the static metadata of the instance at `pin`.
///
/// Axes are emptied — `axes.is_empty()` is fontc's universal "this is a static
/// font" test — and the variation model collapses to the single pinned
/// location. `all_source_axes` and the private default location are left alone:
/// they are what the pinned keys are built from, and the frontends that read
/// `all_source_axes` have already run.
///
/// What this deliberately does *not* do, because it belongs to the caller:
///
/// - Feature variation rules must already have been applied (or rejected):
///   `variations` is cleared here because the feature-variation writer looks a
///   rule's axis up in `axes` and panics when it isn't there, and after this
///   `axes` is empty.
/// - Name records >= 256 minted for axis labels and named instances are left in
///   place; with fvar and STAT skipped they are orphans and want pruning.
/// - `misc.us_weight_class`/`us_width_class`, `selection_flags` and the name
///   table still describe the family, not the instance. Until they don't,
///   [`pin_postscript`] cannot derive `weight_name` either — fontMath reads it
///   off the interpolated OS/2 weight class.
/// - `italic_angle` is a single scalar taken from the default master, so an
///   instance on a `slnt`/`ital` axis gets the wrong `post.italicAngle`.
pub fn pin_static_metadata(
    static_metadata: &StaticMetadata,
    pin: &NormalizedLocation,
) -> Result<StaticMetadata, DeltaError> {
    let key = static_metadata.default_location().clone();
    let number_values = pin_number_values(static_metadata, pin)?;
    let postscript = HashMap::from([(key.clone(), pin_postscript(static_metadata, pin))]);

    // clone-and-overwrite, not struct update syntax: the private
    // default_location field must survive untouched
    let mut pinned = static_metadata.clone();
    pinned.axes = Axes::default();
    pinned.named_instances = Vec::new();
    pinned.variation_model = VariationModel::new(HashSet::from([key]), Vec::new());
    pinned.number_values = number_values;
    pinned.postscript = postscript;
    pinned.variations = None;
    Ok(pinned)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use fontdrasil::{
        coords::UserCoord,
        types::{Axis, GlyphName},
    };
    use kurbo::{Affine, BezPath};
    use write_fonts::types::Tag;

    use crate::ir::{AnchorKind, Component, GlobalMetric, KernGroup, KernSide};

    use super::*;

    const WGHT: Tag = Tag::new(b"wght");

    fn wght() -> Axis {
        Axis {
            name: "Weight".to_string(),
            tag: WGHT,
            min: UserCoord::new(400.0),
            default: UserCoord::new(400.0),
            max: UserCoord::new(700.0),
            hidden: false,
            converter: fontdrasil::coords::CoordConverter::unmapped(
                UserCoord::new(400.0),
                UserCoord::new(400.0),
                UserCoord::new(700.0),
            ),
            localized_names: Default::default(),
        }
    }

    fn at(pos: f64) -> NormalizedLocation {
        NormalizedLocation::for_pos(&[("wght", pos)])
    }

    fn regular() -> NormalizedLocation {
        at(0.0)
    }

    fn bold() -> NormalizedLocation {
        at(1.0)
    }

    fn mid() -> NormalizedLocation {
        at(0.5)
    }

    fn test_static_metadata_at(locations: HashSet<NormalizedLocation>) -> StaticMetadata {
        StaticMetadata::new(
            1000,
            Default::default(),
            vec![wght()],
            Default::default(),
            locations,
            None,
            0.0,
            None,
            false,
        )
        .unwrap()
    }

    /// Two masters on one wght axis, nothing else populated.
    fn test_static_metadata() -> StaticMetadata {
        test_static_metadata_at(HashSet::from([regular(), bold()]))
    }

    /// Three masters on one wght axis, at 0, 0.5 and 1.
    fn test_static_metadata_3() -> StaticMetadata {
        test_static_metadata_at(HashSet::from([regular(), mid(), bold()]))
    }

    fn floats(values: &[f64]) -> Vec<OrderedFloat<f64>> {
        values.iter().copied().map(OrderedFloat).collect()
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        let mut path = BezPath::new();
        path.move_to((x0, y0));
        path.line_to((x1, y0));
        path.line_to((x1, y1));
        path.line_to((x0, y1));
        path.close_path();
        path
    }

    fn instance(width: f64, contours: Vec<BezPath>, components: Vec<Component>) -> GlyphInstance {
        GlyphInstance {
            width,
            height: None,
            vertical_origin: None,
            contours,
            components,
        }
    }

    fn two_master_glyph(
        name: &str,
        regular_inst: GlyphInstance,
        bold_inst: GlyphInstance,
    ) -> Glyph {
        Glyph::new(
            name.into(),
            true,
            Default::default(),
            HashMap::from([(regular(), regular_inst), (bold(), bold_inst)]),
        )
        .unwrap()
    }

    fn parse(spec: &str) -> Result<InstanceSpec, InstanceSpecError> {
        spec.parse()
    }

    #[test]
    fn parse_a_location() {
        assert_eq!(
            parse("wght=700,wdth=87.5").unwrap(),
            InstanceSpec::Location(
                vec![
                    (WGHT, UserCoord::new(700.0)),
                    (Tag::new(b"wdth"), UserCoord::new(87.5)),
                ]
                .into()
            )
        );
    }

    #[test]
    fn parse_a_style_name() {
        assert_eq!(
            parse("Bold Condensed").unwrap(),
            InstanceSpec::Named("Bold Condensed".to_string())
        );
    }

    #[test]
    fn parse_rejects_what_it_cannot_read() {
        for (spec, wanted) in [
            ("", "not empty"),
            ("weight=700", "not an axis tag"),
            ("wght=heavy", "not a position"),
            ("wght=700,wdth", "not 'axis=position'"),
            ("wght=1,wght=2", "more than once"),
        ] {
            let e = parse(spec).unwrap_err().to_string();
            assert!(e.contains(wanted), "'{spec}' gave '{e}', wanted '{wanted}'");
        }
    }

    #[test]
    fn pin_glyph_midpoint_of_contours_and_width() {
        let meta = test_static_metadata();
        let glyph = two_master_glyph(
            "box",
            instance(500.0, vec![rect(0.0, 0.0, 100.0, 700.0)], Vec::new()),
            instance(600.0, vec![rect(0.0, 0.0, 300.0, 701.0)], Vec::new()),
        );

        let pinned = pin_glyph(&meta, &glyph, &mid()).unwrap();

        // one source, keyed by the source's own default location
        assert_eq!(
            pinned.sources().keys().collect::<Vec<_>>(),
            vec![meta.default_location()]
        );
        let inst = pinned.default_instance();
        assert_eq!(inst.width, 550.0);
        // the odd y1 lands on .5: interpolation must not round
        assert_eq!(
            inst.contours[0].to_svg(),
            rect(0.0, 0.0, 200.0, 700.5).to_svg()
        );
    }

    #[test]
    fn pin_glyph_midpoint_of_component_transforms() {
        let meta = test_static_metadata();
        let component = |scale: f64, dx: f64| Component {
            base: "box".into(),
            transform: Affine::new([scale, 0.0, 0.0, scale, dx, 0.0]),
            anchor: None,
        };
        let glyph = two_master_glyph(
            "boxbox",
            instance(500.0, Vec::new(), vec![component(1.0, 0.0)]),
            instance(600.0, Vec::new(), vec![component(2.0, 41.0)]),
        );

        let pinned = pin_glyph(&meta, &glyph, &mid()).unwrap();

        let coeffs = pinned.default_instance().components[0]
            .transform
            .as_coeffs();
        assert_eq!(coeffs, [1.5, 0.0, 0.0, 1.5, 20.5, 0.0]);
    }

    #[test]
    fn pin_glyph_at_a_master_is_that_master() {
        let meta = test_static_metadata();
        let bold_instance = instance(600.0, vec![rect(0.0, 0.0, 300.0, 701.0)], Vec::new());
        let glyph = two_master_glyph(
            "box",
            instance(500.0, vec![rect(0.0, 0.0, 100.0, 700.0)], Vec::new()),
            bold_instance.clone(),
        );

        let pinned = pin_glyph(&meta, &glyph, &bold()).unwrap();

        assert_eq!(*pinned.default_instance(), bold_instance);
    }

    #[test]
    fn pin_glyph_sparse_master() {
        let meta = test_static_metadata();
        // only defined at the default; nothing to interpolate from
        let glyph = Glyph::new(
            "box".into(),
            true,
            Default::default(),
            HashMap::from([(
                regular(),
                instance(500.0, vec![rect(0.0, 0.0, 100.0, 700.0)], Vec::new()),
            )]),
        )
        .unwrap();

        let pinned = pin_glyph(&meta, &glyph, &mid()).unwrap();

        assert_eq!(pinned.default_instance().width, 500.0);
    }

    #[test]
    fn pin_anchors_midpoint() {
        let meta = test_static_metadata();
        let anchors = GlyphAnchors::new(
            GlyphName::from("A"),
            vec![
                Anchor {
                    kind: AnchorKind::Base("top".into()),
                    original_name: "top".into(),
                    positions: HashMap::from([
                        (regular(), Point::new(50.0, 700.0)),
                        (bold(), Point::new(60.0, 701.0)),
                    ]),
                },
                Anchor {
                    // present at only one master: interpolates from what it has
                    kind: AnchorKind::Base("bottom".into()),
                    original_name: "bottom".into(),
                    positions: HashMap::from([(regular(), Point::new(50.0, 0.0))]),
                },
            ],
        );

        let pinned = pin_anchors(&meta, &anchors, &mid()).unwrap();

        assert_eq!(pinned.anchors.len(), 2);
        for anchor in &pinned.anchors {
            assert_eq!(
                anchor.positions.keys().collect::<Vec<_>>(),
                vec![meta.default_location()]
            );
        }
        assert_eq!(pinned.anchors[0].default_pos(), Point::new(55.0, 700.5));
        assert_eq!(pinned.anchors[1].default_pos(), Point::new(50.0, 0.0));
    }

    fn kern_pair(one: &str, two: &str) -> KernPair {
        (KernSide::Glyph(one.into()), KernSide::Glyph(two.into()))
    }

    fn side1(name: &str) -> KernSide {
        KernSide::Group(KernGroup::Side1(name.into()))
    }

    fn side2(name: &str) -> KernSide {
        KernSide::Group(KernGroup::Side2(name.into()))
    }

    fn kerns(pairs: &[(KernPair, f64)]) -> BTreeMap<KernPair, OrderedFloat<f64>> {
        pairs
            .iter()
            .map(|(pair, value)| (pair.clone(), OrderedFloat(*value)))
            .collect()
    }

    fn kern_groups(groups: &[(KernGroup, &[&str])]) -> BTreeMap<KernGroup, BTreeSet<GlyphName>> {
        groups
            .iter()
            .map(|(group, members)| {
                (
                    group.clone(),
                    members.iter().map(|m| GlyphName::from(*m)).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn pin_kerning_midpoint() {
        let meta = test_static_metadata();
        let group = KernGroup::Side1("A".into());
        let groups = BTreeMap::from([(group.clone(), BTreeSet::from([GlyphName::from("A")]))]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: kerns(&[
                    (kern_pair("A", "V"), -20.0),
                    (kern_pair("A", "space"), -3.0),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: kerns(&[(kern_pair("A", "V"), -41.0)]),
                // a non-default master's groups are ignored
                groups: Default::default(),
            },
        ];

        let pinned = pin_kerning(&meta, instances.iter(), &mid()).unwrap();

        assert_eq!(&pinned.location, meta.default_location());
        assert_eq!(pinned.groups, groups);
        assert_eq!(
            pinned.kerns.get(&kern_pair("A", "V")),
            Some(&OrderedFloat(-30.5))
        );
        // nothing in Bold covers ('A', 'space') - not the pair, not a class
        // pair - so Bold contributes 0 to it
        assert_eq!(
            pinned.kerns.get(&kern_pair("A", "space")),
            Some(&OrderedFloat(-1.5))
        );
    }

    /// Thin kerns literally, Bold only by class, at t = 0.5 and t = 0.25.
    ///
    /// Numbers measured from `fontmake -m Kern.designspace -i`, fontmake 3.12.1:
    /// see `Kern-Mid.ufo` / `Kern-Q.ufo`. `('A','O')` is the point of the whole
    /// cascade — Bold has no such pair, and contributing 0 there would give
    /// -30, not -130.
    #[test]
    fn pin_kerning_resolves_class_kerning_like_fontmake() {
        let meta = test_static_metadata();
        let groups = kern_groups(&[
            (KernGroup::Side1("A".into()), &["A", "T"]),
            (KernGroup::Side2("O".into()), &["O"]),
        ]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: kerns(&[
                    (kern_pair("A", "O"), -60.0),
                    (kern_pair("A", "space"), -3.0),
                    (kern_pair("V", "O"), 20.0),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: kerns(&[
                    (kern_pair("V", "O"), -20.0),
                    ((side1("A"), side2("O")), -200.0),
                ]),
                groups: groups.clone(),
            },
        ];

        let mid = pin_kerning(&meta, instances.iter(), &mid()).unwrap();
        assert_eq!(
            mid.kerns,
            kerns(&[
                // 0.5 * -60 + 0.5 * (-200, via the class pair)
                (kern_pair("A", "O"), -130.0),
                // 0.5 * -3 + 0.5 * 0; nothing in Bold covers 'space'
                (kern_pair("A", "space"), -1.5),
                // zero, but survives cleanup because 'O' is a kern2 member
                (kern_pair("V", "O"), 0.0),
                ((side1("A"), side2("O")), -100.0),
            ])
        );

        let quarter = pin_kerning(&meta, instances.iter(), &at(0.25)).unwrap();
        assert_eq!(
            quarter.kerns,
            kerns(&[
                (kern_pair("A", "O"), -95.0),
                (kern_pair("A", "space"), -2.25),
                (kern_pair("V", "O"), 10.0),
                ((side1("A"), side2("O")), -50.0),
            ])
        );
    }

    /// Three masters, and at every pin one of them has a zero scalar.
    ///
    /// Numbers measured from `fontmake -m K3.designspace -i`, fontmake 3.12.1:
    /// see `K3-P250.ufo` / `K3-P600.ufo` / `K3-P900.ufo`. The float noise is
    /// fontmake's own and is reproduced exactly. What this pins beyond the
    /// two-master case:
    ///
    /// - at 0.6 and 0.9 the first master's scalar is zero, so `('A','space')`
    ///   — a pair only *it* states — is not in the instance at all;
    /// - `('T','O')` is in no master at 0.25 as a literal pair, and resolves
    ///   through a different rung of the cascade in each master.
    #[test]
    fn pin_kerning_three_masters_matches_fontmake() {
        let meta = test_static_metadata_3();
        let groups = kern_groups(&[
            (KernGroup::Side1("A".into()), &["A", "T"]),
            (KernGroup::Side1("V".into()), &["V"]),
            (KernGroup::Side2("O".into()), &["O"]),
        ]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: kerns(&[
                    (kern_pair("A", "O"), 100.0),
                    (kern_pair("A", "space"), -7.0),
                    ((side1("A"), side2("O")), -50.0),
                    ((side1("V"), KernSide::Glyph("O".into())), 30.0),
                    (kern_pair("x", "x"), 5.0),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: mid(),
                kerns: kerns(&[
                    (kern_pair("A", "O"), -100.0),
                    (kern_pair("T", "O"), 11.0),
                    ((side1("A"), side2("O")), -50.0),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: kerns(&[
                    (kern_pair("A", "O"), 0.0),
                    ((side1("A"), KernSide::Glyph("space".into())), 9.0),
                    ((side1("V"), KernSide::Glyph("O".into())), -30.0),
                    (kern_pair("x", "x"), -5.0),
                ]),
                groups: groups.clone(),
            },
        ];

        let pinned = pin_kerning(&meta, instances.iter(), &at(0.25)).unwrap();
        assert_eq!(
            pinned.kerns,
            kerns(&[
                // 0.5 * 100 + 0.5 * -100; kept, 'A' is a kern1 member
                (kern_pair("A", "O"), 0.0),
                (kern_pair("A", "space"), -3.5),
                // 0.5 * (-50, via the class pair) + 0.5 * 11
                (kern_pair("T", "O"), -19.5),
                ((side1("A"), side2("O")), -50.0),
                ((side1("V"), KernSide::Glyph("O".into())), 15.0),
                (kern_pair("x", "x"), 2.5),
            ])
        );

        let pinned = pin_kerning(&meta, instances.iter(), &at(0.6)).unwrap();
        assert_eq!(
            pinned.kerns,
            kerns(&[
                (kern_pair("A", "O"), -80.0),
                (kern_pair("T", "O"), 8.8),
                ((side1("A"), side2("O")), -40.0),
                (
                    (side1("A"), KernSide::Glyph("space".into())),
                    1.7999999999999996
                ),
                (
                    (side1("V"), KernSide::Glyph("O".into())),
                    -5.999999999999998
                ),
                (kern_pair("x", "x"), -0.9999999999999998),
            ])
        );

        let pinned = pin_kerning(&meta, instances.iter(), &at(0.9)).unwrap();
        assert_eq!(
            pinned.kerns,
            kerns(&[
                (kern_pair("A", "O"), -19.999999999999996),
                (kern_pair("T", "O"), 2.1999999999999993),
                ((side1("A"), side2("O")), -9.999999999999998),
                ((side1("A"), KernSide::Glyph("space".into())), 7.2),
                ((side1("V"), KernSide::Glyph("O".into())), -24.0),
                (kern_pair("x", "x"), -4.0),
            ])
        );
    }

    #[test]
    fn pin_kerning_cleanup_keeps_only_zero_exceptions() {
        let meta = test_static_metadata();
        let groups = kern_groups(&[
            (KernGroup::Side1("A".into()), &["A"]),
            (KernGroup::Side2("O".into()), &["O"]),
        ]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: kerns(&[
                    (kern_pair("A", "space"), 10.0),
                    (kern_pair("space", "O"), 10.0),
                    (kern_pair("x", "x"), 5.0),
                    ((side1("A"), side2("O")), 12.0),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: kerns(&[
                    (kern_pair("A", "space"), -10.0),
                    (kern_pair("space", "O"), -10.0),
                    (kern_pair("x", "x"), -5.0),
                    ((side1("A"), side2("O")), -12.0),
                ]),
                groups: groups.clone(),
            },
        ];

        let pinned = pin_kerning(&meta, instances.iter(), &mid()).unwrap();

        assert_eq!(
            pinned.kerns,
            kerns(&[
                // 'A' is a kern1 member, so this zero is an exception: kept
                (kern_pair("A", "space"), 0.0),
                // and so is 'O' on side 2
                (kern_pair("space", "O"), 0.0),
            ]),
            "a zero pair only survives if a side is a plain glyph in a kern group"
        );
    }

    #[test]
    fn pin_number_values_midpoint() {
        let mut meta = test_static_metadata();
        meta.number_values = HashMap::from([
            (
                regular(),
                BTreeMap::from([
                    (SmolStr::new("shoulder"), OrderedFloat(10.0)),
                    (SmolStr::new("crossbar"), OrderedFloat(4.0)),
                ]),
            ),
            (
                bold(),
                BTreeMap::from([(SmolStr::new("shoulder"), OrderedFloat(21.0))]),
            ),
        ]);

        let pinned = pin_number_values(&meta, &mid()).unwrap();

        let values = pinned.get(meta.default_location()).unwrap();
        assert_eq!(values.get("shoulder"), Some(&OrderedFloat(15.5)));
        // defined at only one master, so it doesn't vary
        assert_eq!(values.get("crossbar"), Some(&OrderedFloat(4.0)));
    }

    #[test]
    fn pin_global_metrics_midpoint() {
        let meta = test_static_metadata();
        let mut builder = GlobalMetricsBuilder::new();
        for (loc, x_height, ascender) in [(regular(), 500.0, 700.0), (bold(), 511.0, 700.0)] {
            builder.populate_defaults(&loc, 1000, Some(x_height), Some(ascender), None, None);
        }
        let metrics = builder.build(&Axes::new(vec![wght()])).unwrap();

        let pinned = pin_global_metrics(&meta, &metrics, &mid()).unwrap();

        // every metric the input had, so GlobalMetrics::deltas can't panic
        assert_eq!(pinned.iter().count(), metrics.iter().count());
        let at = pinned.at(meta.default_location());
        // the metric really is evaluated at the pin, not at a master; 505.5 is
        // rounded on the way back in, see the fn docs
        assert_eq!(at.x_height, OrderedFloat(506.0));
        assert_eq!(at.ascender, OrderedFloat(700.0));
        // and the pinned space really is static: same answer anywhere
        assert_eq!(pinned.get(GlobalMetric::XHeight, &bold()), at.x_height);
    }

    #[test]
    fn pin_static_metadata_is_static() {
        let mut meta = test_static_metadata();
        meta.named_instances = vec![crate::ir::NamedInstance {
            name: "Bold".to_string(),
            postscript_name: None,
            location: vec![(WGHT, UserCoord::new(700.0))].into(),
        }];

        let pinned = pin_static_metadata(&meta, &mid()).unwrap();

        assert!(pinned.axes.is_empty(), "axes.is_empty() means static");
        assert!(pinned.named_instances.is_empty());
        assert!(pinned.variations.is_none());
        assert_eq!(pinned.variation_model.num_locations(), 1);
        // what the pin is keyed by is untouched, and so are the source axes
        assert_eq!(pinned.default_location(), meta.default_location());
        assert_eq!(
            pinned.default_location(),
            &NormalizedLocation::for_pos(&[("wght", 0.0)])
        );
        assert_eq!(
            pinned
                .all_source_axes
                .iter()
                .map(|a| a.tag)
                .collect::<Vec<_>>(),
            vec![WGHT]
        );
        assert_eq!(pinned.units_per_em, meta.units_per_em);
    }

    /// The two masters of the `BlueMatch` fixture, whose list lengths agree.
    fn blue_match() -> HashMap<NormalizedLocation, PostscriptSettings> {
        HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    blue_values: floats(&[-10.0, 0.0, 500.0, 510.0, 700.0, 710.0]),
                    other_blues: floats(&[-210.0, -200.0]),
                    family_blues: floats(&[-8.0, 0.0, 490.0, 500.0]),
                    stem_snap_h: floats(&[20.0, 40.0]),
                    stem_snap_v: floats(&[21.0]),
                    blue_scale: Some(OrderedFloat(0.0375)),
                    blue_shift: Some(OrderedFloat(7.0)),
                    blue_fuzz: Some(OrderedFloat(1.0)),
                    force_bold: Some(false),
                    weight_name: Some("Thin".to_string()),
                    full_name: Some("WN Thin".to_string()),
                    default_width_x: Some(OrderedFloat(100.0)),
                    nominal_width_x: Some(OrderedFloat(200.0)),
                    ..Default::default()
                },
            ),
            (
                bold(),
                PostscriptSettings {
                    blue_values: floats(&[-20.0, 0.0, 600.0, 620.0, 800.0, 820.0]),
                    other_blues: floats(&[-310.0, -300.0]),
                    family_blues: floats(&[-18.0, 0.0, 590.0, 601.0]),
                    stem_snap_h: floats(&[30.0, 50.0]),
                    stem_snap_v: floats(&[31.0]),
                    blue_scale: Some(OrderedFloat(0.0475)),
                    blue_shift: Some(OrderedFloat(8.0)),
                    blue_fuzz: Some(OrderedFloat(2.0)),
                    force_bold: Some(true),
                    weight_name: Some("Black".to_string()),
                    full_name: Some("WN Black".to_string()),
                    default_width_x: Some(OrderedFloat(201.0)),
                    nominal_width_x: Some(OrderedFloat(301.0)),
                    ..Default::default()
                },
            ),
        ])
    }

    /// Numbers measured from `fontmake -m BlueMatch.designspace -i` (fontmake
    /// 3.12.1, `--optimize-cff 0`), which is why `blue_scale` carries fontmake's
    /// own float noise rather than a tidy 0.0425.
    #[test]
    fn pin_postscript_interpolates_every_number() {
        let mut meta = test_static_metadata();
        meta.postscript = blue_match();

        let pinned = pin_postscript(&meta, &mid());

        assert_eq!(
            pinned.blue_values,
            floats(&[-15.0, 0.0, 550.0, 565.0, 750.0, 765.0])
        );
        assert_eq!(pinned.other_blues, floats(&[-260.0, -250.0]));
        assert_eq!(pinned.family_blues, floats(&[-13.0, 0.0, 540.0, 550.5]));
        assert_eq!(pinned.stem_snap_h, floats(&[25.0, 45.0]));
        assert_eq!(pinned.stem_snap_v, floats(&[26.0]));
        assert_eq!(pinned.blue_scale, Some(OrderedFloat(0.042499999999999996)));
        assert_eq!(pinned.blue_shift, Some(OrderedFloat(7.5)));
        assert_eq!(pinned.blue_fuzz, Some(OrderedFloat(1.5)));
        assert_eq!(pinned.default_width_x, Some(OrderedFloat(150.5)));
        assert_eq!(pinned.nominal_width_x, Some(OrderedFloat(250.5)));
        // copied from the default master, never interpolated
        assert_eq!(pinned.force_bold, Some(false));
        // both masters name themselves; the instance gets neither
        assert_eq!(pinned.weight_name, None);
        assert_eq!(pinned.full_name, None);
    }

    /// `BlueMismatch`: blues 4 vs 6, stems 2 vs 3. fontmake's instance UFO has
    /// no `postscriptBlueValues` and no `postscriptStemSnapH` at all, and its
    /// CFF then loses `StemSnapV` too because ufo2ft writes stems only when both
    /// directions survive.
    #[test]
    fn pin_postscript_drops_length_mismatched_lists() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    blue_values: floats(&[-10.0, 0.0, 500.0, 510.0]),
                    other_blues: floats(&[-210.0, -200.0]),
                    stem_snap_h: floats(&[20.0, 40.0]),
                    stem_snap_v: floats(&[21.0]),
                    ..Default::default()
                },
            ),
            (
                bold(),
                PostscriptSettings {
                    blue_values: floats(&[-20.0, 0.0, 600.0, 620.0, 800.0, 820.0]),
                    other_blues: floats(&[-310.0, -300.0]),
                    stem_snap_h: floats(&[30.0, 50.0, 70.0]),
                    stem_snap_v: floats(&[31.0]),
                    ..Default::default()
                },
            ),
        ]);

        let pinned = pin_postscript(&meta, &mid());

        assert!(pinned.blue_values.is_empty(), "{:?}", pinned.blue_values);
        assert!(pinned.stem_snap_h.is_empty(), "{:?}", pinned.stem_snap_h);
        // the lists that do agree are unaffected
        assert_eq!(pinned.other_blues, floats(&[-260.0, -250.0]));
        assert_eq!(pinned.stem_snap_v, floats(&[26.0]));
    }

    /// A master's scalar at another master is zero, and a zero-scalar master
    /// contributes nothing at all - not even a length to disagree with. So a pin
    /// at a master is that master, mismatched lists and all, exactly as ufo2ft's
    /// exact-master short circuit gives.
    #[test]
    fn pin_postscript_at_a_master_is_that_master() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    blue_values: floats(&[-10.0, 0.0]),
                    force_bold: Some(false),
                    ..Default::default()
                },
            ),
            (
                bold(),
                PostscriptSettings {
                    blue_values: floats(&[-20.0, 0.0, 600.0, 620.0]),
                    blue_shift: Some(OrderedFloat(8.0)),
                    force_bold: Some(true),
                    weight_name: Some("Black".to_string()),
                    full_name: Some("WN Black".to_string()),
                    ..Default::default()
                },
            ),
        ]);

        let pinned = pin_postscript(&meta, &bold());

        assert_eq!(pinned.blue_values, floats(&[-20.0, 0.0, 600.0, 620.0]));
        assert_eq!(pinned.blue_shift, Some(OrderedFloat(8.0)));
        // ... except force_bold, which is always the *default* master's
        assert_eq!(pinned.force_bold, Some(false));
        // and these two, which no instance ever gets
        assert_eq!(pinned.weight_name, None);
        assert_eq!(pinned.full_name, None);
    }

    /// fontMath does not renormalise around a master that is simply silent: the
    /// instance value is the scaled sum over the masters that speak. Measured
    /// with `openTypeHheaAscender` 900 in one master and unset in the other,
    /// whose midpoint is 450.
    #[test]
    fn pin_postscript_partial_sum_when_a_master_is_silent() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    blue_values: floats(&[-16.0, 0.0]),
                    blue_shift: Some(OrderedFloat(8.0)),
                    ..Default::default()
                },
            ),
            (bold(), Default::default()),
        ]);

        let pinned = pin_postscript(&meta, &mid());

        assert_eq!(pinned.blue_shift, Some(OrderedFloat(4.0)));
        assert_eq!(pinned.blue_values, floats(&[-8.0, 0.0]));
    }

    #[test]
    fn pin_static_metadata_pins_postscript() {
        let mut meta = test_static_metadata();
        meta.postscript = blue_match();

        let pinned = pin_static_metadata(&meta, &mid()).unwrap();

        assert_eq!(
            pinned.postscript.keys().collect::<Vec<_>>(),
            vec![meta.default_location()]
        );
        assert_eq!(
            pinned.postscript_default().blue_values,
            floats(&[-15.0, 0.0, 550.0, 565.0, 750.0, 765.0])
        );
    }

    #[test]
    fn pinned_glyph_key_matches_pinned_metadata_default() {
        // the bug this guards: fontbe looks glyph sources up by an exact
        // default_location() key, so the two must not drift apart
        let meta = test_static_metadata();
        let glyph = two_master_glyph(
            "box",
            instance(500.0, vec![rect(0.0, 0.0, 100.0, 700.0)], Vec::new()),
            instance(600.0, vec![rect(0.0, 0.0, 300.0, 700.0)], Vec::new()),
        );

        let pinned_meta = pin_static_metadata(&meta, &mid()).unwrap();
        let pinned_glyph = pin_glyph(&meta, &glyph, &mid()).unwrap();

        assert!(
            pinned_glyph
                .sources()
                .contains_key(pinned_meta.default_location()),
            "{:?} has no source at {:?}",
            pinned_glyph.sources().keys().collect::<Vec<_>>(),
            pinned_meta.default_location()
        );
        assert_eq!(
            pinned_meta.default_location(),
            &NormalizedLocation::for_pos(&[("wght", 0.0)]),
        );
    }
}
