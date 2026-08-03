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
    coords::{DesignLocation, NormalizedLocation, UserCoord, UserLocation},
    types::{Axes, GlyphName, WidthClass},
    variations::{DeltaError, VariationModel},
};
use kurbo::Point;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use write_fonts::{
    tables::os2::SelectionFlags,
    types::{InvalidTag, NameId, Tag},
};

use crate::{
    error::{BadGlyph, Error},
    ir::{
        Anchor, Condition, GlobalMetrics, GlobalMetricsBuilder, Glyph, GlyphAnchors, GlyphInstance,
        KernGroup, KernPair, KernSide, KerningInstance, NameBuilder, NameKey, NamedInstance,
        PostscriptSettings, Rule, StaticMetadata, StyleMapStyle, VariableFeature,
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

/// Why an [`InstanceSpec`] doesn't name a position this source has.
///
/// The wording is load-bearing: ttx_diff and fontc_crater classify a source
/// they cannot compare by matching on it, the way they already do for
/// "--flavor otf requires a static source".
#[derive(Debug, thiserror::Error)]
pub enum PinError {
    #[error("--instance requires a variable source; this source has no variable axes")]
    RequiresVariableSource,
    #[error("--instance does not know axis '{tag}'; this source has {available}")]
    UnknownAxis { tag: Tag, available: String },
    #[error("--instance puts '{tag}' at {pos}, outside its range {min}..={max}")]
    AxisOutOfRange {
        tag: Tag,
        pos: f64,
        min: f64,
        max: f64,
    },
    #[error("--instance does not know an instance named '{name}'; this source has {available}")]
    UnknownInstance { name: String, available: String },
    #[error("--instance cannot resolve that location: {0}")]
    Location(String),
}

/// The user-space location `spec` names, on every axis, or why it names none.
///
/// Axes the user didn't mention take their default, so the result is a
/// complete location and not the subset that was asked for.
///
/// Lives here rather than beside the CLI because both the pin barrier and the
/// global metrics work need the answer, and the metrics work runs long before
/// the barrier — see [`GlobalMetricsBuilder::build_pinned`].
pub fn resolve_user(
    static_metadata: &StaticMetadata,
    spec: &InstanceSpec,
) -> Result<UserLocation, PinError> {
    // fontc's universal "this is static" test. Point axes are fine: fontmake
    // pins a single-master designspace without complaint, and only rejects
    // `-i` outright for bare UFO input.
    if static_metadata.axes.is_empty() {
        return Err(PinError::RequiresVariableSource);
    }

    let asked_for = match spec {
        InstanceSpec::Named(name) => static_metadata
            .named_instances
            .iter()
            .find(|instance| instance.name == *name)
            .ok_or_else(|| PinError::UnknownInstance {
                name: name.clone(),
                available: comma_separated(
                    static_metadata
                        .named_instances
                        .iter()
                        .map(|instance| instance.name.clone()),
                ),
            })?
            .location
            .clone(),
        InstanceSpec::Location(location) => {
            for (tag, pos) in location.iter() {
                let Some(axis) = static_metadata.axes.get(tag) else {
                    return Err(PinError::UnknownAxis {
                        tag: *tag,
                        available: comma_separated(
                            static_metadata.axes.iter().map(|axis| axis.tag.to_string()),
                        ),
                    });
                };
                // deliberately not clamped: a silently moved pin is worse than
                // a rejected one
                if *pos < axis.min || *pos > axis.max {
                    return Err(PinError::AxisOutOfRange {
                        tag: *tag,
                        pos: pos.to_f64(),
                        min: axis.min.to_f64(),
                        max: axis.max.to_f64(),
                    });
                }
            }
            location.clone()
        }
    };

    // an axis nobody mentioned sits at its default
    Ok(static_metadata
        .axes
        .iter()
        .map(|axis| (axis.tag, asked_for.get(axis.tag).unwrap_or(axis.default)))
        .collect())
}

/// Where `spec` puts us in normalized space, which is how the IR keys everything.
///
/// The conversion runs through the source's own axis mapping (what becomes
/// `avar`), which is the same normalization `fontmake -i` applies to a
/// designspace `<instance>`'s location for any strictly monotonic map.
pub fn resolve(
    static_metadata: &StaticMetadata,
    spec: &InstanceSpec,
) -> Result<NormalizedLocation, PinError> {
    resolve_user(static_metadata, spec)?
        .to_normalized(&static_metadata.axes)
        .map_err(|e| PinError::Location(e.to_string()))
}

fn comma_separated(items: impl Iterator<Item = String>) -> String {
    let items: Vec<_> = items.collect();
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

/// Finish a frontend's [`GlobalMetricsBuilder`], pinned if we're building an instance.
///
/// The one pin that cannot wait for the pin barrier: what `fontmake -i`
/// interpolates is the *unrounded* master values, and
/// [`GlobalMetricsBuilder::build`] rounds them on the way in. So the frontend
/// asks here, while it still has them. Every other pin runs later, on published
/// IR.
///
/// `instance` is [`Context::instance`](crate::orchestration::Context::instance).
/// A spec that doesn't resolve is not this work's business to report — the pin
/// barrier says the same thing with the same words, and stops the build — but
/// there is no sensible metrics to publish either, so it errors.
pub fn build_global_metrics(
    builder: GlobalMetricsBuilder,
    static_metadata: &StaticMetadata,
    instance: Option<&InstanceSpec>,
) -> Result<GlobalMetrics, Error> {
    let Some(spec) = instance else {
        return builder.build(&static_metadata.axes);
    };
    let pin = resolve(static_metadata, spec)?;
    // the instance's own metric parameters, which glyphsLib replays over the
    // interpolated fontinfo; empty when the pin isn't a named instance
    let overrides = named_instance_at(static_metadata, spec, &pin)
        .map(|instance| instance.overrides.metrics.clone())
        .unwrap_or_default();
    builder.build_pinned(
        &static_metadata.axes,
        &pin,
        static_metadata.default_location(),
        &overrides,
    )
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
///
/// # A pin on a master is a copy, not an interpolation
///
/// ufo2ft's `Variator.instance_at` short-circuits when the normalized location
/// equals a master's: that master is `deepcopy`-returned and the variation
/// model never runs. So a glyph whose masters are *point-incompatible* — a
/// different number of points, or a different number of components — still
/// instances fine at any of its own masters, which is Glyphs.app semantics.
/// Interpolating anyway fails with "every point sequence must have the same
/// length" on fonts that `fontmake -i` builds without complaint and that
/// fontc's own *variable* build is happy with, because there the glyph is
/// converted to quadratics first and `fontbe` only warns about what is left.
/// 26 of the 27 corpus targets that hit the delta error at `@default` build
/// once the pin stops interpolating a master into itself.
///
/// Only when the glyph *has* a source there: a sparse glyph that the pin
/// happens to sit on top of has nothing to copy and still interpolates from
/// the masters it does have. [`pin_kerning`] does the same thing for the same
/// reason.
///
/// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L1233-L1235>
pub fn pin_glyph(
    static_metadata: &StaticMetadata,
    glyph: &Glyph,
    pin: &NormalizedLocation,
) -> Result<Glyph, BadGlyph> {
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    let at = fit(pin, &axis_order);
    let master = glyph
        .sources()
        .iter()
        .find(|(loc, _)| fit(loc, &axis_order) == at)
        .map(|(_, instance)| instance.clone());
    let instance = match master {
        Some(instance) => instance,
        None => interpolate_glyph_instance(static_metadata, glyph, pin)?,
    };
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

/// The glyph swaps `variations`' rules ask for at `pin`, in the order to do them.
///
/// This is ufo2ft's `process_rules_swaps`: walk the rules in document order,
/// and for each one whose conditions hold at the pin take every substitution,
/// in order. The result is a *sequence of transpositions*, not a substitution
/// map — each swap sees the previous swap's result, so rules `a -> b` then
/// `b -> c` compose to the 3-cycle `(a b)(b c)`, in which `a` ends up with
/// `b`'s content, `b` with `c`'s and `c` with `a`'s. Resolving the chain to
/// `a -> c` first, or taking a fixed point, is a different font.
///
/// `pin` is in **design** space, because that is the space a rule's conditions
/// are written in ([`Condition::min`]) and the space ufo2ft evaluates them in.
///
/// A rule's condition sets are OR-ed and the conditions within one are AND-ed,
/// so a rule with *no* condition sets never applies while a condition set with
/// no conditions always does. Both bounds of a condition are inclusive, and a
/// condition may state only one of them.
///
/// `exists` filters the glyph being *replaced* only, deliberately: ufo2ft
/// leaves a substitution whose target is missing to fail loudly in the swap
/// rather than silently dropping the rule, so the caller should reject one.
///
/// The designspace `processing="first"/"last"` attribute plays no part.
/// `rulesProcessingLast` is read in exactly one place in the whole toolchain —
/// varLib choosing between the `rvrn` and `rclt` feature tags for a *variable*
/// font's feature variations — and never by the instantiator. A static
/// instance has no feature variations to tag, and its rules are always applied
/// last (after every glyph is interpolated) and always in rule order.
pub fn rule_swaps(
    variations: &VariableFeature,
    pin: &DesignLocation,
    exists: impl Fn(&GlyphName) -> bool,
) -> Vec<(GlyphName, GlyphName)> {
    let mut swaps = Vec::new();
    for rule in &variations.rules {
        if !rule_applies(rule, pin) {
            continue;
        }
        for sub in &rule.substitutions {
            // ufo2ft skips a swap of a glyph with itself at the call site
            if sub.replace != sub.with && exists(&sub.replace) {
                swaps.push((sub.replace.clone(), sub.with.clone()));
            }
        }
    }
    swaps
}

/// Does any of `rule`'s condition sets hold at `pin`?
fn rule_applies(rule: &Rule, pin: &DesignLocation) -> bool {
    rule.conditions
        .iter()
        .any(|set| set.iter().all(|condition| condition_holds(condition, pin)))
}

/// `designspaceLib.evaluateConditions` for one condition: both bounds inclusive.
///
/// A condition on an axis the pin doesn't have cannot be met. ufo2ft raises a
/// `KeyError` here instead, but it evaluates against a location it has already
/// merged over every axis, which is what the caller hands us too — so this is
/// unreachable for rules either frontend builds.
fn condition_holds(condition: &Condition, pin: &DesignLocation) -> bool {
    let Some(pos) = pin.get(condition.axis) else {
        log::warn!(
            "feature variation rule conditions on '{}', which the pin has no position for",
            condition.axis
        );
        return false;
    };
    condition.min.is_none_or(|min| min <= pos) && condition.max.is_none_or(|max| pos <= max)
}

/// Exchange two glyphs' drawings, as ufo2ft's `swap_glyph_names` step 1 does.
///
/// Contours, components and the advance *width*, and nothing else: ufo2ft
/// swaps what a point pen draws plus `width`, so `height` — and with it IR's
/// `vertical_origin` — stays with the glyph it was written for. Nor does the
/// caller swap codepoints, glyph order or `emit_to_binary`: "the rules
/// mechanism is supposed to swap glyphs, not characters", and both glyphs keep
/// their identity and their GID.
pub fn swap_geometry(a: &mut GlyphInstance, b: &mut GlyphInstance) {
    std::mem::swap(&mut a.width, &mut b.width);
    std::mem::swap(&mut a.contours, &mut b.contours);
    std::mem::swap(&mut a.components, &mut b.components);
}

/// Rewrite `old` <-> `new` wherever `instance` uses either as a component base.
///
/// ufo2ft's `swap_glyph_names` step 3, which walks every glyph in the font —
/// including the two being swapped, which by then hold each other's
/// components. Returns whether anything changed.
pub fn swap_component_bases(
    instance: &mut GlyphInstance,
    old: &GlyphName,
    new: &GlyphName,
) -> bool {
    let mut changed = false;
    for component in instance.components.iter_mut() {
        if component.base == *old {
            component.base = new.clone();
            changed = true;
        } else if component.base == *new {
            component.base = old.clone();
            changed = true;
        }
    }
    changed
}

/// Exchange every mention of `old` and `new` in pinned kerning.
///
/// ufo2ft's `swap_glyph_names` steps 4 and 5, applied — like the rest of the
/// swap — to the *finished* instance, i.e. after interpolation: literal pair
/// keys naming either glyph on either side, and kern *group membership*. Group
/// names stay as they are, so a side1 group named for `A` can legitimately end
/// up listing `B`; a group is just a set of glyphs that kern alike, and the
/// swap is what makes that true.
///
/// Renaming the pair keys cannot collide: `old <-> new` is a bijection on
/// glyph names, so it is a bijection on pairs.
pub fn swap_kerning(kerning: &mut KerningInstance, old: &GlyphName, new: &GlyphName) {
    let swap_side = |side: &KernSide| match side {
        KernSide::Glyph(name) if name == old => KernSide::Glyph(new.clone()),
        KernSide::Glyph(name) if name == new => KernSide::Glyph(old.clone()),
        other => other.clone(),
    };
    kerning.kerns = kerning
        .kerns
        .iter()
        .map(|((first, second), value)| ((swap_side(first), swap_side(second)), *value))
        .collect();
    for members in kerning.groups.values_mut() {
        if members.contains(old) || members.contains(new) {
            *members = members
                .iter()
                .map(|member| {
                    if member == old {
                        new.clone()
                    } else if member == new {
                        old.clone()
                    } else {
                        member.clone()
                    }
                })
                .collect();
        }
    }
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

    // ufo2ft's Variator.instance_at short-circuits when the pin IS a master:
    // that master's kerning comes back verbatim (a deep copy), so neither the
    // class-fallback resolution nor MathKerning.cleanup ever runs, and a pair
    // a designer explicitly kerned to zero between two ungrouped glyphs
    // survives. Measured on Bona Nova, whose Bold master states
    // ('maqaf-hb', 'kafdagesh-hb') = 0: fontmake -i "Bona Nova Bold" keeps
    // the zero PairPos rule, and every @default pin is an exact-master pin.
    if let Some(instance) = by_location.get(&fit(pin, &axis_order)) {
        return Ok(KerningInstance {
            location: key.clone(),
            kerns: instance.kerns.clone(),
            groups,
        });
    }

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

    // ufo2ft's instantiator: `if self.info_mutator.is_static_font() and
    // is_at_default`, i.e. one info master and a pin at the default location,
    // the instance inherits the *whole* fontinfo of that master rather than
    // ufo2ft's copy whitelist — "it's OK for it to inherit ALL the fontinfo
    // from the default source". So the two attributes an interpolated instance
    // always loses, `postscriptWeightName` and `postscriptFullName`, come
    // through here. Measured on the three single-master designspaces in the
    // corpus — docrepair-fonts' Caprasimo, Lugrasimo and Bacasime Antique —
    // whose only `--instance @default --flavor otf` diff was fontmake's CFF
    // `Weight` entry.
    //
    // <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L725-L740>
    if let Some(only) = single_master_at_default(static_metadata, &by_location, pin) {
        return only.clone();
    }

    let model = VariationModel::new(by_location.keys().cloned().collect(), axis_order.clone());

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

    // fontMath's math ops derive postscriptWeightName from the accumulated
    // openTypeOS2WeightClass, *unrounded*: `MathInfo.round()` would re-derive
    // it from the rounded value, but fontmake's `round_instances` defaults to
    // False and neither ttx_diff nor gftools passes `--round-instances`.
    let os2_weight_class = number(|ps| ps.os2_weight_class);
    // ...but only when there is arithmetic to do. `Variator.instance_at`
    // short-circuits a location that *is* a master to a deepcopy of that
    // master's MathInfo, so no math op runs, and `MathInfo.__init__` copies
    // only `_infoAttrs` — postscriptWeightName is a "special attribute" and is
    // not among them. The copy therefore has no such attribute at all,
    // `extractInfo`'s `hasattr` guard skips it, and the instance gets no CFF
    // `Weight`. Measured: a two-master designspace with weight classes 100/900
    // emits `Weight` at wght 550 but none at wght 400 or 700.
    //
    // <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L1219-L1237>
    // <https://github.com/robotools/fontMath/blob/0.10.0/Lib/fontMath/mathInfo.py#L11-L14>
    let interpolated = !by_location.contains_key(&fit(pin, &axis_order));

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
        // Past the single-master short circuit above there is more than one
        // info master, so `extractInfo` runs.
        //
        // fontMath does not interpolate `postscriptWeightName`; it derives it
        // from the interpolated `openTypeOS2WeightClass`, which is not the
        // class that reaches OS/2 — for an instance that comes from the axis,
        // so the two legitimately disagree. A `.glyphs` source states no
        // per-master class, so this stays `None` there and no `Weight` is
        // written, which is what fontmake does too.
        //
        // <https://github.com/robotools/fontMath/blob/0.10.0/Lib/fontMath/mathInfo.py#L154-L169>
        weight_name: interpolated
            .then(|| postscript_weight_name(os2_weight_class))
            .flatten(),
        os2_weight_class,
        // `postscriptFullName` is not a `MathInfo` attribute and not on the
        // copy whitelist either, so it is lost — but the *instance's* own is
        // replayed over the top, see [`pin_instance_overrides`].
        full_name: None,
        default_width_x: number(|ps| ps.default_width_x),
        nominal_width_x: number(|ps| ps.nominal_width_x),
    }
}

/// The one and only info master, when the pin sits on it.
///
/// ufo2ft's `is_static_font() and is_at_default`: a designspace with a single
/// non-sparse source, pinned at the default location. Sparse sources contribute
/// no fontinfo and so are not in [`StaticMetadata::postscript`] either, which is
/// what makes counting that map the same test ufo2ft's `collect_info_masters`
/// makes.
fn single_master_at_default<'a>(
    static_metadata: &StaticMetadata,
    by_location: &HashMap<NormalizedLocation, &'a PostscriptSettings>,
    pin: &NormalizedLocation,
) -> Option<&'a PostscriptSettings> {
    if by_location.len() != 1 {
        return None;
    }
    let axis_order: Vec<_> = static_metadata.axes.iter().map(|a| a.tag).collect();
    (fit(pin, &axis_order) == fit(static_metadata.default_location(), &axis_order))
        .then(|| by_location.values().next().copied())
        .flatten()
}

/// The CFF `Weight` an interpolated OS/2 weight class implies.
///
/// fontMath's `_processPostscriptWeightName`: round to the nearest 100 the
/// Python 2 way — halves away from zero, so 250 becomes 300 and not the 200
/// Python 3's banker's rounding would give — clamp to 100..=900, then look the
/// result up in the OS/2 weight-class names.
///
/// <https://github.com/robotools/fontMath/blob/0.10.0/Lib/fontMath/mathInfo.py#L154-L169>
fn postscript_weight_name(os2_weight_class: Option<OrderedFloat<f64>>) -> Option<String> {
    let v = os2_weight_class?.into_inner();
    // round2(v, -2): f64::round is already half-away-from-zero
    let hundreds = (v / 100.0).round() as i32;
    let name = match hundreds.clamp(1, 9) {
        1 => "Thin",
        2 => "Extra-light",
        3 => "Light",
        4 => "Normal",
        5 => "Medium",
        6 => "Semi-bold",
        7 => "Bold",
        8 => "Extra-bold",
        9 => "Black",
        _ => unreachable!("clamped to 1..=9"),
    };
    Some(name.to_string())
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

/// `static_metadata` rewritten as the static metadata of the instance at `pin`.
///
/// Axes are emptied — `axes.is_empty()` is fontc's universal "this is a static
/// font" test — and the variation model collapses to the single pinned
/// location. `all_source_axes` and the private default location are left alone:
/// they are what the pinned keys are built from, and the frontends that read
/// `all_source_axes` have already run.
///
/// `user_pin` is the same position in user space, which is what the OS/2
/// classes are read off, and `instance` is the named instance the pin lands on
/// if it lands on one — see [`named_instance_at`].
///
/// What this deliberately does *not* do, because it belongs to the caller:
///
/// - Feature variation rules must already have been applied (or rejected):
///   `variations` is cleared here because the feature-variation writer looks a
///   rule's axis up in `axes` and panics when it isn't there, and after this
///   `axes` is empty.
/// - Name records >= 256 minted for axis labels and named instances are left in
///   place; with fvar and STAT skipped they are orphans and want pruning.
/// - `italic_angle` is a single scalar taken from the default master, so an
///   instance on a `slnt`/`ital` axis gets the wrong `post.italicAngle`.
///   ufo2ft's own `italicAngle = clamp(slnt_user, -90, 90)` derivation is live
///   under `fontmake -i`, so this is a real gap; it needs a per-master italic
///   angle in IR, which is a change of its own.
pub fn pin_static_metadata(
    static_metadata: &StaticMetadata,
    pin: &NormalizedLocation,
    user_pin: &UserLocation,
    instance: Option<&NamedInstance>,
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

    if let Some(instance) = instance {
        pinned.names = pin_names(static_metadata, instance);
        pinned.misc.selection_flags = pin_selection_flags(static_metadata, instance);
    }
    pin_os2_classes(&mut pinned, &static_metadata.axes, user_pin);
    pin_instance_fontinfo(&mut pinned, static_metadata);
    if let Some(instance) = instance {
        pin_instance_overrides(&mut pinned, instance);
    }

    Ok(pinned)
}

/// Replay the instance's own custom parameters over the pinned metadata.
///
/// glyphsLib's `apply_instance_data_to_ufo` runs *after* the instantiator, on
/// the finished instance UFO, so an instance parameter beats everything the
/// masters said — including the values [`pin_instance_fontinfo`] just chose.
/// Last, therefore.
///
/// The parameters here are the ones with a `ParamHandler` that lands in
/// `ufo.info` (or, for `meta Table`, in a lib key ufo2ft compiles) *and* an
/// equivalent in fontc's static metadata. Name records are
/// [`pin_names`]'s and metrics are
/// [`GlobalMetricsBuilder::build_pinned`](crate::ir::GlobalMetricsBuilder::build_pinned)'s;
/// both run before this and read the same [`InstanceOverrides`].
///
/// Deliberately not here, because glyphsLib has no handler for them at all and
/// so fontmake ignores them too: `xHeight`, `capHeight`, `italicAngle`,
/// `weightClass`/`widthClass` (blacklisted out of the instance lib; the axis
/// mapping decides, see [`pin_os2_classes`]), `Disable Masters`, instance-level
/// `glyphOrder`. Deliberately not here because they need machinery fontc does
/// not have: `Filter`/`PreFilter`, `Rename`/`Reencode Glyphs`,
/// `Replace Feature`/`Prefix`, `Keep`/`Remove Glyphs`, `TTFAutohint options`.
///
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/custom_params.py#L314-L448>
fn pin_instance_overrides(pinned: &mut StaticMetadata, instance: &NamedInstance) {
    let overrides = &instance.overrides;
    if let Some(panose) = overrides.panose.as_ref() {
        pinned.misc.panose = Some(panose.clone());
    }
    if let Some(fs_type) = overrides.fs_type {
        pinned.misc.fs_type = Some(fs_type);
    }
    if let Some(is_fixed_pitch) = overrides.is_fixed_pitch {
        pinned.misc.is_fixed_pitch = Some(is_fixed_pitch);
    }
    if let Some(bits) = overrides.unicode_range_bits.as_ref() {
        pinned.misc.unicode_range_bits = Some(bits.clone());
    }
    if let Some(bits) = overrides.codepage_range_bits.as_ref() {
        pinned.misc.codepage_range_bits = Some(bits.clone());
    }
    if let Some(meta_table) = overrides.meta_table.as_ref() {
        pinned.misc.meta_table = Some(meta_table.clone());
    }
    if let Some(full_name) = overrides.postscript_full_name.as_ref() {
        // one entry, keyed at the pin; see `pin_static_metadata`
        for settings in pinned.postscript.values_mut() {
            settings.full_name = Some(full_name.clone());
        }
    }
    // glyphsLib unions these two into `openTypeOS2Selection`, so an instance
    // that says nothing leaves whatever the font said standing
    for (stated, flag) in [
        (overrides.use_typo_metrics, SelectionFlags::USE_TYPO_METRICS),
        (overrides.has_wws_names, SelectionFlags::WWS),
    ] {
        match stated {
            Some(true) => pinned.misc.selection_flags |= flag,
            Some(false) => pinned.misc.selection_flags -= flag,
            None => (),
        }
    }
    match overrides.use_production_names {
        // `Don't use Production Names`: ufo2ft renames on the compiled binary
        // from `public.postscriptNames`, and this turns that off
        Some(false) => pinned.postscript_names = None,
        // the other direction cannot be honoured here: if the *font* said don't
        // use them the frontend never built the map, and there is nothing at
        // the pin to build it from
        Some(true) if pinned.postscript_names.is_none() => {
            log::warn!(
                "instance '{}' asks for production names but the font turned them off; ignoring",
                instance.name
            );
        }
        _ => (),
    }
}

/// The named instance `pin` builds, if the source has one there.
///
/// `fontmake -i` only ever builds a *named* instance, so this is the case that
/// matters; a pin at an arbitrary location has no style name, no style linking
/// and no PostScript name, and keeps the family's.
///
/// Matching is by resolved normalized location, not by user coordinates: two
/// ways of writing the same position — an axis left out and an axis given its
/// default — have to name the same instance.
///
/// A `--instance` that named an instance outright wins over the location
/// match, and it is the only way to be unambiguous: several instances may sit
/// at one location (a family that ships "Regular" and "Book" at the same
/// weight, say) and a location pin then picks the first in source order. There
/// is nothing better to pick — a location does not say which name was meant —
/// so a caller that cares should pass the name.
///
/// Each candidate's *own* location is normalized, never re-resolved through
/// its name: style names repeat across a second axis all the time — Martian
/// Mono ships four instances called `Regular`, one per width — and looking a
/// name back up finds the first of them, so every one but the first would fail
/// to match itself and the pin would silently fall back to the family's names.
pub fn named_instance_at<'a>(
    static_metadata: &'a StaticMetadata,
    spec: &InstanceSpec,
    pin: &NormalizedLocation,
) -> Option<&'a NamedInstance> {
    let by_name = match spec {
        InstanceSpec::Named(name) => static_metadata
            .named_instances
            .iter()
            .find(|instance| instance.name == *name),
        InstanceSpec::Location(..) => None,
    };
    by_name.or_else(|| {
        static_metadata.named_instances.iter().find(|instance| {
            // an axis the instance doesn't mention sits at its default, the
            // same completion `resolve_user` does
            let at: UserLocation = static_metadata
                .axes
                .iter()
                .map(|axis| {
                    (
                        axis.tag,
                        instance.location.get(axis.tag).unwrap_or(axis.default),
                    )
                })
                .collect();
            at.to_normalized(&static_metadata.axes)
                .is_ok_and(|at| at == *pin)
        })
    })
}

/// The name table of the instance, as ufo2ft would build it from its UFO.
///
/// Every name id the compiler *derives* is rebuilt, because the instance UFO
/// does not inherit any of them: `openTypeNamePreferredFamilyName`,
/// `...SubfamilyName`, `styleMapFamilyName`, `styleMapStyleName`, `styleName`,
/// `postscriptFontName` and `openTypeNameUniqueID` are all on the
/// deliberately-not-copied list, so ids 1, 2, 3, 4, 6, 16 and 17 come from the
/// `<instance>` and from ufo2ft's fallbacks alone. Everything else — copyright,
/// version, designer, licence, the arbitrary `openTypeNameRecords` — is on the
/// copy whitelist and passes through.
///
/// Two ids that are *not* on the whitelist and so are dropped rather than
/// rebuilt: 18 `openTypeNameCompatibleFullName` and 21/22, the WWS names.
///
/// Non-English records are left exactly as they are. For a UFO source they came
/// from `openTypeNameRecords`, which the instance keeps and which override the
/// computed records anyway; [`NameBuilder`] only ever writes English ones.
///
/// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L108-L180>
/// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L387-L464>
fn pin_names(
    static_metadata: &StaticMetadata,
    instance: &NamedInstance,
) -> HashMap<NameKey, String> {
    /// Derived from the instance, so never inherited.
    const REBUILT: [NameId; 7] = [
        NameId::FAMILY_NAME,
        NameId::SUBFAMILY_NAME,
        NameId::UNIQUE_ID,
        NameId::FULL_NAME,
        NameId::POSTSCRIPT_NAME,
        NameId::TYPOGRAPHIC_FAMILY_NAME,
        NameId::TYPOGRAPHIC_SUBFAMILY_NAME,
    ];
    /// Instance-specific and not on ufo2ft's copy whitelist: simply lost.
    ///
    /// Name id 25, the Variations PostScript Name Prefix, is here for a
    /// different reason: it is meaningless in a static font — it exists only so
    /// that a *variable* font can build its named instances' PostScript names —
    /// and glyphsLib registers no handler mapping Glyphs'
    /// `variationsPostScriptNamePrefix` onto any UFO attribute, so no
    /// interpolated instance can carry one however the source spells it.
    /// A `Name Table Entry` naming id 25 still wins: those are applied verbatim
    /// after the table is built, exactly as ufo2ft writes `openTypeNameRecords`.
    ///
    /// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/custom_params.py#L405>
    const DROPPED: [NameId; 4] = [
        NameId::COMPATIBLE_FULL_NAME,
        NameId::WWS_FAMILY_NAME,
        NameId::WWS_SUBFAMILY_NAME,
        NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX,
    ];
    const ENGLISH: u16 = 0x409;

    let mut builder = NameBuilder::default();
    builder.set_version(
        static_metadata.misc.version_major,
        static_metadata.misc.version_minor,
    );
    let mut kept: Vec<(&NameKey, &String)> = static_metadata
        .names
        .iter()
        .filter(|(key, _)| key.lang_id == ENGLISH)
        .filter(|(key, _)| !REBUILT.contains(&key.name_id) && !DROPPED.contains(&key.name_id))
        .collect();
    // HashMap order would otherwise decide which of two records sharing a name
    // id survives NameBuilder's one-key-per-id map
    kept.sort_by_key(|(key, _)| **key);
    for (key, value) in kept {
        builder.add(key.name_id, value.clone());
    }

    // ID 16 <- the instance UFO's familyName, ID 17 <- its styleName. Neither
    // "preferred" name is inherited, so the fallback chain always lands here.
    if let Some(family_name) = instance.family_name.as_ref() {
        builder.add(NameId::TYPOGRAPHIC_FAMILY_NAME, family_name.clone());
    }
    builder.add(NameId::TYPOGRAPHIC_SUBFAMILY_NAME, instance.name.clone());
    // ID 1 and 2, when the source states them. When it doesn't, NameBuilder's
    // RIBBI fallback is already ufo2ft's `styleMapFamilyNameFallback` /
    // `styleMapStyleNameFallback`.
    if let Some(style_map_family_name) = instance.style_map_family_name.as_ref() {
        builder.add(NameId::FAMILY_NAME, style_map_family_name.clone());
    }
    // Verbatim, RIBBI or not: ufo2ft `.title()`s whatever the instance says
    // into id 2 and only *warns* when it isn't one of the four.
    if let Some(style) = instance.style_map_style_display() {
        builder.add(NameId::SUBFAMILY_NAME, style);
    }
    if let Some(postscript_name) = instance.postscript_name.as_ref() {
        builder.add(NameId::POSTSCRIPT_NAME, postscript_name.clone());
    }
    // The instance's own name parameters, which glyphsLib writes onto the
    // interpolated UFO's fontinfo — so they beat the derived ids, including the
    // 16/17 just added and the three `DROPPED` ones, which come back if the
    // instance states them, and they feed the id 4 and 6 fallbacks.
    for (name_id, value) in instance.overrides.names.iter() {
        builder.add(*name_id, value.clone());
    }

    // Name id 3's fallback is `f"{version};{vendorID};{psName}"` with the
    // *raw* `openTypeOS2VendorID`, trailing spaces and all — only `achVendID`
    // is `ljust`ed. So `IFF ` gives `2.000;IFF ;Teko-Light` and Geom's single
    // space gives `1.102; ;Geom-Regular`, neither of which survives a trim or a
    // round trip through `Tag`. Both are what fontc's own *variable* build of
    // the same source emits, since that hands `NameBuilder` the same string.
    //
    // <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/fontInfoData.py#L178-L185>
    let vendor_id = static_metadata
        .misc
        .raw_vendor_id
        .clone()
        .unwrap_or_else(|| static_metadata.misc.vendor_id.to_string());
    let mut names = builder.build(&vendor_id);
    names.extend(
        static_metadata
            .names
            .iter()
            .filter(|(key, _)| key.lang_id != ENGLISH)
            .map(|(key, value)| (*key, value.clone())),
    );
    // `openTypeNameRecords`, set last and overriding, exactly as the font-level
    // `Name Table Entry` is applied in the frontends
    names.extend(
        instance
            .overrides
            .name_records
            .iter()
            .map(|(key, value)| (*key, value.clone())),
    );
    names
}

/// `fsSelection` for the instance, which is also where `head.macStyle` comes from.
///
/// ufo2ft unions whatever `openTypeOS2Selection` says with exactly one of
/// regular / bold / italic / bold-italic, chosen by `styleMapStyleName`; and it
/// builds macStyle from `styleMapStyleName` alone. IR keeps one set of flags
/// for both, so the RIBBI three are replaced and the rest — USE_TYPO_METRICS,
/// WWS, OBLIQUE, the legacy bits — are kept. A source that put a RIBBI bit in
/// `openTypeOS2Selection` explicitly loses it, which is the price of not
/// keeping the two apart in IR; the family's own style linking would have set
/// the same bit for the default instance anyway.
///
/// With no `styleMapStyleName` the fallback is ufo2ft's: the style name if it
/// is one of the four, else regular.
///
/// A `styleMapStyleName` the source states that *isn't* one of the four — Doto
/// ships `Black` — earns **no** RIBBI bit at all: ufo2ft's `if/elif` chain
/// simply falls off the end, so `fsSelection` keeps only what
/// `openTypeOS2Selection` said and `macStyle` is 0. Note this is not the same
/// as the no-`styleMapStyleName` case, which does fall back to regular; the
/// fallback only runs when the attribute is absent.
///
/// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L714-L725>
/// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L365-L374>
fn pin_selection_flags(
    static_metadata: &StaticMetadata,
    instance: &NamedInstance,
) -> SelectionFlags {
    let ribbi = SelectionFlags::REGULAR | SelectionFlags::BOLD | SelectionFlags::ITALIC;
    let style = match instance.style_map_style_name.as_deref() {
        Some(stated) => StyleMapStyle::parse(stated),
        None => StyleMapStyle::parse(&instance.name).or(Some(StyleMapStyle::Regular)),
    };
    let style = style
        .map(StyleMapStyle::selection_flags)
        .unwrap_or_default();
    (static_metadata.misc.selection_flags - ribbi) | style
}

/// `usWeightClass` and `usWidthClass` from where the pin is in *user* space.
///
/// glyphsLib's `apply_instance_data_to_ufo`, which fontmake runs over every
/// instance it interpolates — for `.designspace` sources as much as for
/// `.glyphs` ones — sets both unconditionally whenever the axis exists,
/// overriding what the masters interpolated and even an explicit
/// `public.fontInfo` override. Only when the axis is *absent* does the
/// interpolated value stand, which is what leaving `misc` alone does here.
///
/// The weight class is `int(user)`: truncation toward zero, not rounding, so
/// an instance at user 442.857 is class **442**. There is no clamp either.
/// The width class is the nearest of the nine standard percentages with ties
/// going to the lower class, which is what [`WidthClass::nearest`] does.
///
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/axes.py#L85-L103>
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/instances.py#L454-L470>
fn pin_os2_classes(pinned: &mut StaticMetadata, axes: &Axes, user_pin: &UserLocation) {
    let user_at = |tag: Tag| {
        axes.get(&tag)
            .map(|axis| user_pin.get(tag).unwrap_or(axis.default).to_f64())
    };
    if let Some(weight) = user_at(Tag::new(b"wght")) {
        pinned.misc.us_weight_class = Some(weight.trunc().clamp(0.0, u16::MAX as f64) as u16);
    }
    if let Some(width) = user_at(Tag::new(b"wdth")) {
        pinned.misc.us_width_class = Some(WidthClass::nearest(width) as u16);
    }
}

/// The fontinfo an interpolated instance loses, and what fills the gaps.
///
/// An instance UFO inherits only what is on ufo2ft's copy whitelist, and then
/// glyphsLib's `apply_instance_data_to_ufo` fills three of the holes with
/// *Glyphs.app's* defaults rather than ufo2ft's. Both happen for
/// `.designspace` sources too — fontmake calls glyphsLib on the instances it
/// interpolates whatever the source was — and neither happens for an ordinary
/// build, so this is `--instance` only.
///
/// - **PANOSE** is merged across the masters rather than copied, see
///   [`MiscMetadata::instance_panose`].
/// - **`fsType`** defaults to Glyphs.app's `[3]` (editable embedding) instead
///   of ufo2ft's `[2]` (preview and print). A `.glyphs` source already carries
///   `[3]` from glyphsLib, so this only moves UFO sources.
/// - **`postscriptUnderlinePosition` / `Thickness`** default to Glyphs.app's
///   flat -100 and 50 instead of ufo2ft's `upem * -0.075` and `upem * 0.05` —
///   the same numbers at 1000 upem and different at any other. Those two are
///   [`GlobalMetric`]s rather than static metadata, so they are handled where
///   the metrics are pinned, in [`GlobalMetricsBuilder::build_pinned`].
///
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/custom_params.py#L1161-L1181>
fn pin_instance_fontinfo(pinned: &mut StaticMetadata, static_metadata: &StaticMetadata) {
    pinned.misc.panose = static_metadata.misc.instance_panose.clone();
    pinned.misc.fs_type = Some(static_metadata.misc.fs_type.unwrap_or(1 << 3));
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

    use fontdrasil::coords::DesignCoord;

    use crate::ir::{
        AnchorKind, Component, ConditionSet, GlobalMetric, KernGroup, KernSide, Substitution,
    };

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

    /// [`mid`] in user space: `wght` runs 400..700, unmapped.
    fn user_mid() -> UserLocation {
        vec![(WGHT, UserCoord::new(550.0))].into()
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

    /// ufo2ft's `Variator` short circuit: a pin on a master copies it, so
    /// masters that could never interpolate together still build there.
    ///
    /// This is the shape of `EpundaSans`' `S.001`, `RobotoSlab`'s `utildeacute`
    /// and the other 14 sources that fail `--instance @default` in fontc but
    /// build fine under `fontmake -i`.
    ///
    /// The midpoint case below is deliberately *not* claimed to match
    /// fontmake. For mismatched **point** counts fontMath raises too —
    /// `_processMathOneContours` indexes `points2[index]` — but for mismatched
    /// **components** it does not: `_pairComponents` matches by base glyph name
    /// and silently drops whatever is left over, so `fontmake -i "Roboto Slab
    /// Medium"` succeeds (measured) where fontc reports the delta error. That
    /// gap is component *pairing*, a separate thing from this short circuit.
    ///
    /// <https://github.com/robotools/fontMath/blob/master/Lib/fontMath/mathGlyph.py#L494-L507>
    /// <https://github.com/robotools/fontMath/blob/master/Lib/fontMath/mathGlyph.py#L653-L682>
    #[test]
    fn pin_glyph_at_an_incompatible_master_is_that_master() {
        let meta = test_static_metadata();
        // three points vs four: no variation model can relate them
        let mut triangle = BezPath::new();
        triangle.move_to((0.0, 0.0));
        triangle.line_to((100.0, 0.0));
        triangle.line_to((50.0, 700.0));
        triangle.close_path();
        let regular_instance = instance(500.0, vec![triangle], Vec::new());
        let bold_instance = instance(600.0, vec![rect(0.0, 0.0, 300.0, 701.0)], Vec::new());
        let glyph = two_master_glyph("S.001", regular_instance.clone(), bold_instance.clone());

        assert_eq!(
            *pin_glyph(&meta, &glyph, &regular())
                .unwrap()
                .default_instance(),
            regular_instance
        );
        assert_eq!(
            *pin_glyph(&meta, &glyph, &bold())
                .unwrap()
                .default_instance(),
            bold_instance
        );

        // between them there is nothing to copy and no way to interpolate
        let e = pin_glyph(&meta, &glyph, &mid()).unwrap_err().to_string();
        assert!(e.contains("same length"), "{e}");
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

    /// A pin that lands on a master takes that master's kerning verbatim —
    /// ufo2ft's `Variator.instance_at` short circuit — so an explicit zero
    /// pair between two ungrouped glyphs survives there, while any other pin
    /// runs the cascade and cleanup, which drops it. Measured on Bona Nova:
    /// the Bold master states ('maqaf-hb', 'kafdagesh-hb') = 0 and
    /// `fontmake -i "Bona Nova Bold"` emits the zero PairPos rule.
    #[test]
    fn pin_kerning_at_a_master_is_that_master_verbatim() {
        let meta = test_static_metadata();
        let groups = kern_groups(&[(KernGroup::Side1("A".into()), &["A"])]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: kerns(&[(kern_pair("A", "V"), -20.0)]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: kerns(&[
                    (kern_pair("A", "V"), -41.0),
                    // explicitly zero, neither glyph in any group
                    (kern_pair("maqaf-hb", "kafdagesh-hb"), 0.0),
                ]),
                groups: Default::default(),
            },
        ];

        // at the Bold master: verbatim, zero pair kept, no resolution
        let pinned = pin_kerning(&meta, instances.iter(), &bold()).unwrap();
        assert_eq!(&pinned.location, meta.default_location());
        assert_eq!(pinned.groups, groups);
        assert_eq!(
            pinned.kerns,
            kerns(&[
                (kern_pair("A", "V"), -41.0),
                (kern_pair("maqaf-hb", "kafdagesh-hb"), 0.0),
            ])
        );

        // off-master: the general path's cleanup drops the ungrouped zero
        let pinned = pin_kerning(&meta, instances.iter(), &mid()).unwrap();
        assert_eq!(
            pinned.kerns.get(&kern_pair("maqaf-hb", "kafdagesh-hb")),
            None
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

    /// Two masters, one of each formatter class, pinned at the midpoint.
    ///
    /// The x-height is the discriminator: it is a `_numberFormatter`
    /// attribute, so 505.5 has to *stay* 505.5. The old pin, which read an
    /// already-built (and therefore already-rounded) `GlobalMetrics`, gave 506
    /// — and, worse, fed 506 to the strikeout-position fallback.
    #[test]
    fn build_pinned_midpoint_rounds_per_attribute() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        for (loc, x_height, ascender) in [(regular(), 500.0, 700.0), (bold(), 511.0, 700.0)] {
            builder.populate_defaults(&loc, 1000, Some(x_height), Some(ascender), None, None);
        }

        let pinned = builder
            .build_pinned(&axes, &mid(), meta.default_location(), &Default::default())
            .unwrap();

        let at = pinned.at(meta.default_location());
        // _numberFormatter: unrounded
        assert_eq!(at.x_height, OrderedFloat(505.5));
        assert_eq!(at.ascender, OrderedFloat(700.0));
        // _integerFormatter, and the reason the line above matters. ufo2ft
        // computes yStrikeoutPosition as otRound(0.6 * xHeight) on the
        // *instance*: 0.6 * 505.5 = 303.3 -> 303. Rounding the masters first
        // gives otRound(0.6 * 500) = 300 and otRound(0.6 * 511) = 307, whose
        // midpoint 303.5 rounds to 304.
        assert_eq!(at.strikeout_position, OrderedFloat(303.0));
        // and the pinned space really is static: same answer anywhere
        assert_eq!(pinned.get(GlobalMetric::XHeight, &bold()), at.x_height);
        // `at` reads every metric, so reaching here says the map is complete;
        // `GlobalMetrics::deltas` unwraps and would have panicked otherwise
        assert!(pinned.iter().count() > 30);
    }

    /// Every formatter class, against numbers `fontmake -i` actually produced.
    ///
    /// Fixture `BlueMatch.designspace` (masters `k = 0` and `k = 101`, so every
    /// value is `base + 50.5` at the midpoint) built with fontmake 3.12.1 as
    /// `fontmake -m BlueMatch.designspace -i`, and read out of the generated
    /// `BlueMatch-Mid.ufo`'s fontinfo.plist. `-250` rather than `-251` is
    /// otRound being `floor(v + 0.5)`, i.e. half toward *positive* infinity
    /// rather than away from zero.
    #[test]
    fn build_pinned_matches_fontmake_per_formatter() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        for (loc, k) in [(regular(), 0.0), (bold(), 101.0)] {
            for (metric, base) in [
                // _integerFormatter
                (GlobalMetric::Os2TypoAscender, 800.0),
                (GlobalMetric::Os2TypoDescender, -200.0 - 2.0 * k),
                (GlobalMetric::Os2TypoLineGap, 0.0),
                (GlobalMetric::HheaDescender, -250.0 - 2.0 * k),
                (GlobalMetric::SubscriptYOffset, 75.0),
                // _nonNegativeIntegerFormatter
                (GlobalMetric::Os2WinAscent, 1000.0),
                // _numberFormatter
                (GlobalMetric::CapHeight, 700.0),
                (GlobalMetric::UnderlinePosition, -100.0 - 2.0 * k),
                (GlobalMetric::UnderlineThickness, 50.0),
            ] {
                builder.set(metric, loc.clone(), base + k);
            }
            builder.populate_defaults(&loc, 1000, Some(500.0 + k), Some(800.0 + k), None, None);
        }

        let pinned = builder
            .build_pinned(&axes, &mid(), meta.default_location(), &Default::default())
            .unwrap();
        let at = pinned.at(meta.default_location());

        // integers, otRounded once
        assert_eq!(at.os2_typo_ascender, OrderedFloat(851.0)); // 850.5
        assert_eq!(at.os2_typo_descender, OrderedFloat(-250.0)); // -250.5
        assert_eq!(at.os2_typo_line_gap, OrderedFloat(51.0)); // 50.5
        assert_eq!(at.hhea_descender, OrderedFloat(-300.0)); // -300.5
        assert_eq!(at.subscript_y_offset, OrderedFloat(126.0)); // 125.5
        assert_eq!(at.os2_win_ascent, OrderedFloat(1051.0)); // 1050.5
        // reals, kept
        assert_eq!(at.ascender, OrderedFloat(850.5));
        assert_eq!(at.cap_height, OrderedFloat(750.5));
        assert_eq!(at.x_height, OrderedFloat(550.5));
        assert_eq!(at.underline_position, OrderedFloat(-150.5));
        assert_eq!(at.underline_thickness, OrderedFloat(100.5));
    }

    /// A pin that lands on a master is that master, exactly.
    #[test]
    fn build_pinned_at_a_master_is_that_master() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        for (loc, x_height) in [(regular(), 500.5), (bold(), 511.25)] {
            builder.set(GlobalMetric::Os2TypoAscender, loc.clone(), 800.0);
            builder.populate_defaults(&loc, 1000, Some(x_height), None, None, None);
        }

        let pinned = builder
            .build_pinned(&axes, &bold(), meta.default_location(), &Default::default())
            .unwrap();

        let at = pinned.at(meta.default_location());
        assert_eq!(at.x_height, OrderedFloat(511.25));
        assert_eq!(at.os2_typo_ascender, OrderedFloat(800.0));
    }

    /// A metric only one master states is an un-normalised partial sum.
    ///
    /// fontMath's `_processMathOne` treats a missing term as the identity, not
    /// as a zero-weighted contribution, so the masters that *do* state an
    /// attribute are simply scaled and added — and their scalars no longer sum
    /// to one. Measured with fontmake 3.12.1: `openTypeHheaAscender = 900` in
    /// the default master and unset in the other gives **450** at the midpoint.
    ///
    /// A metric *no* master states falls back to the densified per-master
    /// values, which is our stand-in for the fallback ufo2ft would compute on
    /// the instance UFO.
    #[test]
    fn build_pinned_partial_declaration_is_a_partial_sum() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        builder.set(GlobalMetric::HheaAscender, regular(), 900.0);
        for loc in [regular(), bold()] {
            builder.populate_defaults(&loc, 1000, None, Some(800.0), None, None);
        }

        let pinned = builder
            .build_pinned(&axes, &mid(), meta.default_location(), &Default::default())
            .unwrap();

        let at = pinned.at(meta.default_location());
        assert_eq!(at.hhea_ascender, OrderedFloat(450.0));
        // nobody stated the line gap, so it interpolates over every master
        assert_eq!(at.hhea_line_gap, OrderedFloat(0.0));
        assert_eq!(at.ascender, OrderedFloat(800.0));
    }

    /// The pin reaches no master that stated the metric: fall back, don't zero.
    #[test]
    fn build_pinned_undeclared_at_the_pin_falls_back() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        // stated only by the master the pin does *not* reach
        builder.set(GlobalMetric::HheaAscender, regular(), 900.0);
        for loc in [regular(), bold()] {
            builder.populate_defaults(&loc, 1000, None, Some(770.0), None, None);
        }

        let pinned = builder
            .build_pinned(&axes, &bold(), meta.default_location(), &Default::default())
            .unwrap();

        // 770 + the computed typo line gap (1200 - 770 - 200 = 230), not 0
        assert_eq!(
            pinned.at(meta.default_location()).hhea_ascender,
            OrderedFloat(1000.0)
        );
    }

    #[test]
    fn pin_static_metadata_is_static() {
        let mut meta = test_static_metadata();
        meta.named_instances = vec![crate::ir::NamedInstance {
            name: "Bold".to_string(),
            postscript_name: None,
            location: vec![(WGHT, UserCoord::new(700.0))].into(),
            ..Default::default()
        }];

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();

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

    /// `usWeightClass` truncates the user coordinate; it does not round it.
    ///
    /// glyphsLib's `user_loc_value_to_class` is `int(user_loc)`. Measured with
    /// fontmake 3.12.1 on a fixture whose `wght` map is `[(100, 0), (900, 7)]`
    /// with the instance at design 3, i.e. user 442.857...: the instance's
    /// `usWeightClass` is **442**, where otRound would have said 443.
    #[test]
    fn pin_weight_class_truncates() {
        let mut meta = test_static_metadata();
        // the fixture's axis, so that user 442.857 is inside it
        meta.axes = Axes::new(vec![Axis {
            min: UserCoord::new(100.0),
            default: UserCoord::new(100.0),
            max: UserCoord::new(900.0),
            converter: fontdrasil::coords::CoordConverter::unmapped(
                UserCoord::new(100.0),
                UserCoord::new(100.0),
                UserCoord::new(900.0),
            ),
            ..wght()
        }]);

        for (user, class) in [(442.857, 442), (443.0, 443), (899.999, 899), (100.0, 100)] {
            let mut pinned = meta.clone();
            pin_os2_classes(
                &mut pinned,
                &meta.axes,
                &vec![(WGHT, UserCoord::new(user))].into(),
            );
            assert_eq!(pinned.misc.us_weight_class, Some(class), "wght={user}");
        }
    }

    /// `usWidthClass` is the nearest of the nine, and a tie goes to the lower.
    ///
    /// Measured with fontmake 3.12.1: an instance at user width **106.25**,
    /// exactly between class 5 (100%) and class 6 (112.5%), comes out class
    /// **5**. The instantiator's own piecewise-linear formula — which
    /// glyphsLib overrides — would have said 6.
    #[test]
    fn pin_width_class_breaks_ties_low() {
        const WDTH: Tag = Tag::new(b"wdth");
        let mut meta = test_static_metadata();
        let wdth = Axis {
            name: "Width".to_string(),
            tag: WDTH,
            min: UserCoord::new(50.0),
            default: UserCoord::new(100.0),
            max: UserCoord::new(200.0),
            hidden: false,
            converter: fontdrasil::coords::CoordConverter::unmapped(
                UserCoord::new(50.0),
                UserCoord::new(100.0),
                UserCoord::new(200.0),
            ),
            localized_names: Default::default(),
        };
        meta.axes = Axes::new(vec![wdth]);

        for (user, class) in [
            (106.25, 5),
            (106.26, 6),
            (100.0, 5),
            (68.75, 2),
            (200.0, 9),
            (50.0, 1),
        ] {
            let mut pinned = meta.clone();
            pin_os2_classes(
                &mut pinned,
                &meta.axes,
                &vec![(WDTH, UserCoord::new(user))].into(),
            );
            assert_eq!(pinned.misc.us_width_class, Some(class), "wdth={user}");
        }
    }

    /// An axis the source doesn't have leaves the interpolated class alone.
    #[test]
    fn pin_classes_only_where_there_is_an_axis() {
        let mut meta = test_static_metadata();
        meta.misc.us_weight_class = Some(501);
        meta.misc.us_width_class = Some(3);

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();

        // there is a wght axis, so the pin decides
        assert_eq!(pinned.misc.us_weight_class, Some(550));
        // there is no wdth axis, so whatever the masters said stands
        assert_eq!(pinned.misc.us_width_class, Some(3));
    }

    fn named(name: &str) -> NamedInstance {
        NamedInstance {
            name: name.to_string(),
            family_name: Some("Fam".to_string()),
            location: vec![(WGHT, UserCoord::new(550.0))].into(),
            ..Default::default()
        }
    }

    /// Style-linked instances put ids 1 and 2 where the source said.
    #[test]
    fn pin_names_uses_the_style_map_names() {
        let mut meta = test_static_metadata();
        meta.names = HashMap::from([
            (NameKey::new(NameId::FAMILY_NAME, "Fam"), "Fam".to_string()),
            (
                NameKey::new(NameId::COPYRIGHT_NOTICE, "(c) Nobody"),
                "(c) Nobody".to_string(),
            ),
        ]);
        let instance = NamedInstance {
            style_map_family_name: Some("Fam".to_string()),
            style_map_style_name: Some(StyleMapStyle::BoldItalic.to_name().to_string()),
            ..named("Bold Italic")
        };

        let names = pin_names(&meta, &instance);

        let get = |id: NameId| {
            names
                .iter()
                .find(|(key, _)| key.name_id == id)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get(NameId::FAMILY_NAME), Some("Fam".to_string()));
        assert_eq!(get(NameId::SUBFAMILY_NAME), Some("Bold Italic".to_string()));
        assert_eq!(get(NameId::FULL_NAME), Some("Fam Bold Italic".to_string()));
        assert_eq!(
            get(NameId::POSTSCRIPT_NAME),
            Some("Fam-BoldItalic".to_string())
        );
        // 16 and 17 match 1 and 2, so ufo2ft drops both
        assert_eq!(get(NameId::TYPOGRAPHIC_FAMILY_NAME), None);
        assert_eq!(get(NameId::TYPOGRAPHIC_SUBFAMILY_NAME), None);
        // the copy whitelist keeps this one
        assert_eq!(
            get(NameId::COPYRIGHT_NOTICE),
            Some("(c) Nobody".to_string())
        );

        assert_eq!(
            pin_selection_flags(&meta, &instance),
            SelectionFlags::BOLD | SelectionFlags::ITALIC
        );
    }

    /// Without style linking, the RIBBI fallback runs on the *style name*.
    #[test]
    fn pin_names_falls_back_for_a_non_ribbi_style() {
        let mut meta = test_static_metadata();
        meta.names = HashMap::from([(NameKey::new(NameId::FAMILY_NAME, "Fam"), "Fam".to_string())]);
        // ufo2ft's OBLIQUE bit is not RIBBI and has to survive
        meta.misc.selection_flags = SelectionFlags::REGULAR | SelectionFlags::USE_TYPO_METRICS;
        let instance = named("SemiBold Condensed");

        let names = pin_names(&meta, &instance);

        let get = |id: NameId| {
            names
                .iter()
                .find(|(key, _)| key.name_id == id)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            get(NameId::FAMILY_NAME),
            Some("Fam SemiBold Condensed".to_string())
        );
        assert_eq!(get(NameId::SUBFAMILY_NAME), Some("Regular".to_string()));
        assert_eq!(
            get(NameId::TYPOGRAPHIC_FAMILY_NAME),
            Some("Fam".to_string())
        );
        assert_eq!(
            get(NameId::TYPOGRAPHIC_SUBFAMILY_NAME),
            Some("SemiBold Condensed".to_string())
        );
        assert_eq!(
            get(NameId::FULL_NAME),
            Some("Fam SemiBold Condensed".to_string())
        );

        assert_eq!(
            pin_selection_flags(&meta, &instance),
            SelectionFlags::REGULAR | SelectionFlags::USE_TYPO_METRICS
        );
    }

    /// The instance loses what ufo2ft's copy whitelist doesn't carry.
    #[test]
    fn pin_drops_the_names_an_instance_cannot_inherit() {
        let mut meta = test_static_metadata();
        for (id, value) in [
            (NameId::COMPATIBLE_FULL_NAME, "Fam Bold"),
            (NameId::WWS_FAMILY_NAME, "Fam"),
            (NameId::WWS_SUBFAMILY_NAME, "Bold"),
            (NameId::UNIQUE_ID, "hand written"),
        ] {
            meta.names
                .insert(NameKey::new(id, value), value.to_string());
        }

        let names = pin_names(&meta, &named("Bold"));

        for id in [
            NameId::COMPATIBLE_FULL_NAME,
            NameId::WWS_FAMILY_NAME,
            NameId::WWS_SUBFAMILY_NAME,
        ] {
            assert!(
                !names.keys().any(|key| key.name_id == id),
                "{id:?} should be gone"
            );
        }
        // the unique id is rebuilt, not inherited
        assert_eq!(
            names
                .iter()
                .find(|(key, _)| key.name_id == NameId::UNIQUE_ID)
                .map(|(_, value)| value.as_str()),
            Some("0.000;NONE;Fam-Bold")
        );
    }

    /// PANOSE, fsType: what the instance path replaces rather than inherits.
    #[test]
    fn pin_instance_only_fontinfo_fallbacks() {
        let mut meta = test_static_metadata();
        meta.misc.panose = Some(crate::ir::Panose::from_digits([
            2, 11, 5, 2, 4, 5, 4, 2, 2, 4,
        ]));
        meta.misc.instance_panose = None; // the masters disagreed
        meta.misc.fs_type = None; // the source said nothing

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();

        assert_eq!(pinned.misc.panose, None);
        assert_eq!(pinned.misc.fs_type, Some(1 << 3));

        // an explicit fsType is not overridden
        meta.misc.fs_type = Some(0);
        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();
        assert_eq!(pinned.misc.fs_type, Some(0));
    }

    /// Name id 3 keeps the vendor id exactly as the source spelled it.
    ///
    /// Only `achVendID` is padded to four bytes; `openTypeOS2VendorID` reaches
    /// the unique id verbatim. Teko states `IFF ` and fontmake writes
    /// `2.000;IFF ;Teko-Light`; Geom states a single space and gets
    /// `1.102; ;Geom-Regular`. Both are what fontc's variable build already
    /// emits, which is what makes a trimmed or `Tag`-padded id a regression.
    #[test]
    fn pin_names_keeps_the_vendor_id_verbatim() {
        for raw in ["IFF ", " ", "NONE"] {
            let mut meta = test_static_metadata();
            meta.names =
                HashMap::from([(NameKey::new(NameId::FAMILY_NAME, "Fam"), "Fam".to_string())]);
            meta.misc.raw_vendor_id = Some(raw.to_string());

            let names = pin_names(&meta, &named("Bold"));

            assert_eq!(
                names
                    .iter()
                    .find(|(key, _)| key.name_id == NameId::UNIQUE_ID)
                    .map(|(_, value)| value.as_str()),
                Some(format!("0.000;{raw};Fam-Bold").as_str()),
                "vendor id {raw:?}"
            );
        }
    }

    /// A `styleMapStyleName` that isn't RIBBI goes to name id 2 untouched.
    ///
    /// Doto's `@default` is `Doto Black`, whose designspace instance states
    /// `stylemapstylename="Black"`. ufo2ft logs "not one of the standard
    /// values" and writes it through anyway; the `fsSelection` if/elif then
    /// falls off the end, so the RIBBI bits stay clear (fontmake emits
    /// `0x0080`, i.e. USE_TYPO_METRICS alone) and `macStyle` is 0.
    #[test]
    fn pin_names_passes_a_non_ribbi_style_map_style_through() {
        let mut meta = test_static_metadata();
        meta.names = HashMap::from([(NameKey::new(NameId::FAMILY_NAME, "Doto"), "Doto".into())]);
        meta.misc.selection_flags = SelectionFlags::REGULAR | SelectionFlags::USE_TYPO_METRICS;
        let instance = NamedInstance {
            style_map_family_name: Some("Doto Black".to_string()),
            style_map_style_name: Some("Black".to_string()),
            family_name: Some("Doto".to_string()),
            ..named("Black")
        };

        let names = pin_names(&meta, &instance);
        let get = |id: NameId| {
            names
                .iter()
                .find(|(key, _)| key.name_id == id)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get(NameId::FAMILY_NAME), Some("Doto Black".to_string()));
        assert_eq!(get(NameId::SUBFAMILY_NAME), Some("Black".to_string()));

        assert_eq!(
            pin_selection_flags(&meta, &instance),
            SelectionFlags::USE_TYPO_METRICS,
            "a non-RIBBI style map style earns no RIBBI bit"
        );
        // ufo2ft lowercases on the way in and title-cases on the way out
        assert_eq!(
            NamedInstance {
                style_map_style_name: Some("BLACK".to_string()),
                ..instance
            }
            .style_map_style_display(),
            Some("Black".to_string())
        );
    }

    /// Two instances may share a style name; each has to match its own location.
    ///
    /// Martian Mono ships four instances called `Regular`, one per width. Going
    /// back through the *name* to find where an instance sits finds the first
    /// of them every time, so the `@default` pin matched none of them and fell
    /// back to the family's names — family `Martian Mono` instead of
    /// `Martian Mono SemiExpanded`, and a PostScript name to match.
    #[test]
    fn named_instance_at_tells_repeated_style_names_apart() {
        let mut meta = test_static_metadata();
        meta.named_instances = vec![
            NamedInstance {
                family_name: Some("Fam Condensed".to_string()),
                location: vec![(WGHT, UserCoord::new(400.0))].into(),
                ..named("Regular")
            },
            NamedInstance {
                family_name: Some("Fam SemiExpanded".to_string()),
                location: vec![(WGHT, UserCoord::new(700.0))].into(),
                ..named("Regular")
            },
        ];

        let at = named_instance_at(&meta, &InstanceSpec::Location(user_bold()), &bold());
        assert_eq!(
            at.and_then(|instance| instance.family_name.as_deref()),
            Some("Fam SemiExpanded")
        );
    }

    fn user_bold() -> UserLocation {
        vec![(WGHT, UserCoord::new(700.0))].into()
    }

    /// The instance's own PANOSE replaces the merged-across-masters one.
    ///
    /// This is the single biggest instancing gap in the corpus: 44 targets
    /// differ in OS/2 alone because fontc used `instance_panose` — the
    /// element-wise merge that zeroes any digit the masters disagree about —
    /// where glyphsLib replays the instance's `panose` parameter over it.
    #[test]
    fn pin_instance_overrides_replace_the_merged_panose() {
        let mut meta = test_static_metadata();
        meta.misc.instance_panose = Some(crate::ir::Panose::from_digits([
            2, 11, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let stated = crate::ir::Panose::from_digits([2, 11, 5, 2, 4, 5, 4, 2, 2, 4]);
        let instance = NamedInstance {
            overrides: crate::ir::InstanceOverrides {
                panose: Some(stated.clone()),
                ..Default::default()
            },
            ..named("Regular")
        };
        meta.named_instances = vec![instance.clone()];

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), Some(&instance)).unwrap();

        assert_eq!(pinned.misc.panose, Some(stated));
        // without an instance the merge still stands
        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();
        assert_eq!(pinned.misc.panose, meta.misc.instance_panose);
    }

    /// The rest of the OS/2-ish parameters, and `Don't use Production Names`.
    #[test]
    fn pin_instance_overrides_the_rest_of_os2() {
        let mut meta = test_static_metadata();
        meta.misc.fs_type = Some(0);
        meta.misc.selection_flags = SelectionFlags::REGULAR;
        meta.postscript_names = Some(HashMap::from([("a".into(), "uni0061".into())]));
        let instance = NamedInstance {
            overrides: crate::ir::InstanceOverrides {
                fs_type: Some(1 << 2),
                is_fixed_pitch: Some(true),
                unicode_range_bits: Some(HashSet::from([0, 1])),
                meta_table: Some(crate::ir::MetaTableValues {
                    dlng: vec!["Latn".into()],
                    slng: vec!["Latn".into()],
                }),
                use_typo_metrics: Some(true),
                // `Don't use Production Names = 1` negates to this
                use_production_names: Some(false),
                ..Default::default()
            },
            ..named("Regular")
        };

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), Some(&instance)).unwrap();

        assert_eq!(pinned.misc.fs_type, Some(1 << 2));
        assert_eq!(pinned.misc.is_fixed_pitch, Some(true));
        assert_eq!(pinned.misc.unicode_range_bits, Some(HashSet::from([0, 1])));
        assert_eq!(
            pinned
                .misc
                .meta_table
                .as_ref()
                .map(|meta| meta.dlng.clone()),
            Some(vec!["Latn".into()])
        );
        assert!(
            pinned
                .misc
                .selection_flags
                .contains(SelectionFlags::USE_TYPO_METRICS)
        );
        assert_eq!(pinned.postscript_names, None, "development names, then");
    }

    /// `Name Table Entry` overrides; `preferredFamilyName` feeds the fallbacks.
    #[test]
    fn pin_names_applies_the_instance_name_parameters() {
        let mut meta = test_static_metadata();
        meta.names = HashMap::from([(NameKey::new(NameId::FAMILY_NAME, "Fam"), "Fam".to_string())]);
        let instance = NamedInstance {
            overrides: crate::ir::InstanceOverrides {
                names: BTreeMap::from([
                    (NameId::TYPOGRAPHIC_FAMILY_NAME, "Preferred".to_string()),
                    (NameId::WWS_FAMILY_NAME, "Wws".to_string()),
                ]),
                // Epilogue's `25; EpilogueRoman`
                name_records: BTreeMap::from([(
                    NameKey::new(NameId::new(25), "FamRoman"),
                    "FamRoman".to_string(),
                )]),
                ..Default::default()
            },
            ..named("Bold")
        };

        let names = pin_names(&meta, &instance);
        let get = |id: NameId| {
            names
                .iter()
                .find(|(key, _)| key.name_id == id)
                .map(|(_, v)| v.clone())
        };
        // `preferredFamilyName` beats the instance's own family name and feeds
        // the id 1/4/6 fallbacks, exactly as `openTypeNamePreferredFamilyName`
        // does for an ordinary static UFO
        assert_eq!(get(NameId::FAMILY_NAME), Some("Preferred".to_string()));
        assert_eq!(get(NameId::FULL_NAME), Some("Preferred Bold".to_string()));
        assert_eq!(
            get(NameId::POSTSCRIPT_NAME),
            Some("Preferred-Bold".to_string())
        );
        // and then 16/17 match 1/2, so ufo2ft drops them
        assert_eq!(get(NameId::TYPOGRAPHIC_FAMILY_NAME), None);
        // 21/22 are dropped unless the instance states them
        assert_eq!(get(NameId::WWS_FAMILY_NAME), Some("Wws".to_string()));
        assert_eq!(get(NameId::new(25)), Some("FamRoman".to_string()));
    }

    /// The font's Variations PostScript Name Prefix does not reach a static
    /// instance: glyphsLib maps `variationsPostScriptNamePrefix` to nothing, so
    /// fontmake never writes id 25 into one. Measured on
    /// `googlefonts/googlesans-code GoogleSansCode` and
    /// `mozilla/mozilla-text-type MozillaText`, whose only `--instance @default`
    /// diff was fontc's extra id 25 (`GoogleSansCode` / `MozillaTextVF`).
    #[test]
    fn pin_names_drops_the_variations_postscript_prefix() {
        let mut meta = test_static_metadata();
        meta.names = HashMap::from([
            (NameKey::new(NameId::FAMILY_NAME, "Fam"), "Fam".to_string()),
            (
                NameKey::new(NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX, "FamVF"),
                "FamVF".to_string(),
            ),
        ]);
        let names = pin_names(&meta, &named("Bold"));
        assert!(
            !names
                .keys()
                .any(|key| key.name_id == NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX),
            "id 25 survived: {names:?}"
        );

        // ...but an explicit `Name Table Entry` for id 25 still does
        let instance = NamedInstance {
            overrides: crate::ir::InstanceOverrides {
                name_records: BTreeMap::from([(
                    NameKey::new(NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX, "FamRoman"),
                    "FamRoman".to_string(),
                )]),
                ..Default::default()
            },
            ..named("Bold")
        };
        let names = pin_names(&meta, &instance);
        assert_eq!(
            names
                .iter()
                .find(|(key, _)| key.name_id == NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX)
                .map(|(_, value)| value.clone()),
            Some("FamRoman".to_string())
        );
    }

    /// A metric the instance states replaces the interpolated one outright.
    #[test]
    fn build_pinned_prefers_an_instance_override() {
        let meta = test_static_metadata();
        let axes = Axes::new(vec![wght()]);
        let mut builder = GlobalMetricsBuilder::new();
        builder.set(GlobalMetric::HheaAscender, regular(), 900.0);
        builder.set(GlobalMetric::HheaAscender, bold(), 1000.0);
        for loc in [regular(), bold()] {
            builder.populate_defaults(&loc, 1000, None, Some(800.0), None, None);
        }
        let overrides = BTreeMap::from([(GlobalMetric::HheaAscender, OrderedFloat(1234.0))]);

        let pinned = builder
            .build_pinned(&axes, &mid(), meta.default_location(), &overrides)
            .unwrap();

        assert_eq!(
            pinned.at(meta.default_location()).hhea_ascender,
            OrderedFloat(1234.0)
        );
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
        // both masters name themselves; the instance gets neither, because
        // fontMath derives the weight name and neither master states a class
        assert_eq!(pinned.weight_name, None);
        assert_eq!(pinned.full_name, None);
    }

    /// fontMath derives the CFF `Weight` from the *interpolated*
    /// `openTypeOS2WeightClass`, not from either master's `postscriptWeightName`.
    ///
    /// Every expectation below is from `fontmake -o otf -i <instance>
    /// --keep-overlaps --optimize-cff 1` on a two-master designspace whose
    /// masters state weight classes 100 and 900 over wght 400..700:
    ///
    /// | user wght | interpolated class | fontmake `Weight` |
    /// |---|---|---|
    /// | 550    | 500   | Medium      |
    /// | 475    | 300   | Light       |
    /// | 418.75 | 150   | Extra-light |
    /// | 456.25 | 250   | Light       |
    /// | 418.6  | 149.6 | Thin        |
    ///
    /// The last two are the interesting ones: 250 -> "Light" is Python 2
    /// rounding (Python 3 would say 200, "Extra-light"), and 149.6 -> "Thin"
    /// shows the *unrounded* class is what gets rounded to the nearest 100 —
    /// `MathInfo.round()` would have made it 150 and then "Extra-light", but
    /// fontmake's `round_instances` defaults to False.
    #[test]
    fn pin_postscript_derives_the_weight_name_from_the_interpolated_class() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    weight_name: Some("Thin".to_string()),
                    os2_weight_class: Some(OrderedFloat(100.0)),
                    ..Default::default()
                },
            ),
            (
                bold(),
                PostscriptSettings {
                    weight_name: Some("Black".to_string()),
                    os2_weight_class: Some(OrderedFloat(900.0)),
                    ..Default::default()
                },
            ),
        ]);

        let at = |t: f64| pin_postscript(&meta, &NormalizedLocation::for_pos(&[("wght", t)]));
        // wght 400..700, so these are (user - 400) / 300
        assert_eq!(at(0.5).os2_weight_class, Some(OrderedFloat(500.0)));
        assert_eq!(at(0.5).weight_name.as_deref(), Some("Medium"));
        assert_eq!(at(0.25).weight_name.as_deref(), Some("Light"));
        assert_eq!(
            at(18.75 / 300.0).weight_name.as_deref(),
            Some("Extra-light")
        );
        assert_eq!(at(56.25 / 300.0).weight_name.as_deref(), Some("Light"));
        assert_eq!(at(18.6 / 300.0).weight_name.as_deref(), Some("Thin"));
        // a pin that *is* a master does no math at all, so fontMath never
        // derives a name and fontmake writes no `Weight`: measured at wght 400
        // and wght 700 on the same designspace
        assert_eq!(at(0.0).weight_name, None);
        assert_eq!(at(1.0).weight_name, None);
    }

    /// A `.glyphs` source states no per-master weight class, so there is
    /// nothing to derive from and the instance gets no CFF `Weight` — which is
    /// what fontmake does too.
    #[test]
    fn pin_postscript_without_weight_classes_writes_no_weight() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    weight_name: Some("Thin".to_string()),
                    ..Default::default()
                },
            ),
            (
                bold(),
                PostscriptSettings {
                    weight_name: Some("Black".to_string()),
                    ..Default::default()
                },
            ),
        ]);

        let pinned = pin_postscript(&meta, &mid());
        assert_eq!(pinned.os2_weight_class, None);
        assert_eq!(pinned.weight_name, None);
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
        // and these two, which a *multi*-master instance never gets
        assert_eq!(pinned.weight_name, None);
        assert_eq!(pinned.full_name, None);
    }

    /// One info master pinned at the default: ufo2ft copies its whole fontinfo,
    /// so `postscriptWeightName` and `postscriptFullName` survive. Measured on
    /// `docrepair-fonts/caprasimo-fonts Caprasimo-Regular.designspace`, a
    /// one-source designspace whose only `--instance @default --flavor otf`
    /// diff was fontmake's CFF `<Weight value="Regular"/>`.
    #[test]
    fn pin_postscript_one_master_at_default_keeps_the_whole_fontinfo() {
        let mut meta = test_static_metadata_at(HashSet::from([regular()]));
        meta.postscript = HashMap::from([(
            regular(),
            PostscriptSettings {
                blue_values: floats(&[-10.0, 0.0]),
                weight_name: Some("Regular".to_string()),
                full_name: Some("Caprasimo Regular".to_string()),
                ..Default::default()
            },
        )]);

        let pinned = pin_postscript(&meta, &regular());

        assert_eq!(pinned.weight_name, Some("Regular".to_string()));
        assert_eq!(pinned.full_name, Some("Caprasimo Regular".to_string()));
        assert_eq!(pinned.blue_values, floats(&[-10.0, 0.0]));
    }

    /// ...but two masters is two masters, even where one of them is silent.
    #[test]
    fn pin_postscript_two_masters_still_lose_the_names() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                PostscriptSettings {
                    weight_name: Some("Regular".to_string()),
                    full_name: Some("Fam Regular".to_string()),
                    ..Default::default()
                },
            ),
            (bold(), PostscriptSettings::default()),
        ]);

        let pinned = pin_postscript(&meta, &regular());

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

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();

        assert_eq!(
            pinned.postscript.keys().collect::<Vec<_>>(),
            vec![meta.default_location()]
        );
        assert_eq!(
            pinned.postscript_default().blue_values,
            floats(&[-15.0, 0.0, 550.0, 565.0, 750.0, 765.0])
        );
    }

    /// The instance's own `postscriptFullName` is the CFF `FullName`, beating
    /// the family-plus-style fallback the backend would otherwise build.
    /// Measured on `Omnibus-Type/Saira_Stencil SairaStencil-Italic`, whose
    /// `Thin Italic` instance carries `postscriptFullName` =
    /// `SairaStencilThinItalic` and whose only `--flavor otf` diff was fontc's
    /// `Saira Stencil Thin Italic`.
    #[test]
    fn pin_static_metadata_applies_the_instance_postscript_full_name() {
        let mut meta = test_static_metadata();
        meta.postscript = blue_match();
        let instance = NamedInstance {
            overrides: crate::ir::InstanceOverrides {
                postscript_full_name: Some("FamThinItalic".to_string()),
                ..Default::default()
            },
            ..named("Thin Italic")
        };

        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), Some(&instance)).unwrap();

        assert_eq!(
            pinned.postscript_default().full_name.as_deref(),
            Some("FamThinItalic")
        );
        // and an instance that says nothing still gets nothing
        let pinned = pin_static_metadata(&meta, &mid(), &user_mid(), Some(&named("Thin"))).unwrap();
        assert_eq!(pinned.postscript_default().full_name, None);
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

        let pinned_meta = pin_static_metadata(&meta, &mid(), &user_mid(), None).unwrap();
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

    // ---- feature variation rules ----

    fn design(pos: f64) -> DesignLocation {
        vec![(WGHT, fontdrasil::coords::DesignCoord::new(pos))].into()
    }

    fn variations(rules: Vec<Rule>) -> VariableFeature {
        VariableFeature {
            features: vec![Tag::new(b"rvrn")],
            rules,
        }
    }

    fn swaps_at(variations: &VariableFeature, pos: f64) -> Vec<(String, String)> {
        rule_swaps(variations, &design(pos), |_| true)
            .into_iter()
            .map(|(replace, with)| (replace.to_string(), with.to_string()))
            .collect()
    }

    /// `designspaceLib.evaluateConditions` includes both bounds, and treats a
    /// condition that states only one of them as unbounded on the other side.
    #[test]
    fn a_rule_range_includes_its_ends() {
        let closed = variations(vec![Rule::for_test(
            &[&[("wght", (550.0, 700.0))]],
            &[("a", "b")],
        )]);
        assert_eq!(swaps_at(&closed, 549.999), Vec::new());
        assert_eq!(
            swaps_at(&closed, 550.0),
            vec![("a".to_string(), "b".to_string())],
            "the minimum is inside the range"
        );
        assert_eq!(
            swaps_at(&closed, 700.0),
            vec![("a".to_string(), "b".to_string())],
            "and so is the maximum"
        );
        assert_eq!(swaps_at(&closed, 700.001), Vec::new());

        let open = variations(vec![Rule {
            conditions: vec![
                [Condition::new(WGHT, Some(DesignCoord::new(550.0)), None)]
                    .into_iter()
                    .collect(),
            ],
            substitutions: vec![Substitution {
                replace: "a".into(),
                with: "b".into(),
            }],
        }]);
        assert_eq!(swaps_at(&open, 549.0), Vec::new());
        assert_eq!(swaps_at(&open, 550.0).len(), 1);
        assert_eq!(swaps_at(&open, 10_000.0).len(), 1, "no maximum, no ceiling");
    }

    /// Condition sets are OR-ed, conditions within one AND-ed, which makes a
    /// rule with no condition sets dead and a condition set with no conditions
    /// unconditional.
    #[test]
    fn a_rule_needs_one_whole_condition_set() {
        let two_sets = variations(vec![Rule::for_test(
            &[&[("wght", (400.0, 450.0))], &[("wght", (650.0, 700.0))]],
            &[("a", "b")],
        )]);
        assert_eq!(swaps_at(&two_sets, 425.0).len(), 1);
        assert_eq!(swaps_at(&two_sets, 500.0).len(), 0);
        assert_eq!(swaps_at(&two_sets, 675.0).len(), 1);

        let no_sets = variations(vec![Rule {
            conditions: Vec::new(),
            substitutions: vec![Substitution {
                replace: "a".into(),
                with: "b".into(),
            }],
        }]);
        assert_eq!(swaps_at(&no_sets, 500.0).len(), 0, "nothing to satisfy");

        let empty_set = variations(vec![Rule {
            conditions: vec![ConditionSet::from_iter([])],
            substitutions: vec![Substitution {
                replace: "a".into(),
                with: "b".into(),
            }],
        }]);
        assert_eq!(swaps_at(&empty_set, 500.0).len(), 1, "nothing to fail");
    }

    /// Rule order, then substitution order, and the list is not deduplicated
    /// or resolved: `a -> b` then `b -> c` is two swaps, not one.
    #[test]
    fn swaps_come_out_in_rule_then_sub_order() {
        let chain = variations(vec![
            Rule::for_test(&[&[("wght", (550.0, 700.0))]], &[("a", "b"), ("x", "y")]),
            Rule::for_test(&[&[("wght", (600.0, 700.0))]], &[("b", "c")]),
        ]);
        assert_eq!(
            swaps_at(&chain, 550.0),
            vec![
                ("a".to_string(), "b".to_string()),
                ("x".to_string(), "y".to_string())
            ],
            "only the first rule fires"
        );
        assert_eq!(
            swaps_at(&chain, 650.0),
            vec![
                ("a".to_string(), "b".to_string()),
                ("x".to_string(), "y".to_string()),
                ("b".to_string(), "c".to_string())
            ],
        );
    }

    /// A rule naming a glyph the source hasn't got is a no-op, and a glyph
    /// swapped with itself is nothing at all.
    #[test]
    fn a_rule_on_a_glyph_that_is_not_there_does_nothing() {
        let rules = variations(vec![Rule::for_test(
            &[&[("wght", (400.0, 700.0))]],
            &[("missing", "b"), ("a", "a"), ("a", "b")],
        )]);
        let swaps = rule_swaps(&rules, &design(500.0), |name| name.as_str() != "missing");
        assert_eq!(
            swaps,
            vec![(GlyphName::from("a"), GlyphName::from("b"))],
            "the substitute is not filtered here: a missing one is the caller's error"
        );
    }

    #[test]
    fn a_swap_exchanges_drawings_and_widths_but_not_heights() {
        let mut a = GlyphInstance {
            width: 100.0,
            height: Some(700.0),
            vertical_origin: Some(800.0),
            contours: vec![rect(0.0, 0.0, 10.0, 10.0)],
            components: vec![Component {
                base: "x".into(),
                transform: Affine::IDENTITY,
                anchor: None,
            }],
        };
        let mut b = GlyphInstance {
            width: 200.0,
            height: Some(900.0),
            vertical_origin: Some(1000.0),
            contours: vec![rect(0.0, 0.0, 20.0, 20.0)],
            components: Vec::new(),
        };

        swap_geometry(&mut a, &mut b);

        assert_eq!((a.width, b.width), (200.0, 100.0));
        assert_eq!(a.contours, vec![rect(0.0, 0.0, 20.0, 20.0)]);
        assert_eq!(b.contours, vec![rect(0.0, 0.0, 10.0, 10.0)]);
        assert!(a.components.is_empty());
        assert_eq!(b.components.len(), 1);
        // ufo2ft's swap is a point pen plus `width`; nothing else moves
        assert_eq!((a.height, b.height), (Some(700.0), Some(900.0)));
        assert_eq!(
            (a.vertical_origin, b.vertical_origin),
            (Some(800.0), Some(1000.0))
        );
    }

    #[test]
    fn a_swap_remaps_component_references_both_ways() {
        let mut glyph = instance(
            100.0,
            Vec::new(),
            ["a", "b", "c"]
                .into_iter()
                .map(|base| Component {
                    base: base.into(),
                    transform: Affine::IDENTITY,
                    anchor: None,
                })
                .collect(),
        );

        assert!(swap_component_bases(&mut glyph, &"a".into(), &"b".into()));
        assert_eq!(
            glyph
                .components
                .iter()
                .map(|c| c.base.as_str())
                .collect::<Vec<_>>(),
            ["b", "a", "c"]
        );
        assert!(
            !swap_component_bases(&mut glyph, &"y".into(), &"z".into()),
            "a swap of glyphs nobody references changes nothing"
        );
    }

    /// The kerning half of a swap: literal pair keys on either side, and group
    /// *membership* — never the group's name.
    ///
    /// Numbers from `fontmake -m Chain.designspace -i`: a pair `('A','C')` and
    /// groups `kern1.A = [A]`, `kern2.C = [C]` come out of the swaps
    /// `A <-> B` then `B <-> C` as `('C','B')`, `kern1.A = [C]`,
    /// `kern2.C = [B]`.
    #[test]
    fn a_swap_renames_kern_pairs_and_group_members() {
        let mut kerning = KerningInstance {
            location: regular(),
            kerns: kerns(&[(kern_pair("A", "C"), -60.0)]),
            groups: kern_groups(&[
                (KernGroup::Side1("A".into()), &["A"]),
                (KernGroup::Side2("C".into()), &["C"]),
            ]),
        };

        swap_kerning(&mut kerning, &"A".into(), &"B".into());
        swap_kerning(&mut kerning, &"B".into(), &"C".into());

        assert_eq!(kerning.kerns, kerns(&[(kern_pair("C", "B"), -60.0)]));
        assert_eq!(
            kerning.groups,
            kern_groups(&[
                (KernGroup::Side1("A".into()), &["C"]),
                (KernGroup::Side2("C".into()), &["B"]),
            ]),
        );
    }

    #[test]
    fn a_swap_leaves_group_sides_alone() {
        let mut kerning = KerningInstance {
            location: regular(),
            kerns: kerns(&[(
                (
                    KernSide::Group(KernGroup::Side1("A".into())),
                    KernSide::Glyph("A".into()),
                ),
                -10.0,
            )]),
            groups: Default::default(),
        };

        swap_kerning(&mut kerning, &"A".into(), &"B".into());

        assert_eq!(
            kerning.kerns.keys().next(),
            Some(&(
                KernSide::Group(KernGroup::Side1("A".into())),
                KernSide::Glyph("B".into()),
            )),
            "the group named for A is still named for A"
        );
    }
}
