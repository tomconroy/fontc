//! Pinning the frontend's IR at one instance.
//!
//! [`fontir::instance`] knows how to pin one IR value; this module knows what
//! the whole [`Context`] has to be rewritten to, and when.
//!
//! # Why the pin is a barrier and not a work item
//!
//! Every consumer of variable IR would have to declare a dependency on a pin
//! *job* for it to be safe, and there are around thirty of them: the scheduler
//! satisfies `Variant` dependencies from a per-discriminant counter, so a new
//! work id with its own discriminant makes nobody wait. `Access::All` is no
//! help either — a job that reads everything only becomes runnable when it is
//! the last one pending, which is the opposite of a barrier. So the pin runs
//! on the orchestrator thread, at a point where the scheduler has shown us
//! that nothing else can run. See `Workload::advance_pin`.
//!
//! # Why it is two barriers
//!
//! fontmake interpolates the instance and *then* runs its filters, so the pin
//! has to precede `GlyphOrderWork` — which decomposes, flattens, propagates
//! anchors and synthesizes `.notdef`. Decomposition especially: applying a
//! component's transform and then interpolating is not the same number as
//! interpolating and then applying, because the product is bilinear in the
//! transform and the base outline.
//!
//! Kerning cannot be pinned there, though. A `KernInstance` work reads
//! `GlyphOrder` (it validates kern participants against it), so kerning IR
//! does not exist until after the work we are trying to precede. Hence
//! [`pin_frontend`] before glyph order, and [`pin_kerning`] once every
//! `KernInstance` has run — by which time `StaticMetadata` is already static,
//! which is why the caller has to hand the original back.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use fontdrasil::{
    coords::{DesignLocation, NormalizedLocation, UserLocation},
    orchestration::Access,
    types::GlyphName,
};
use fontir::{
    instance::{self, InstanceSpec},
    ir::{FeaturesSource, Glyph, GlyphAnchors, GlyphInstance, KerningLocations, StaticMetadata},
    orchestration::{Context, WorkId},
};

use crate::Error;

/// Everything the pin needs that the pin itself destroys.
///
/// [`pin_frontend`] empties `StaticMetadata::axes`, so the second phase can no
/// longer work out which axes the kerning masters vary on, nor where the pin
/// was. It gets told.
pub(crate) struct Pin {
    /// The frontend's metadata as it was before the pin.
    pub(crate) source_metadata: Arc<StaticMetadata>,
    /// Where we pinned, in normalized space.
    pub(crate) location: NormalizedLocation,
    /// The glyph swaps the source's feature variation rules asked for here,
    /// in the order they were applied to glyphs.
    ///
    /// Phase 1 has already done them to glyphs, anchors and components; the
    /// kerning they also touch does not exist until phase 2, which repeats
    /// them there. `variations` is gone from the pinned metadata by then, and
    /// so are the axes its conditions are written on, so the answer travels
    /// rather than being recomputed.
    pub(crate) swaps: Vec<(GlyphName, GlyphName)>,
}

/// Rewrite everything but kerning as the instance `spec` asks for.
///
/// Must run after the frontend's own work and before `GlyphOrderWork`.
pub(crate) fn pin_frontend(fe_root: &Context, spec: &InstanceSpec) -> Result<Pin, Error> {
    let context = fe_root.copy_for_work(Access::All, Access::All);
    let source_metadata = context.static_metadata.get();
    let asked_for = resolve_user(&source_metadata, spec)?;
    let location = normalize(&source_metadata, &asked_for)?;

    if let Some(features) = context.features.try_get()
        && fea_declares_a_conditionset(&features)?
    {
        return Err(Error::InstanceOfSourceWithFeaConditionSet);
    }

    log::info!("Pinning at {location:?}");

    let mut glyph_names = HashSet::new();
    for (_, glyph) in context.glyphs.all() {
        let pinned = instance::pin_glyph(&source_metadata, &glyph, &location)
            .map_err(fontir::error::Error::from)?;
        glyph_names.insert(pinned.name.clone());
        context.glyphs.set(pinned);
    }
    for (_, anchors) in context.anchors.all() {
        let pinned = instance::pin_anchors(&source_metadata, &anchors, &location)
            .map_err(fontir::error::Error::from)?;
        context.anchors.set(pinned);
    }
    if let Some(metrics) = context.global_metrics.try_get() {
        context.global_metrics.set(instance::pin_global_metrics(
            &source_metadata,
            &metrics,
            &location,
        )?);
    }

    // Feature variation rules become plain glyph swaps in a static instance,
    // and they run *after* interpolation — on the glyphs we just pinned — the
    // way ufo2ft applies them to a finished instance. Before
    // `pin_static_metadata`, which drops both the rules and the axes their
    // conditions are written on.
    let swaps = match source_metadata.variations.as_ref() {
        Some(variations) => {
            let design = design_pin(&source_metadata, &asked_for);
            let swaps =
                instance::rule_swaps(variations, &design, |name| glyph_names.contains(name));
            apply_swaps(&context, &swaps)?;
            swaps
        }
        None => Vec::new(),
    };

    // last, because everything above reads the *source's* axes to interpolate
    let mut pinned = instance::pin_static_metadata(&source_metadata, &location)?;
    prune_variable_names(&mut pinned, &source_metadata);
    context.static_metadata.set(pinned);

    Ok(Pin {
        source_metadata,
        location,
        swaps,
    })
}

/// Do the rules' glyph swaps, in order, each on the result of the last.
///
/// Sequential by construction: `a -> b` then `b -> c` is the 3-cycle
/// `(a b)(b c)`, not `a -> c`. See [`instance::rule_swaps`].
fn apply_swaps(context: &Context, swaps: &[(GlyphName, GlyphName)]) -> Result<(), Error> {
    for (old, new) in swaps {
        swap_glyphs(context, old, new)?;
    }
    Ok(())
}

/// ufo2ft's `swap_glyph_names`, minus the kerning, which isn't built yet.
///
/// Both glyphs survive, with their own name, codepoints and place in glyph
/// order; what moves is what they draw, how wide they are, their anchors, and
/// which of them everyone else's components point at.
fn swap_glyphs(context: &Context, old: &GlyphName, new: &GlyphName) -> Result<(), Error> {
    let (Some(old_glyph), Some(new_glyph)) = (
        context.try_get_glyph(old.clone()),
        context.try_get_glyph(new.clone()),
    ) else {
        // `rule_swaps` has already checked the glyph being replaced, so it is
        // the substitute that is missing. ufo2ft raises here too, deliberately
        // rather than dropping the rule.
        return Err(Error::InstanceRuleSubstituteMissing {
            replace: old.to_string(),
            with: new.to_string(),
        });
    };

    // 1. what they draw, and how wide they are. Both glyphs are pinned by
    //    now, so this is one instance each, at the same location.
    let mut old_sources = old_glyph.sources().clone();
    let mut new_sources = new_glyph.sources().clone();
    for (at, old_instance) in old_sources.iter_mut() {
        let Some(new_instance) = new_sources.get_mut(at) else {
            continue;
        };
        instance::swap_geometry(old_instance, new_instance);
    }
    context.glyphs.set(republish(&old_glyph, old_sources)?);
    context.glyphs.set(republish(&new_glyph, new_sources)?);

    // 2. their anchors
    let anchors_of = |name: &GlyphName| {
        context
            .anchors
            .try_get(&WorkId::Anchor(name.clone()))
            .map(|anchors| anchors.anchors.clone())
            .unwrap_or_default()
    };
    let (old_anchors, new_anchors) = (anchors_of(old), anchors_of(new));
    if !old_anchors.is_empty() || !new_anchors.is_empty() {
        context
            .anchors
            .set(GlyphAnchors::new(old.clone(), new_anchors));
        context
            .anchors
            .set(GlyphAnchors::new(new.clone(), old_anchors));
    }

    // 3. every component reference in the font, including the two glyphs we
    //    just swapped, which now hold each other's components
    for (_, glyph) in context.glyphs.all() {
        let mut sources = glyph.sources().clone();
        let mut changed = false;
        for instance in sources.values_mut() {
            // every source, so no short-circuiting the remapping
            changed |= instance::swap_component_bases(instance, old, new);
        }
        if changed {
            context.glyphs.set(republish(&glyph, sources)?);
        }
    }

    Ok(())
}

/// `glyph` with new drawings, rebuilt rather than mutated.
///
/// [`Glyph`] caches whether its components' 2x2 transforms are consistent and
/// whether any of them overflow, and a swap can change both answers.
fn republish(
    glyph: &Glyph,
    sources: HashMap<NormalizedLocation, GlyphInstance>,
) -> Result<Glyph, Error> {
    Glyph::new(
        glyph.name.clone(),
        glyph.emit_to_binary,
        glyph.codepoints.clone(),
        sources,
    )
    .map_err(|e| fontir::error::Error::from(e).into())
}

/// Reduce every source's kerning to the one instance at the pin.
///
/// Must run once every `KernInstance` work has completed. Listing only the pin
/// in [`KerningLocations`] is what makes the leftover instances at the master
/// locations invisible — the backend resolves pairs at the locations that
/// struct names and ignores the rest — so there is no need for a way to delete
/// them.
pub(crate) fn pin_kerning(fe_root: &Context, pin: &Pin) -> Result<(), Error> {
    let context = fe_root.copy_for_work(Access::All, Access::All);
    // no kerning at all (--skip-features, or a source that doesn't kern)
    let Some(locations) = context.kerning_locations.try_get() else {
        return Ok(());
    };
    if locations.locations.is_empty() {
        return Ok(());
    }

    let instances: Vec<_> = locations
        .locations
        .iter()
        .filter_map(|at| {
            context
                .kerning_at
                .try_get(&WorkId::KernInstance(at.clone()))
        })
        .collect();
    let mut pinned = instance::pin_kerning(
        &pin.source_metadata,
        instances.iter().map(|instance| instance.as_ref()),
        &pin.location,
    )?;
    // The other half of phase 1's glyph swaps. ufo2ft swaps kerning as part of
    // the same operation, on a font whose kerning has already been
    // interpolated, which is exactly here: interpolate, then swap.
    for (old, new) in &pin.swaps {
        instance::swap_kerning(&mut pinned, old, new);
    }

    let locations = BTreeSet::from([pinned.location.clone()]);
    context.kerning_at.set(pinned);
    context
        .kerning_locations
        .set(KerningLocations { locations });
    Ok(())
}

/// The user-space location `spec` names, on every axis, or why it names none.
///
/// Axes the user didn't mention take their default, so the result is a
/// complete location and not the subset that was asked for.
fn resolve_user(
    static_metadata: &StaticMetadata,
    spec: &InstanceSpec,
) -> Result<UserLocation, Error> {
    // fontc's universal "this is static" test. Point axes are fine: fontmake
    // pins a single-master designspace without complaint, and only rejects
    // `-i` outright for bare UFO input.
    if static_metadata.axes.is_empty() {
        return Err(Error::InstanceRequiresVariableSource);
    }

    let asked_for = match spec {
        InstanceSpec::Named(name) => static_metadata
            .named_instances
            .iter()
            .find(|instance| instance.name == *name)
            .ok_or_else(|| Error::UnknownInstance {
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
                    return Err(Error::UnknownInstanceAxis {
                        tag: *tag,
                        available: comma_separated(
                            static_metadata.axes.iter().map(|axis| axis.tag.to_string()),
                        ),
                    });
                };
                // deliberately not clamped: a silently moved pin is worse than
                // a rejected one
                if *pos < axis.min || *pos > axis.max {
                    return Err(Error::InstanceAxisOutOfRange {
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

/// The pin in normalized space, which is how the IR keys everything.
///
/// The conversion runs through the source's own axis mapping (what becomes
/// `avar`), which is the same normalization `fontmake -i` applies to a
/// designspace `<instance>`'s location for any strictly monotonic map.
fn normalize(
    static_metadata: &StaticMetadata,
    user: &UserLocation,
) -> Result<NormalizedLocation, Error> {
    user.to_normalized(&static_metadata.axes)
        .map_err(|e| Error::InstanceLocation(e.to_string()))
}

/// The pin in design space, which is the space rule conditions are written in.
///
/// Over *every* source axis, including point axes, because ufo2ft evaluates a
/// rule against the instance's location merged over the whole default design
/// location — a condition on an axis the instance doesn't mention still has to
/// find a position there.
///
/// Straight from user space rather than back out of the normalized pin:
/// normalizing and denormalizing is exact in real arithmetic but not
/// necessarily in f64, and a rule's bounds are *inclusive*, so an answer one
/// ulp under a boundary is a different font.
fn design_pin(static_metadata: &StaticMetadata, user: &UserLocation) -> DesignLocation {
    static_metadata
        .all_source_axes
        .iter()
        .map(|axis| {
            let pos = user.get(axis.tag).unwrap_or(axis.default);
            (axis.tag, pos.to_design(&axis.converter))
        })
        .collect()
}

fn comma_separated(items: impl Iterator<Item = String>) -> String {
    let items: Vec<_> = items.collect();
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

/// Drop the name records that only fvar and STAT were ever going to reference.
///
/// `StaticMetadata::new` mints a name id >= 256 for each axis's UI label and
/// for every named instance's style and PostScript name. A genuinely static
/// source mints none of them — the loops run over the variable axes, and named
/// instances are cleared — so a pinned font that kept them would carry name
/// records nothing points at.
///
/// Matching on the string rather than the id is deliberate: the ids are handed
/// out by a counter and a name is reused if the source already had it, so the
/// value is the only thing that says what a record is *for*.
fn prune_variable_names(pinned: &mut StaticMetadata, source_metadata: &StaticMetadata) {
    let mut orphaned: HashSet<&str> = HashSet::new();
    for axis in source_metadata.axes.iter() {
        orphaned.insert(axis.ui_label_name());
    }
    for instance in source_metadata.named_instances.iter() {
        orphaned.insert(instance.name.as_str());
        if let Some(postscript_name) = instance.postscript_name.as_deref() {
            orphaned.insert(postscript_name);
        }
    }
    pinned
        .names
        .retain(|key, value| key.name_id.to_u16() <= 255 || !orphaned.contains(value.as_str()));
}

/// Does the FEA declare a `conditionset`?
///
/// fea-rs resolves a conditionset's axes against the ones we hand it, so on a
/// pinned font every one of them is an "unknown axis" and the user gets that
/// instead of the truth. Say the truth here.
///
/// A scan, not a parse: the feature file is only parsed in the backend, which
/// by construction runs after the pin. It follows no `include()`, so a
/// conditionset in an included file still produces fea-rs's confusing error;
/// that is a strictly smaller set of cases than before.
fn fea_declares_a_conditionset(features: &FeaturesSource) -> Result<bool, Error> {
    let fea = match features {
        FeaturesSource::Empty => return Ok(false),
        FeaturesSource::Memory { fea_content, .. } => fea_content.clone(),
        FeaturesSource::File { fea_file, .. } => {
            std::fs::read_to_string(fea_file).map_err(|source| Error::FileIo {
                path: fea_file.clone(),
                source,
            })?
        }
    };
    Ok(fea
        .lines()
        // '#' starts a comment that runs to end of line
        .map(|line| line.split('#').next().unwrap_or_default())
        .any(declares_a_conditionset))
}

/// `conditionset` as a whole token, so `conditionsetting` doesn't count.
fn declares_a_conditionset(line: &str) -> bool {
    const KEYWORD: &str = "conditionset";
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.';
    line.match_indices(KEYWORD).any(|(at, _)| {
        let before = at.checked_sub(1).map(|i| line.as_bytes()[i]);
        let after = line.as_bytes().get(at + KEYWORD.len()).copied();
        !before.is_some_and(is_word) && !after.is_some_and(is_word)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use fontdrasil::{
        coords::{CoordConverter, NormalizedLocation, UserCoord},
        types::Axis,
    };
    use fontir::ir::{NameKey, NamedInstance, StaticMetadata};
    use write_fonts::types::{NameId, Tag};

    use super::*;

    const WGHT: Tag = Tag::new(b"wght");
    const WDTH: Tag = Tag::new(b"wdth");

    fn axis(tag: Tag, min: f64, default: f64, max: f64) -> Axis {
        let (min, default, max) = (
            UserCoord::new(min),
            UserCoord::new(default),
            UserCoord::new(max),
        );
        Axis {
            name: tag.to_string(),
            tag,
            min,
            default,
            max,
            hidden: false,
            converter: CoordConverter::unmapped(min, default, max),
            localized_names: Default::default(),
        }
    }

    fn metadata(axes: Vec<Axis>) -> StaticMetadata {
        let locations = HashSet::from([axes
            .iter()
            .map(|axis| (axis.tag, Default::default()))
            .collect::<NormalizedLocation>()]);
        StaticMetadata::new(
            1000,
            Default::default(),
            axes,
            Default::default(),
            locations,
            None,
            0.0,
            None,
            false,
        )
        .unwrap()
    }

    fn variable() -> StaticMetadata {
        metadata(vec![
            axis(WGHT, 400.0, 400.0, 700.0),
            axis(WDTH, 75.0, 100.0, 125.0),
        ])
    }

    fn spec(raw: &str) -> InstanceSpec {
        raw.parse().unwrap()
    }

    /// What the pin does with a `--instance` argument, in one step.
    fn resolve(
        static_metadata: &StaticMetadata,
        spec: &InstanceSpec,
    ) -> Result<NormalizedLocation, Error> {
        normalize(static_metadata, &resolve_user(static_metadata, spec)?)
    }

    #[test]
    fn resolve_a_location_and_default_the_rest() {
        let resolved = resolve(&variable(), &spec("wght=550")).unwrap();
        assert_eq!(
            resolved,
            NormalizedLocation::for_pos(&[("wght", 0.5), ("wdth", 0.0)])
        );
    }

    #[test]
    fn resolve_every_axis() {
        let resolved = resolve(&variable(), &spec("wght=700,wdth=75")).unwrap();
        assert_eq!(
            resolved,
            NormalizedLocation::for_pos(&[("wght", 1.0), ("wdth", -1.0)])
        );
    }

    #[test]
    fn resolve_applies_the_axis_mapping() {
        // user 500 sits at design/normalized 0.5 of the *mapped* range, so a
        // linear reading of user space would say 0.25
        let mut wght = axis(WGHT, 400.0, 400.0, 800.0);
        wght.converter = CoordConverter::new(
            vec![
                (
                    UserCoord::new(400.0),
                    fontdrasil::coords::DesignCoord::new(0.0),
                ),
                (
                    UserCoord::new(500.0),
                    fontdrasil::coords::DesignCoord::new(50.0),
                ),
                (
                    UserCoord::new(800.0),
                    fontdrasil::coords::DesignCoord::new(100.0),
                ),
            ],
            0,
        )
        .unwrap();
        let resolved = resolve(&metadata(vec![wght]), &spec("wght=500")).unwrap();
        assert_eq!(resolved, NormalizedLocation::for_pos(&[("wght", 0.5)]));
    }

    #[test]
    fn resolve_a_named_instance() {
        let mut meta = variable();
        meta.named_instances = vec![NamedInstance {
            name: "Bold".to_string(),
            postscript_name: None,
            location: vec![(WGHT, UserCoord::new(700.0)), (WDTH, UserCoord::new(100.0))].into(),
        }];

        let resolved = resolve(&meta, &spec("Bold")).unwrap();
        assert_eq!(
            resolved,
            NormalizedLocation::for_pos(&[("wght", 1.0), ("wdth", 0.0)])
        );
    }

    #[test]
    fn a_static_source_cannot_be_instanced() {
        // one point axis: static by fontc's definition, and the shape a
        // single-master designspace produces
        let meta = metadata(vec![axis(WGHT, 400.0, 400.0, 400.0)]);
        let e = resolve(&meta, &spec("wght=400")).unwrap_err();
        assert!(e.to_string().contains("requires a variable source"), "{e}");
    }

    #[test]
    fn an_unknown_axis_lists_the_ones_there_are() {
        let e = resolve(&variable(), &spec("opsz=14")).unwrap_err();
        assert!(e.to_string().contains("'opsz'"), "{e}");
        assert!(e.to_string().contains("wght, wdth"), "{e}");
    }

    #[test]
    fn a_position_outside_the_axis_is_not_clamped() {
        let e = resolve(&variable(), &spec("wght=900")).unwrap_err();
        assert!(e.to_string().contains("'wght'"), "{e}");
        assert!(e.to_string().contains("400"), "{e}");
        assert!(e.to_string().contains("700"), "{e}");
    }

    #[test]
    fn an_unknown_instance_lists_the_ones_there_are() {
        let mut meta = variable();
        meta.named_instances = vec![NamedInstance {
            name: "Bold".to_string(),
            postscript_name: None,
            location: vec![(WGHT, UserCoord::new(700.0)), (WDTH, UserCoord::new(100.0))].into(),
        }];

        let e = resolve(&meta, &spec("Boldish")).unwrap_err();
        assert!(e.to_string().contains("'Boldish'"), "{e}");
        assert!(e.to_string().contains("Bold"), "{e}");
    }

    #[test]
    fn no_named_instances_at_all_still_reads() {
        let e = resolve(&variable(), &spec("Bold")).unwrap_err();
        assert!(e.to_string().contains("none"), "{e}");
    }

    #[test]
    fn prune_names_minted_for_axes_and_instances() {
        let mut meta = variable();
        meta.named_instances = vec![NamedInstance {
            name: "Bold".to_string(),
            postscript_name: Some("Test-Bold".to_string()),
            location: vec![(WGHT, UserCoord::new(700.0)), (WDTH, UserCoord::new(100.0))].into(),
        }];
        // re-run the constructor so the axis/instance names get minted
        let meta = StaticMetadata::new(
            1000,
            HashMap::from([(
                NameKey::new(NameId::FAMILY_NAME, "Test"),
                "Test".to_string(),
            )]),
            meta.all_source_axes.clone().into_inner(),
            meta.named_instances.clone(),
            meta.variation_model.locations().cloned().collect(),
            None,
            0.0,
            None,
            false,
        )
        .unwrap();
        let minted: Vec<_> = meta
            .names
            .iter()
            .filter(|(key, _)| key.name_id.to_u16() > 255)
            .map(|(_, value)| value.clone())
            .collect();
        assert_eq!(minted.len(), 4, "{minted:?}");

        let mut pinned = meta.clone();
        prune_variable_names(&mut pinned, &meta);

        assert!(
            pinned.names.keys().all(|key| key.name_id.to_u16() <= 255),
            "{:?}",
            pinned.names
        );
        assert_eq!(
            pinned.names.get(&NameKey::new(NameId::FAMILY_NAME, "Test")),
            Some(&"Test".to_string()),
            "names the source gave us are untouched"
        );
    }

    #[test]
    fn spot_a_conditionset() {
        assert!(declares_a_conditionset("conditionset Light {"));
        assert!(declares_a_conditionset("  conditionset Light {"));
        assert!(!declares_a_conditionset("feature liga {"));
        // not the keyword
        assert!(!declares_a_conditionset("@conditionsets = [a b];"));
        assert!(!declares_a_conditionset("sub conditionset.alt by x;"));
    }

    #[test]
    fn a_commented_out_conditionset_is_not_one() {
        let features = FeaturesSource::from_string(
            "# conditionset Light {\nfeature liga { sub f i by f_i; } liga;\n".to_string(),
        );
        assert!(!fea_declares_a_conditionset(&features).unwrap());
    }

    #[test]
    fn a_real_conditionset_is_one() {
        let features = FeaturesSource::from_string(
            "conditionset Light {\n wght 400 500;\n} Light;\n".to_string(),
        );
        assert!(fea_declares_a_conditionset(&features).unwrap());
    }
}
