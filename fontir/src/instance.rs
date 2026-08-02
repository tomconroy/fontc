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
//! *counts*. Every interpolation below therefore uses
//! [`RoundingBehaviour::None`].
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
    collections::{BTreeMap, HashMap, HashSet},
};

use fontdrasil::{
    coords::NormalizedLocation,
    types::Axes,
    variations::{DeltaError, RoundingBehaviour, VariationModel},
};
use kurbo::Point;
use ordered_float::OrderedFloat;
use smol_str::SmolStr;

use crate::{
    error::{BadGlyph, Error},
    ir::{
        Anchor, GlobalMetrics, GlobalMetricsBuilder, Glyph, GlyphAnchors, GlyphInstance,
        KerningInstance, StaticMetadata,
    },
};

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
/// every master. Unrounded, see the module docs.
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
    // when instantiating intermediates we don't want to do rounding (this is
    // a significant problem if we round some component transformations, where
    // the fractional bits can be very important).
    // This matches fonttools, see https://github.com/googlefonts/ufo2ft/blob/01d3faee/Lib/ufo2ft/_compilers/baseCompiler.py#L266
    let deltas = model
        .deltas_with_rounding(&point_seqs, RoundingBehaviour::None)
        .map_err(|e| BadGlyph::new(&glyph.name, e))?;
    let points = model.interpolate_from_deltas(loc, &deltas);
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
            let deltas = model
                .deltas_with_rounding(&point_seqs, RoundingBehaviour::None)
                .map_err(|e| BadGlyph::new(anchors.glyph_name.clone(), e))?;
            let pos = model
                .interpolate_from_deltas(pin, &deltas)
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
/// backend ignore whatever else is still lying around.
///
/// Groups are the default master's, matching ufo2ft: it builds `MathKerning`
/// with the default source's groups precisely so kerning math never has to
/// union group definitions that disagree.
///
/// # Not yet fontmake's class-kerning fallback
///
/// `fontMath` resolves *every* pair, at every master, through
/// `exact -> (group1, glyph2) -> (glyph1, group2) -> (group1, group2) -> 0`,
/// so a pair one master omits contributes whatever a group pair covering it
/// says, and only contributes 0 when nothing covers it. We implement only the
/// final rung: a pair absent at a master contributes 0 there. That is right for
/// glyph-glyph pairs no group covers, and wrong for mixed class/exception
/// kerning, which is common. Implementing the rest of the cascade is a
/// follow-up; see the parity notes.
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

    let model = VariationModel::new(
        instances
            .iter()
            .map(|instance| instance.location.clone())
            .collect(),
        axis_order,
    );

    let pairs: HashSet<_> = instances
        .iter()
        .flat_map(|instance| instance.kerns.keys())
        .collect();

    let mut kerns = BTreeMap::new();
    for pair in pairs {
        let point_seqs: HashMap<_, _> = instances
            .iter()
            .map(|instance| {
                let value = instance
                    .kerns
                    .get(pair)
                    .copied()
                    .unwrap_or_default()
                    .into_inner();
                (instance.location.clone(), vec![value])
            })
            .collect();
        let deltas = model.deltas_with_rounding(&point_seqs, RoundingBehaviour::None)?;
        let value = model
            .interpolate_from_deltas(pin, &deltas)
            .first()
            .copied()
            .unwrap_or(0.0);
        kerns.insert(pair.clone(), OrderedFloat(value));
    }

    Ok(KerningInstance {
        location: key.clone(),
        kerns,
        groups,
    })
}

/// Glyphs.app per-master numbers reduced to their values at `pin`.
///
/// Each named number interpolates from the masters that define it, and a master
/// that doesn't define it simply isn't in that number's model — unlike kerning,
/// where an absent pair means zero. There is no fontmake behaviour to match
/// here: a number every master lacks has no value to fall back to.
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
                    .map(|value| (loc.clone(), vec![value.into_inner()]))
            })
            .collect();
        let model = VariationModel::new(point_seqs.keys().cloned().collect(), axis_order.clone());
        let deltas = model.deltas_with_rounding(&point_seqs, RoundingBehaviour::None)?;
        let value = model
            .interpolate_from_deltas(pin, &deltas)
            .first()
            .copied()
            .unwrap_or(0.0);
        pinned.insert(name.clone(), OrderedFloat(value));
    }

    Ok(HashMap::from([(key.clone(), pinned)]))
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
///   table still describe the family, not the instance.
/// - `italic_angle` is a single scalar taken from the default master, so an
///   instance on a `slnt`/`ital` axis gets the wrong `post.italicAngle`.
pub fn pin_static_metadata(
    static_metadata: &StaticMetadata,
    pin: &NormalizedLocation,
) -> Result<StaticMetadata, DeltaError> {
    let key = static_metadata.default_location().clone();
    let number_values = pin_number_values(static_metadata, pin)?;

    // TODO(reconciliation): fontMath interpolates the postscript* fontinfo
    // attributes like any other numeric field, and *drops* a blues/stems list
    // outright when two masters state different-length lists. Until that is
    // settled we take the nearest-master (really: pin's own, else the default
    // master's) values verbatim.
    let postscript =
        HashMap::from([(key.clone(), static_metadata.postscript_at(pin).into_owned())]);

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

    fn regular() -> NormalizedLocation {
        NormalizedLocation::for_pos(&[("wght", 0.0)])
    }

    fn bold() -> NormalizedLocation {
        NormalizedLocation::for_pos(&[("wght", 1.0)])
    }

    fn mid() -> NormalizedLocation {
        NormalizedLocation::for_pos(&[("wght", 0.5)])
    }

    /// Two masters on one wght axis, nothing else populated.
    fn test_static_metadata() -> StaticMetadata {
        StaticMetadata::new(
            1000,
            Default::default(),
            vec![wght()],
            Default::default(),
            HashSet::from([regular(), bold()]),
            None,
            0.0,
            None,
            false,
        )
        .unwrap()
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

    fn kern_pair(one: &str, two: &str) -> (KernSide, KernSide) {
        (KernSide::Glyph(one.into()), KernSide::Glyph(two.into()))
    }

    #[test]
    fn pin_kerning_midpoint() {
        let meta = test_static_metadata();
        let group = KernGroup::Side1("A".into());
        let groups = BTreeMap::from([(group.clone(), BTreeSet::from([GlyphName::from("A")]))]);
        let instances = [
            KerningInstance {
                location: regular(),
                kerns: BTreeMap::from([
                    (kern_pair("A", "V"), OrderedFloat(-20.0)),
                    (kern_pair("A", "space"), OrderedFloat(-3.0)),
                ]),
                groups: groups.clone(),
            },
            KerningInstance {
                location: bold(),
                kerns: BTreeMap::from([(kern_pair("A", "V"), OrderedFloat(-41.0))]),
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
        // absent in Bold, so Bold contributes 0
        assert_eq!(
            pinned.kerns.get(&kern_pair("A", "space")),
            Some(&OrderedFloat(-1.5))
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

    #[test]
    fn pin_static_metadata_collapses_postscript() {
        let mut meta = test_static_metadata();
        meta.postscript = HashMap::from([
            (
                regular(),
                crate::ir::PostscriptSettings {
                    blue_values: vec![OrderedFloat(-16.0), OrderedFloat(0.0)],
                    ..Default::default()
                },
            ),
            (
                bold(),
                crate::ir::PostscriptSettings {
                    blue_values: vec![
                        OrderedFloat(-18.0),
                        OrderedFloat(0.0),
                        OrderedFloat(700.0),
                        OrderedFloat(718.0),
                    ],
                    ..Default::default()
                },
            ),
        ]);

        let pinned = pin_static_metadata(&meta, &bold()).unwrap();

        assert_eq!(
            pinned.postscript.keys().collect::<Vec<_>>(),
            vec![meta.default_location()]
        );
        // TODO(reconciliation): today the pin's own master's values, verbatim
        assert_eq!(
            pinned.postscript_default().blue_values,
            [-18.0, 0.0, 700.0, 718.0].map(OrderedFloat)
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
