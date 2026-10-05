use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    path::PathBuf,
    str::FromStr,
    sync::OnceLock,
};

use indexmap::IndexMap;
use kurbo::{BezPath, Point};
use log::{debug, trace, warn};
use ordered_float::OrderedFloat;

use smol_str::SmolStr;
use write_fonts::types::Tag;

use fontdrasil::{
    coords::{CoordConverter, DesignCoord, DesignLocation, NormalizedLocation, UserCoord},
    piecewise_linear_map::PiecewiseLinearMap,
    types::{Axes, GlyphName},
};
use fontir::{
    error::{BadGlyph, BadGlyphKind, Error, PathConversionError},
    ir::{
        self, Color, ColorStop, GlyphPathBuilder, Paint, PaintLinearGradient, PaintRadialGradient,
        PaintSolid,
    },
};
use glyphs_reader::{
    Component, FeatureSnippet, Font, Glyph, InstanceType, Layer, NodeType, Path, Shape,
    ShapeAttributes,
};

pub(crate) fn to_ir_contours_and_components(
    glyph_name: GlyphName,
    shapes: &[Shape],
    erase_open_corners: bool,
) -> Result<(Vec<BezPath>, Vec<ir::Component>), BadGlyph> {
    // For most glyphs in most fonts all the shapes are contours so it's a good guess
    let mut contours = Vec::with_capacity(shapes.len());
    let mut components = Vec::new();

    for shape in shapes.iter() {
        match shape {
            Shape::Component(component) => {
                components.push(to_ir_component(glyph_name.clone(), component))
            }
            Shape::Path(path) => contours.push(
                to_ir_path(glyph_name.clone(), path, erase_open_corners)
                    .map_err(|e| BadGlyph::new(glyph_name.clone(), e))?,
            ),
        }
    }

    Ok((contours, components))
}

fn to_ir_component(glyph_name: GlyphName, component: &Component) -> ir::Component {
    trace!(
        "{} reuses {} with transform {:?}",
        glyph_name, component.name, component.transform
    );
    ir::Component {
        base: component.name.as_str().into(),
        transform: component.transform,
        anchor: component.anchor.clone(),
    }
}

fn add_to_path<'a>(
    path_builder: &'a mut GlyphPathBuilder,
    nodes: impl Iterator<Item = &'a glyphs_reader::Node>,
) -> Result<(), PathConversionError> {
    // Walk through the remaining points, accumulating off-curve points until we see an on-curve
    // https://github.com/googlefonts/glyphsLib/blob/24b4d340e4c82948ba121dcfe563c1450a8e69c9/Lib/glyphsLib/pens.py#L92
    for node in nodes {
        // Smooth is only relevant to editors so ignore here
        match node.node_type {
            NodeType::Line | NodeType::LineSmooth => path_builder.line_to((node.pt.x, node.pt.y)),
            NodeType::Curve | NodeType::CurveSmooth => {
                path_builder.curve_to((node.pt.x, node.pt.y))
            }
            NodeType::OffCurve => path_builder.offcurve((node.pt.x, node.pt.y)),
            NodeType::QCurve | NodeType::QCurveSmooth => {
                path_builder.qcurve_to((node.pt.x, node.pt.y))
            }
        }?
    }
    Ok(())
}

fn to_ir_path(
    glyph_name: GlyphName,
    src_path: &Path,
    erase_open_corners: bool,
) -> Result<BezPath, PathConversionError> {
    // Based on https://github.com/googlefonts/glyphsLib/blob/24b4d340e4c82948ba121dcfe563c1450a8e69c9/Lib/glyphsLib/builder/paths.py#L20
    // See also https://github.com/fonttools/ufoLib2/blob/4d8a9600148b670b0840120658d9aab0b38a9465/src/ufoLib2/pointPens/glyphPointPen.py#L16
    if src_path.nodes.is_empty() {
        return Ok(BezPath::new());
    }

    let mut path_builder = GlyphPathBuilder::new(src_path.nodes.len());

    // First is a delicate butterfly
    if !src_path.closed {
        let first = src_path.nodes.first().unwrap();
        if first.node_type == NodeType::OffCurve {
            return Err(PathConversionError::Parse(
                "Open path starts with off-curve points".into(),
            ));
        }
        path_builder.move_to((first.pt.x, first.pt.y))?;
        add_to_path(&mut path_builder, src_path.nodes[1..].iter())?;
    } else {
        // In Glyphs.app, the starting node of a closed contour is always
        // stored at the end of the nodes list.
        // Rotate right by 1 by way of chaining iterators
        //
        // glyphsLib rotates every closed contour, including one made only of
        // off-curve points (the implied-quadratic case, rare but real). That
        // rotation is not a no-op there: with no on-curve point to start from,
        // the contour starts at the midpoint of the last and first off-curves,
        // so which node sits first decides where it begins.
        let last_idx = src_path.nodes.len() - 1;
        add_to_path(
            &mut path_builder,
            std::iter::once(&src_path.nodes[last_idx]).chain(&src_path.nodes[..last_idx]),
        )?;
    };

    if erase_open_corners && path_builder.erase_open_corners()? {
        log::debug!("erased open contours for {glyph_name}");
    }

    let path = path_builder.build()?;

    trace!(
        "Built a {} entry path for {glyph_name}",
        path.elements().len(),
    );
    Ok(path)
}

pub(crate) fn to_ir_features(
    features: &[FeatureSnippet],
    include_dir: Option<PathBuf>,
) -> Result<ir::FeatureSources, Error> {
    // Based on https://github.com/googlefonts/glyphsLib/blob/24b4d340e4c82948ba121dcfe563c1450a8e69c9/Lib/glyphsLib/builder/features.py#L74
    // TODO: token expansion
    // TODO: implement notes
    let fea_snippets: Vec<_> = features.iter().filter_map(|f| f.str_if_enabled()).collect();
    // a .glyphs file has one set of features, shared by every master
    Ok(ir::FeatureSources::single(ir::FeaturesSource::Memory {
        fea_content: fea_snippets.join("\n\n"),
        include_dir,
    }))
}

/// Read a location off a value list that is indexed by *surviving* axis.
///
/// A brace layer's coordinates are such a list: glyphsLib zips them against the
/// designspace axes, so a coordinate for an axis that got dropped is silently read
/// as the next surviving axis' position.
/// <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/sources.py#L188-L190>
pub(crate) fn design_location(
    axes: &fontdrasil::types::Axes,
    axes_values: &[OrderedFloat<f64>],
) -> DesignLocation {
    axes.iter()
        .zip(axes_values.iter())
        .map(|(axis, pos)| (axis.tag, DesignCoord::new(*pos)))
        .collect()
}

/// Read a location off a master's or instance's `axesValues`.
///
/// Unlike a brace layer's coordinates, these are indexed by the axes the *source*
/// declares, dropped ones included, so each surviving axis reads the slot it had
/// before the drop.
/// <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/sources.py#L126-L133>
pub(crate) fn source_design_location(
    axes: &fontdrasil::types::Axes,
    axis_indices: &[usize],
    axes_values: &[OrderedFloat<f64>],
) -> DesignLocation {
    axes.iter()
        .zip(axis_indices)
        .filter_map(|(axis, &idx)| axes_values.get(idx).map(|pos| (axis.tag, *pos)))
        .map(|(tag, pos)| (tag, DesignCoord::new(pos)))
        .collect()
}

/// Read a design coord back through the axis mapping to get a user coord.
///
/// Glyphs masters record only a design location, so glyphsLib reverses the
/// mapping to find the user location that names it. The reverse map is built
/// as `{design: user for user, design in sorted(mapping.items())}`, so when
/// several user values share one design value the *largest* user value wins;
/// values off the ends of the map extrapolate by offset, as
/// [`PiecewiseLinearMap`] does.
///
/// <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L259-L263>
fn to_user_coord(mappings: &[(UserCoord, DesignCoord)], design: DesignCoord) -> UserCoord {
    let mut by_user = mappings.to_vec();
    by_user.sort_by_key(|(user, _)| *user);
    // BTreeMap insertion order gives the last (largest user) writer the win
    let by_design: BTreeMap<_, _> = by_user
        .into_iter()
        .map(|(user, design)| (design.into_inner(), user.into_inner()))
        .collect();
    // a BTreeMap's keys are unique, so the map cannot be ambiguous
    let design_to_user = PiecewiseLinearMap::new(by_design.into_iter().collect())
        .expect("unique inputs are never ambiguous");
    UserCoord::new(design_to_user.map(design.into_inner()))
}

/// An axis whose range is the masters' span, read through the mapping; see
/// [`AxisRange::Masters`].
///
/// The mapping keeps only its points inside that span, plus a point for each of
/// the extreme and default masters it doesn't name already, so the converter's
/// extremes are the axis' and every master sits on it.
fn masters_through_mapping(
    mappings: &[(UserCoord, DesignCoord)],
    min: DesignCoord,
    default: DesignCoord,
    max: DesignCoord,
    axis_name: &str,
) -> Result<(CoordConverter, UserCoord, UserCoord, UserCoord), Error> {
    let user_min = to_user_coord(mappings, min);
    let user_default = to_user_coord(mappings, default);
    let user_max = to_user_coord(mappings, max);

    let mut trimmed: Vec<_> = mappings
        .iter()
        .filter(|(user, design)| {
            (min..=max).contains(design) && (user_min..=user_max).contains(user)
        })
        .copied()
        .collect();
    for point in [(user_min, min), (user_default, default), (user_max, max)] {
        if !trimmed.contains(&point) {
            trimmed.push(point);
        }
    }
    trimmed.sort();
    trimmed.dedup_by_key(|(user, _)| *user);

    let default_idx = trimmed
        .iter()
        .position(|(user, _)| *user == user_default)
        .ok_or_else(|| Error::MissingMappingForUserCoord {
            axis_name: axis_name.to_string(),
            mappings: mappings.to_vec(),
            value: user_default,
        })?;
    Ok((
        CoordConverter::new(trimmed, default_idx)?,
        user_min,
        user_default,
        user_max,
    ))
}

/// Where an axis' user-space range comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisRange {
    /// The masters' span, read through the mapping: how Glyphs itself exports.
    ///
    /// Every master is on the axis, so none is ever dropped, and the mapping is
    /// trimmed to the masters' span. Glyphs exports a source whose "Axis Mappings"
    /// reach from 100 to 900 but whose masters sit at 400 and 700 as a 400-700
    /// axis, with the mapping's interior points as avar.
    Masters,
    /// The mapping's own user-space extremes, as glyphsLib reads them.
    ///
    /// A master the mapping can't reach is then off the axis, and
    /// [`drop_sources_outside_axes`] drops it, as fontmake does. Only a Glyphs 2
    /// source reads its axes this way, and only while some master is left at the
    /// default location; see [`FontInfo::try_from`].
    Mapping,
}

/// Convert .glyphs axes to IR axes.
///
///  See <https://github.com/googlefonts/glyphsLib/blob/6f243c1f732ea1092717918d0328f3b5303ffe56/Lib/glyphsLib/builder/axes.py#L155>
fn to_ir_axis(
    font: &Font,
    axis_values: &[OrderedFloat<f64>],
    default_idx: usize,
    axis: &glyphs_reader::Axis,
    range: AxisRange,
) -> Result<fontdrasil::types::Axis, Error> {
    let min = axis_values.iter().min().unwrap();
    let max = axis_values.iter().max().unwrap();
    let default = axis_values[default_idx];

    // Given in design coords based on a sample file
    let default = DesignCoord::new(default);
    let min = DesignCoord::new(*min);
    let max = DesignCoord::new(*max);

    let mappings: Vec<(UserCoord, DesignCoord)> = font
        .axis_mappings
        .get(&axis.name)
        .filter(|mapping| !mapping.is_identity())
        .map(|mapping| {
            mapping
                .iter()
                .map(|(user, design)| (UserCoord::new(*user), DesignCoord::new(*design)))
                .collect()
        })
        .unwrap_or_default();

    // Read as glyphsLib does (`AxisRange::Mapping`), a mapped axis takes its user-space
    // extremes from the mapping itself, never from the masters: instances contribute
    // mappings too, so the mapped range can reach past the masters, and a master can sit
    // at a design value the mapping never names.
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L284-L285>
    // <https://github.com/googlefonts/fontc/issues/1991>
    //
    // The default master's user location is then the reverse of its design location,
    // clamped into that range.
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L259-L263>
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L286>
    let mapped = (range == AxisRange::Mapping && !mappings.is_empty()).then(|| {
        #[allow(clippy::unwrap_used)] // a non-identity mapping isn't empty
        let user_min = mappings.iter().map(|(user, _)| *user).min().unwrap();
        #[allow(clippy::unwrap_used)] // a non-identity mapping isn't empty
        let user_max = mappings.iter().map(|(user, _)| *user).max().unwrap();
        (user_min, to_user_coord(&mappings, default), user_max)
    });

    // glyphsLib always uses the mapping; we can't when the axis is degenerate *and*
    // the mapping can't reach the default master. The clamp would then invent a user
    // default the mapping never named, and since our normalization is built from the
    // mapping's design vertices every master would land off it. The masters that a
    // mapping can't reach are dropped below - but on a degenerate axis that is all of
    // them, leaving no font. varLib refuses such a source outright; we keep building it
    // as the unmapped axis it may as well be.
    let mapped = mapped.filter(|(user_min, user_default, user_max)| {
        min != max || (user_min <= user_default && user_default <= user_max)
    });

    let (converter, user_min, user_default, user_max) =
        if range == AxisRange::Masters && !mappings.is_empty() {
            masters_through_mapping(&mappings, min, default, max, &axis.name)?
        } else if let Some((user_min, user_default, user_max)) = mapped {
            let user_default = user_default.clamp(user_min, user_max);
            let default_idx = mappings
                .iter()
                .position(|(user, _)| *user == user_default)
                .ok_or_else(|| Error::MissingMappingForUserCoord {
                    axis_name: axis.name.clone(),
                    mappings: mappings.clone(),
                    value: user_default,
                })?;
            (
                CoordConverter::new(mappings, default_idx)?,
                user_min,
                user_default,
                user_max,
            )
        } else {
            // There is no meaningful mapping; design == user. Virtual masters are in
            // axis_values, and this is the only branch where glyphsLib lets them widen
            // the axis: it adds them to an identity mapping only.
            // <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/axes.py#L266-L282>
            let min = UserCoord::new(min.into_inner());
            let max = UserCoord::new(max.into_inner());
            let default = UserCoord::new(default.into_inner());
            (
                CoordConverter::unmapped(min, default, max),
                min,
                default,
                max,
            )
        };

    Ok(fontdrasil::types::Axis {
        name: axis.name.clone(),
        tag: Tag::from_str(&axis.tag).map_err(|cause| Error::InvalidTag {
            raw_tag: axis.tag.clone(),
            cause,
        })?,
        // We keep this where fontmake sometimes can't: a hidden Weight or Width axis
        // still compares equal to the default one glyphsLib synthesises its "Axes"
        // parameter from, so glyphsLib reads the flag back off the defaults and loses
        // it. See `RawFont::declares_axes`; that is a glyphsLib bug, not a rule.
        hidden: axis.hidden.unwrap_or(false),
        min: user_min,
        default: user_default,
        max: user_max,
        converter,
        // localized axis names from .glyphs sources aren't supported yet
        // https://forum.glyphsapp.com/t/localisable-axis-names/19028
        localized_names: Default::default(),
    })
}

/// The user-space position glyphsLib treats as "this axis is doing nothing".
///
/// <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/axes.py#L543-L550>
fn default_user_loc(tag: Tag) -> f64 {
    match tag {
        _ if tag == Tag::new(b"wght") => 400.0,
        _ if tag == Tag::new(b"wdth") => 100.0,
        _ => 0.0,
    }
}

/// Would glyphsLib write this axis into the designspace?
///
/// A Glyphs 2 source has three axis slots whether it wants them or not, so most fonts
/// carry a Width and a Custom axis that never move. glyphsLib throws such an axis away,
/// but only when it is *entirely* inert: parked at the position that axis means nothing
/// at, with a user:design mapping that doesn't bend, and unnamed by the font's "Axes"
/// custom parameter. Anything else - a range, a bent mapping, a source that named the
/// axis - keeps it, and fontmake then writes it to fvar, avar, STAT and name.
///
/// <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/axes.py#L288-L299>
fn wanted_in_designspace(
    axis: &fontdrasil::types::Axis,
    is_identity_map: bool,
    declares_axes: bool,
) -> bool {
    axis.min < axis.max
        || axis.min.into_inner() != default_user_loc(axis.tag)
        || !is_identity_map
        || declares_axes
}

/// Drop the masters that sit outside the axes' user-space ranges, and what goes with them.
///
/// A mapped axis takes its range from the mapping, so a source can sit at a
/// design location the mapping never reaches: a Glyphs 2 file that uses the
/// Width axis as an italic toggle, giving every instance the same (default)
/// widthClass, ends up with a Width axis pinned to one user value while half
/// the masters sit off it.
///
/// fontmake never sees those sources. designspaceLib carves the variable font
/// out of the designspace first, and keeps only the sources whose design
/// location maps back into every axis' user range.
///
/// designspaceLib tests instances the same way, but does it on the way into fvar
/// rather than here: an instance the region excludes is still an instance the
/// designspace declared, and `fontmake -i` will interpolate it. That test lives in
/// [`fontir::ir::StaticMetadata::fvar_instances`]. What has to happen *here* is
/// narrower and not about regions at all: an instance whose masters just went away
/// cannot be built by anything, so it goes with them.
///
/// <https://github.com/fonttools/fonttools/blob/4.63.0/Lib/fontTools/designspaceLib/split.py#L275-L278>
fn drop_sources_outside_axes(
    font: &mut Font,
    axes: &Axes,
    axis_indices: &[usize],
) -> Result<(), Error> {
    // `axes_values` is indexed by the source's own axes, dropped ones included
    let in_range = |axes_values: &[OrderedFloat<f64>]| {
        axes.iter().zip(axis_indices).all(|(axis, &idx)| {
            axes_values.get(idx).is_none_or(|value| {
                let user = DesignCoord::new(*value).to_user(&axis.converter);
                axis.min <= user && user <= axis.max
            })
        })
    };

    let dropped: HashSet<_> = font
        .masters
        .iter()
        .filter(|master| !in_range(&master.axes_values))
        .map(|master| master.id.clone())
        .collect();
    if dropped.is_empty() {
        return Ok(());
    }

    // If the default master is going with them the survivor at the default
    // location takes over, as designspaceLib's `subDoc.findDefault()` does. Work
    // that out before touching the font: with no such survivor the caller reads
    // the axes again rather than dropping anything.
    let default_master_id = font.default_master().id.clone();
    let survivors = || font.masters.iter().filter(|m| !dropped.contains(&m.id));
    let default_master_idx = if dropped.contains(&default_master_id) {
        let at_default = |axes_values: &[OrderedFloat<f64>]| {
            axes.iter().zip(axis_indices).all(|(axis, &idx)| {
                axes_values.get(idx).is_some_and(|value| {
                    *value == axis.default.to_design(&axis.converter).into_inner()
                })
            })
        };
        survivors()
            .position(|master| at_default(&master.axes_values))
            .ok_or(Error::NoDefaultMaster)?
    } else {
        #[allow(clippy::unwrap_used)] // it isn't dropped, so it survives
        survivors()
            .position(|master| master.id == default_master_id)
            .unwrap()
    };

    for master in font.masters.iter().filter(|m| dropped.contains(&m.id)) {
        warn!(
            "Master '{}' is outside the axis ranges the mapping defines; dropping it",
            master.name
        );
    }
    font.masters.retain(|master| !dropped.contains(&master.id));
    // A variable instance describes a whole variable font rather than a point
    // in it; glyphsLib doesn't write it as a designspace instance at all.
    font.instances.retain(|instance| {
        instance.type_ != InstanceType::Single || in_range(&instance.axes_values)
    });
    for glyph in font.glyphs.values_mut() {
        glyph
            .layers
            .retain(|layer| !dropped.contains(layer.master_id()));
    }
    // Kerning is keyed by master id and only ever read for a live master, so
    // the dropped masters' entries can stay where they are.
    font.default_master_idx = default_master_idx;
    Ok(())
}

fn ir_axes(font: &Font, range: AxisRange) -> Result<(fontdrasil::types::Axes, Vec<usize>), Error> {
    // Every master should have a value for every axis
    for master in font.masters.iter() {
        if font.axes.len() != master.axes_values.len() {
            return Err(Error::InconsistentAxisDefinitions(format!(
                "Axes {:?} doesn't match axis values {:?}",
                font.axes, master.axes_values
            )));
        }
    }

    let mut axes = Vec::new();
    let mut axis_indices = Vec::new();
    for (idx, glyphs_axis) in font.axes.iter().enumerate() {
        let axis_values: Vec<_> = font
            .masters
            .iter()
            .map(|m| m.axes_values[idx])
            // extend the masters' axis values with the virtual masters' if any;
            // they will be used to compute the axis min/max values
            .chain(font.virtual_masters.iter().flat_map(|vm| {
                vm.iter().filter_map(|(axis_name, location)| {
                    if axis_name == &glyphs_axis.name {
                        Some(*location)
                    } else {
                        None
                    }
                })
            }))
            .collect();
        let axis = to_ir_axis(
            font,
            &axis_values,
            font.default_master_idx,
            glyphs_axis,
            range,
        )?;
        let is_identity_map = font
            .axis_mappings
            .get(&glyphs_axis.name)
            .is_none_or(|mapping| mapping.is_identity());
        if wanted_in_designspace(&axis, is_identity_map, font.declares_axes) {
            axes.push(axis);
            axis_indices.push(idx);
        }
    }

    Ok((fontdrasil::types::Axes::new(axes), axis_indices))
}

/// A [Font] with some prework to convert to IR predone.
#[derive(Debug)]
pub(crate) struct FontInfo {
    pub font: Font,
    /// Index by master id
    pub master_indices: HashMap<String, usize>,
    // Master id => location
    pub master_positions: HashMap<String, NormalizedLocation>,
    /// Axes values => location for every instance and master
    pub locations: HashMap<Vec<OrderedFloat<f64>>, NormalizedLocation>,
    /// The axes that survive into the designspace; see [`ir_axes`].
    pub axes: fontdrasil::types::Axes,
    /// Name of glyph : color glyphs split from it, if any
    pub color_glyphs: IndexMap<SmolStr, Vec<SmolStr>>,
    /// The kern-group partition, lazily derived once by the
    /// `FontInfo::kern_groups` accessor; per-glyph attributes make it
    /// font-global, unlike UFO sources' per-master groups.
    pub kern_groups: OnceLock<BTreeMap<ir::KernGroup, BTreeSet<GlyphName>>>,
}

impl TryFrom<Font> for FontInfo {
    type Error = Error;

    fn try_from(mut font: Font) -> Result<Self, Self::Error> {
        // A Glyphs 3 or 4 source reads its axes as Glyphs does: the masters are
        // the range, and they all stay. A Glyphs 2 source reads them as glyphsLib
        // does, for fontmake's sake: the axes are read off every master, and only
        // then do the sources the axes can't reach get dropped. Not the default
        // master, though, unless another master can take its place; when none
        // can, fontmake can't build the source at all, and we take the range off
        // the masters instead, so that every one of them stays.
        let (axes, axis_indices) = if font.is_glyphs2() {
            let (axes, axis_indices) = ir_axes(&font, AxisRange::Mapping)?;
            match drop_sources_outside_axes(&mut font, &axes, &axis_indices) {
                Ok(()) => (axes, axis_indices),
                Err(Error::NoDefaultMaster) => {
                    warn!(
                        "The axis mappings leave no master at the default location; \
                         taking the axis ranges from the masters instead"
                    );
                    ir_axes(&font, AxisRange::Masters)?
                }
                Err(e) => return Err(e),
            }
        } else {
            ir_axes(&font, AxisRange::Masters)?
        };

        let master_indices: HashMap<_, _> = font
            .masters
            .iter()
            .enumerate()
            .map(|(idx, m)| (m.id.clone(), idx))
            .collect();

        let locations: HashMap<_, _> = font
            .masters
            .iter()
            .map(|m| {
                (
                    m.axes_values.clone(),
                    source_design_location(&axes, &axis_indices, &m.axes_values)
                        .to_normalized(&axes)
                        .unwrap(),
                )
            })
            .chain(font.instances.iter().map(|i| {
                (
                    i.axes_values.clone(),
                    source_design_location(&axes, &axis_indices, &i.axes_values)
                        .to_normalized(&axes)
                        .unwrap(),
                )
            }))
            .collect();

        let master_positions: HashMap<_, _> = font
            .masters
            .iter()
            .map(|m| (&m.id, locations.get(&m.axes_values).unwrap()))
            .map(|(id, pos)| (id.clone(), pos.clone()))
            .collect();

        let (font, color_glyphs) = split_color_glyphs(font)?;

        Ok(FontInfo {
            font,
            master_indices,
            master_positions,
            locations,
            axes,
            color_glyphs,
            kern_groups: OnceLock::new(),
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum Colrv1RunType {
    NonColor,
    Solid(glyphs_reader::Color),
    // (start, end, colors) - geometry matters for distinguishing gradients
    Linear(
        Vec<OrderedFloat<f64>>,
        Vec<OrderedFloat<f64>>,
        Vec<glyphs_reader::ColorStop>,
    ),
    // (start, end, start radius, end radius, colors)
    Radial(
        Vec<OrderedFloat<f64>>,
        Vec<OrderedFloat<f64>>,
        Option<OrderedFloat<f64>>,
        Option<OrderedFloat<f64>>,
        Vec<glyphs_reader::ColorStop>,
    ),
    Unknown(ShapeAttributes),
}

#[derive(Debug)]
struct Colrv1Run {
    run_type: Colrv1RunType,
    start: usize,
    end: usize,
}

impl Colrv1Run {
    fn color(&self) -> bool {
        !matches!(self.run_type, Colrv1RunType::NonColor)
    }
}

impl Colrv1RunType {
    fn key_for(layer: &Layer, shape: &Shape) -> Self {
        // COLRv1?
        if !layer.attributes.color {
            return Colrv1RunType::NonColor;
        }
        let attr = shape.attributes();
        if let Some(gradient) = &attr.gradient {
            if matches!(gradient.style.as_str(), "circle" | "radial") {
                return Colrv1RunType::Radial(
                    gradient.start.clone(),
                    gradient.end.clone(),
                    gradient.start_radius,
                    gradient.end_radius,
                    gradient.colors.clone(),
                );
            }
            return Colrv1RunType::Linear(
                gradient.start.clone(),
                gradient.end.clone(),
                gradient.colors.clone(),
            );
        }
        if let Some(fill) = attr.fill_color {
            return Colrv1RunType::Solid(fill);
        }
        Colrv1RunType::Unknown(attr.clone())
    }
}

fn new_color_glyph(original: &Glyph, nth: &mut usize) -> Glyph {
    let new_glyph_name: SmolStr = format!("{}.color{nth}", original.name).into();
    let new_production_name = original
        .production_name
        .as_ref()
        .map(|production_name| format!("{}.color{nth}", production_name).into());
    let new_glyph = Glyph {
        name: new_glyph_name.clone(),
        production_name: new_production_name,
        export: original.export,
        category: original.category,
        sub_category: original.sub_category,
        ..Default::default()
    };
    *nth += 1;
    new_glyph
}

/// A master's brace color layers, grouped by location and palette index.
/// Layers with the same location and palette index keep their document order.
#[derive(Default)]
struct BraceLayers<'a>(IndexMap<&'a [OrderedFloat<f64>], IndexMap<i64, Vec<&'a Layer>>>);

impl<'a> BraceLayers<'a> {
    fn insert(&mut self, layer: &'a Layer, palette_idx: i64) {
        self.0
            .entry(layer.attributes.coordinates.as_slice())
            .or_default()
            .entry(palette_idx)
            .or_default()
            .push(layer);
    }

    fn matching(
        &self,
        palette_idx: i64,
        occurrence: usize,
    ) -> impl Iterator<Item = &'a Layer> + '_ {
        self.0.values().filter_map(move |by_palette| {
            by_palette
                .get(&palette_idx)
                .and_then(|layers| layers.get(occurrence))
                .copied()
        })
    }

    fn unmatched<'s>(
        &'s self,
        seen: &'s IndexMap<i64, usize>,
    ) -> impl Iterator<Item = &'a Layer> + 's {
        self.0.values().flat_map(move |by_palette| {
            by_palette.iter().flat_map(move |(palette_idx, layers)| {
                let consumed = seen.get(palette_idx).copied().unwrap_or_default();
                layers.iter().skip(consumed).copied()
            })
        })
    }
}

/// Build the split color glyphs for a COLRv0 glyph.
///
/// As in glyphsLib, color layers are matched across masters by position:
/// the i-th color layer of each master contributes that master's geometry
/// to `[original].color[i]`, so the color glyphs interpolate. The default
/// master determines how many color glyphs are created; a master with fewer
/// color layers simply contributes no source. An intermediate (brace) color
/// layer becomes a sparse intermediate source of the color glyph whose n-th
/// color layer shares its palette index (n counted in document order per
/// location), matching glyphsLib >= 6.14.0.
/// <https://github.com/googlefonts/glyphsLib/blob/v6.14.0/Lib/glyphsLib/builder/color_layers.py#L33-L107>
fn colrv0_color_glyphs(
    original: &Glyph,
    default_master_id: &str,
    master_ids: &[String],
) -> Vec<Glyph> {
    let glyph_name = &original.name;
    let mut layers_by_master: IndexMap<&str, Vec<&Layer>> = IndexMap::new();
    // Layers grouped under an id that is not a font master are never read:
    // consumption below is keyed by master_ids
    let mut braces_by_master: IndexMap<&str, BraceLayers> = IndexMap::new();
    for layer in original.layers.iter() {
        let Some(palette_idx) = layer.attributes.color_palette else {
            continue;
        };
        if layer.shapes.is_empty() {
            continue;
        }
        let Some(master_id) = layer.associated_master_id.as_deref() else {
            continue;
        };
        if layer.is_intermediate() {
            braces_by_master
                .entry(master_id)
                .or_default()
                .insert(layer, palette_idx);
        } else {
            layers_by_master.entry(master_id).or_default().push(layer);
        }
    }
    let num_color_glyphs = layers_by_master
        .get(default_master_id)
        .map(Vec::len)
        .unwrap_or_default();

    let mut nth = 0;
    // new_glyphs[i] is named [original].color{i}, aligned with the i-th color
    // layer of each master below
    let mut new_glyphs: Vec<Glyph> = (0..num_color_glyphs)
        .map(|_| new_color_glyph(original, &mut nth))
        .collect();

    for master_id in master_ids {
        let braces = braces_by_master.get(master_id.as_str());
        // how many color layers of this master used each palette index so
        // far; the n-th intermediate with an index pairs with the n-th
        // color layer with that index (glyphsLib's seen counter)
        let mut seen: IndexMap<i64, usize> = IndexMap::new();
        for (i, &layer) in layers_by_master
            .get(master_id.as_str())
            .into_iter()
            .flatten()
            .enumerate()
        {
            let palette_idx = layer.attributes.color_palette.unwrap();
            let n = seen.entry(palette_idx).or_default();
            let nth_with_index = *n;
            *n += 1;
            let Some(new_glyph) = new_glyphs.get_mut(i) else {
                // more color layers than the default master has; no color
                // glyph to attach to
                continue;
            };
            let mut master_layer = layer.clone();
            master_layer.layer_id = master_id.clone();
            master_layer.associated_master_id = None;
            new_glyph.layers.push(master_layer);
            // attach the matching intermediate, if any, at each location;
            // it keeps its associated master and coordinates and so
            // becomes an intermediate source of the color glyph
            new_glyph.layers.extend(
                braces
                    .into_iter()
                    .flat_map(|braces| braces.matching(palette_idx, nth_with_index))
                    .cloned(),
            );
        }
        for brace in braces
            .into_iter()
            .flat_map(|braces| braces.unmatched(&seen))
        {
            warn!(
                "{glyph_name}: intermediate color layer {} has no matching color layer and will be skipped",
                brace.layer_id
            );
        }
    }
    new_glyphs
}

fn split_colrv0_glyph(
    original: &Glyph,
    default_master_id: &str,
    master_ids: &[String],
    color_glyphs: &mut IndexMap<SmolStr, Vec<SmolStr>>,
    additions: &mut Vec<(SmolStr, Glyph)>,
) -> Result<(), Error> {
    // COLRv0 runs are just consecutive shapes by palette index
    // The original glyph becomes uncolored,
    // each color run becomes a new glyph named [original].color[i]
    let new_glyphs = colrv0_color_glyphs(original, default_master_id, master_ids);

    for new_glyph in new_glyphs {
        debug!("Add COLRv0 {}", new_glyph.name);

        color_glyphs
            .entry(original.name.clone())
            .or_default()
            .push(new_glyph.name.clone());
        additions.push((new_glyph.name.clone(), new_glyph));
    }

    // The color_glyphs entry drives ColorGlyphsWork::exec: absent = not in
    // COLR, empty = paint the base glyph itself, non-empty = paint the splits.
    // Only a color-valued master layer (which glyphsLib reuses as a color
    // layer painting the base) may reserve an empty entry; an uncolored base
    // with no splits stays absent
    if let Some(default_master_layer) = original
        .layers
        .iter()
        .find(|l| l.layer_id == default_master_id)
        && default_master_layer.is_color()
        && !default_master_layer.shapes.is_empty()
    {
        color_glyphs.entry(original.name.clone()).or_default();
    }
    Ok(())
}

fn split_colrv1_glyph(
    glyph: &Glyph,
    default_master_layer: &Layer,
    color_glyphs: &mut IndexMap<SmolStr, Vec<SmolStr>>,
    additions: &mut Vec<(SmolStr, Glyph)>,
) -> Result<(), Error> {
    let glyph_name = &glyph.name;

    // Split into runs of the same paint
    let mut runs = VecDeque::<Colrv1Run>::new();
    for (idx, shape) in default_master_layer.shapes.iter().enumerate() {
        let run_type = Colrv1RunType::key_for(default_master_layer, shape);
        if let Some(curr) = runs.back_mut()
            && curr.run_type == run_type
        {
            // Extend the current run
            curr.end = idx + 1;
        } else {
            // New run
            runs.push_back(Colrv1Run {
                run_type,
                start: idx,
                end: idx + 1,
            });
        }
    }

    // Only one run we're done
    if runs.len() <= 1 {
        return Ok(());
    }

    // There are multiple runs, we must split this glyph apart
    // The original will remain but uncolored

    // Each color run becomes a new glyph named [original].color[i]
    let mut nth = 0;
    for run in runs {
        let new_glyph_name: SmolStr = format!("{glyph_name}.color{nth}").into();
        let mut new_glyph = new_color_glyph(glyph, &mut nth);

        // For each layer, chop the head that matches this paint group off glyph and attach it here
        for old_layer in glyph.layers.iter() {
            let mut new_layer = old_layer.clone();
            new_layer.attributes.color = run.color();
            new_layer.shapes = old_layer.shapes[run.start..run.end].to_vec();
            trace!(
                "{glyph_name} {} takes {} shapes for {run:?}",
                old_layer.layer_id,
                new_layer.shapes.len()
            );
            new_glyph.layers.push(new_layer);
        }

        let mut layer_sizes = new_glyph
            .layers
            .iter()
            .map(|l| l.shapes.len())
            .collect::<Vec<_>>();
        layer_sizes.sort();
        layer_sizes.dedup();
        if layer_sizes.len() != 1 {
            return Err(Error::BadGlyph(BadGlyph::new(
                new_glyph_name,
                BadGlyphKind::FrontendSpecific(format!("Inconsistent layer sizes {layer_sizes:?}")),
            )));
        }
        if layer_sizes.first() == Some(&0) {
            return Err(Error::BadGlyph(BadGlyph::new(
                new_glyph_name,
                BadGlyphKind::FrontendSpecific("All layers are empty?!".to_string()),
            )));
        }

        color_glyphs
            .entry(glyph_name.clone())
            .or_default()
            .push(new_glyph_name.clone());
        additions.push((new_glyph_name, new_glyph));
    }
    Ok(())
}

fn split_color_glyphs(font: Font) -> Result<(Font, IndexMap<SmolStr, Vec<SmolStr>>), Error> {
    // <https://github.com/googlefonts/glyphsLib/blob/99328059ec4799956ecef3d47ebcc13ae70dacff/Lib/glyphsLib/builder/glyph.py#L309-L357>
    let mut font = font;
    let mut color_glyphs: IndexMap<SmolStr, Vec<SmolStr>> = Default::default();
    let default_master_id = font.default_master().id.clone();
    let master_ids: Vec<String> = font.masters.iter().map(|m| m.id.clone()).collect();

    let mut additions: Vec<(SmolStr, Glyph)> = Vec::new();
    for glyph in font.glyphs.values_mut() {
        if let Some(default_master_layer) = glyph
            .layers
            .iter()
            .find(|l| l.layer_id == default_master_id)
        {
            // If 1..N layers with palette indices are associated this is COLRv0
            // See <https://github.com/googlefonts/glyphsLib/blob/99328059ec4799956ecef3d47ebcc13ae70dacff/Lib/glyphsLib/builder/glyph.py#L289-L292>
            if glyph.layers.iter().any(|l| {
                l.attributes.color_palette.is_some()
                    && l.associated_master_id.as_deref() == Some(default_master_id.as_str())
            }) {
                split_colrv0_glyph(
                    glyph,
                    &default_master_id,
                    &master_ids,
                    &mut color_glyphs,
                    &mut additions,
                )?;
            } else if default_master_layer.is_color() {
                split_colrv1_glyph(
                    glyph,
                    default_master_layer,
                    &mut color_glyphs,
                    &mut additions,
                )?;
                // For COLRv1 single-run glyphs (i.e. no split glyphs created, shapes in default layer),
                // reserve an entry with empty vec so it gets included in COLR (see ColorGlyphsWork::exec).
                // For v1 multi-run, an non-empty vec already exists from split_colrv1_glyph.
                if !default_master_layer.shapes.is_empty() {
                    color_glyphs.entry(glyph.name.clone()).or_default();
                }
            }
        }

        // Palette-valued intermediates belong to split color glyphs only;
        // left on the glyph, GlyphIrWork's is_intermediate() filter would
        // admit them into its own variation (as glyphsLib, which excludes
        // them from ordinary intermediate handling). Unconditional: one
        // associated with a non-default master trips no detection above
        glyph
            .layers
            .retain(|l| l.attributes.color_palette.is_none() || !l.is_intermediate());
    }

    font.glyph_order
        .extend(additions.iter().map(|(gn, _)| gn.clone()));
    font.glyphs.extend(additions);

    trace!("updated glyph order {:?}", font.glyph_order);

    Ok((font, color_glyphs))
}

pub(crate) fn to_ir_color(color: glyphs_reader::Color) -> Color {
    Color {
        r: color.r as u8,
        g: color.g as u8,
        b: color.b as u8,
        a: color.a as u8,
    }
}

pub(crate) fn to_ir_color_stops(stops: &[glyphs_reader::ColorStop]) -> Vec<ColorStop> {
    stops
        .iter()
        .map(|cs| ColorStop {
            offset: (cs.stop_offset.0 as f32).into(),
            color: to_ir_color(cs.color),
            alpha: 255.0.into(),
        })
        .collect()
}

pub(crate) fn to_ir_paint(
    palette: Option<&[glyphs_reader::Color]>,
    glyph_name: impl Into<GlyphName>,
    layer: &Layer,
    attr: &ShapeAttributes,
) -> Result<Paint, Error> {
    if let Some(palette_idx) = layer.attributes.color_palette {
        // 0xFFFF is a special COLR palette index meaning "use the text foreground color"
        if palette_idx == 0xFFFF {
            return Ok(Paint::Solid(PaintSolid { color: None }.into()));
        }
        let Some(palette) = palette else {
            return Err(Error::BadGlyph(BadGlyph::new(
                glyph_name,
                BadGlyphKind::FrontendSpecific("Uses palette but there isn't one".to_string()),
            )));
        };
        let Some(color) = palette.get(palette_idx as usize) else {
            return Err(Error::BadGlyph(BadGlyph::new(
                glyph_name,
                BadGlyphKind::FrontendSpecific(format!(
                    "Out of bounds palette index {palette_idx}"
                )),
            )));
        };
        return Ok(Paint::Solid(
            PaintSolid {
                color: Some(to_ir_color(*color)),
            }
            .into(),
        ));
    }
    if let Some(color) = attr.fill_color {
        return Ok(Paint::Solid(
            PaintSolid {
                color: Some(to_ir_color(color)),
            }
            .into(),
        ));
    }

    // Note: Gradient coordinates from Glyphs are percentages (0.0-1.0) of the layer's bounding box.
    // The scaling to absolute coordinates is done later in fontbe/src/colr.rs, in order to reuse
    // the already-computed glyf bounding boxes and avoid redundant work.
    if let Some(gradient) = &attr.gradient {
        // See <https://github.com/googlefonts/glyphsLib/blob/99328059ec4799956ecef3d47ebcc13ae70dacff/Lib/glyphsLib/builder/color_layers.py#L72>
        let start = Point::new(gradient.start[0].0, gradient.start[1].0);
        let end = Point::new(gradient.end[0].0, gradient.end[1].0);
        return match gradient.style.as_str() {
            "circle" => {
                // Glyphs radial gradient only has a single circle centered at 'start'
                // with the radius calculated as % of the max distance to bbox corners.
                Ok(Paint::RadialGradient(
                    PaintRadialGradient {
                        p0: start,
                        p1: start,
                        r0: None, // Defaults to 0
                        r1: None, // Calculated in backend
                        color_line: to_ir_color_stops(&gradient.colors),
                    }
                    .into(),
                ))
            }
            "radial" => {
                // Glyphs 4: from a circle at 'start' to one at 'end', radii
                // relative to the bbox, scaled in the backend like the points
                let radius = |r: Option<OrderedFloat<f64>>| r.map(|r| OrderedFloat(r.0 as f32));
                Ok(Paint::RadialGradient(
                    PaintRadialGradient {
                        p0: start,
                        p1: end,
                        r0: radius(gradient.start_radius),
                        r1: radius(gradient.end_radius),
                        color_line: to_ir_color_stops(&gradient.colors),
                    }
                    .into(),
                ))
            }
            "" => {
                // p2 is calculated in backend after scaling to absolute coordinates
                // (rotation works differently in percentage vs absolute space).
                Ok(Paint::LinearGradient(
                    PaintLinearGradient {
                        p0: start,
                        p1: end,
                        p2: None,
                        color_line: to_ir_color_stops(&gradient.colors),
                    }
                    .into(),
                ))
            }
            _ => Err(Error::BadGlyph(BadGlyph::new(
                glyph_name,
                BadGlyphKind::FrontendSpecific(format!("Unrecognized gradient {}", gradient.style)),
            ))),
        };
    }

    Err(Error::BadGlyph(BadGlyph::new(
        glyph_name,
        BadGlyphKind::FrontendSpecific(format!(
            "Unable to produce paint for {:?}, {attr:?}",
            layer.attributes
        )),
    )))
}

#[cfg(test)]
mod tests {
    use glyphs_reader::{
        Font, Glyph, Layer, LayerAttributes, Node, Path,
        glyphdata::{Category, Subcategory},
    };
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::{FontInfo, split_color_glyphs, to_ir_path};

    fn testdata_dir() -> PathBuf {
        let dir = PathBuf::from("../resources/testdata");
        assert!(dir.is_dir(), "{dir:?} isn't a dir");
        dir
    }

    #[test]
    fn the_last_of_a_closed_contour_is_first() {
        // In glyph's if we start with off-curve points that means start at the *last* point
        let mut path = Path::new(true);

        // A sort of teardrop thing drawn with a single cubic
        // Offcurve, Offcurve, Oncurve should be taken to start and end at the closing Oncurve.
        path.nodes.push(Node {
            pt: (64.0, 64.0).into(),
            node_type: glyphs_reader::NodeType::OffCurve,
        });
        path.nodes.push(Node {
            pt: (64.0, 0.0).into(),
            node_type: glyphs_reader::NodeType::OffCurve,
        });
        path.nodes.push(Node {
            pt: (32.0, 32.0).into(),
            node_type: glyphs_reader::NodeType::Curve,
        });
        let bez = to_ir_path("test".into(), &path, false).unwrap();
        assert_eq!("M32,32 C64,64 64,0 32,32 Z", bez.to_svg());
    }

    /// A closed contour of nothing but off-curve points starts at the midpoint
    /// of the last and first off-curves — *after* glyphsLib's rotation, which
    /// moves the source's last node to the front. fontc used to skip that
    /// rotation here on the grounds that the order was "already correct",
    /// which started the contour a quarter turn away (at (5,0) for these
    /// nodes) and cost MaShanZheng 60 charstrings.
    ///
    /// (0,5) is what fontmake produces for exactly these four nodes, in both
    /// otf and ttf.
    #[test]
    fn no_on_curve_path_order() {
        let nodes = [(10., 0.), (10., 10.), (0., 10.), (0., 0.)]
            .into_iter()
            .map(|pt| Node {
                pt: pt.into(),
                node_type: glyphs_reader::NodeType::OffCurve,
            })
            .collect();
        let path = Path {
            closed: true,
            nodes,
            ..Default::default()
        };

        let bez = to_ir_path("hello".into(), &path, false).unwrap();
        assert_eq!(
            bez.elements().first(),
            Some(&kurbo::PathEl::MoveTo((0., 5.).into()))
        );
    }

    /// Test that glyphs with empty color palette layers are NOT added to color_glyphs.
    ///
    /// This reproduces a bug where a non-printing glyph like "CR" may nominally contain
    /// palette layers that trigger the COLRv0 code path, but none of the layers have shapes.
    /// The glyph was incorrectly added to color_glyphs, causing a panic when trying to access
    /// layer.shapes[0].
    #[test]
    fn colrv0_glyph_with_empty_palette_layers_is_skipped() {
        let mut font = Font::load(&testdata_dir().join("glyphs3/COLRv0-1layer.glyphs")).unwrap();
        let master_id = font.default_master().id.clone();

        // Add a glyph "CR" with palette layers but no shapes
        let cr_glyph = Glyph {
            name: "CR".into(),
            export: true,
            layers: vec![
                // Default master layer with empty shapes
                Layer {
                    layer_id: master_id.clone(),
                    associated_master_id: None,
                    width: 0.0.into(),
                    shapes: vec![], // Empty!
                    anchors: vec![],
                    attributes: LayerAttributes::default(),
                    ..Default::default()
                },
                // Palette layer has color_palette but empty shapes
                Layer {
                    layer_id: "palette-layer-1".to_string(),
                    associated_master_id: Some(master_id.clone()),
                    width: 0.0.into(),
                    shapes: vec![], // Empty!
                    anchors: vec![],
                    attributes: LayerAttributes {
                        color_palette: Some(0), // This triggers COLRv0 path
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        font.glyphs.insert("CR".into(), cr_glyph);
        font.glyph_order.push("CR".into());

        // this would panic with the old code
        let (_, color_glyphs) = split_color_glyphs(font).unwrap();

        // The glyph should NOT be in color_glyphs because it has no color content
        assert!(
            !color_glyphs.contains_key("CR"),
            "Glyph with empty palette layers should not be added to color_glyphs"
        );
    }

    #[test]
    fn color_layer_glyphs_inherit_parent_category() {
        let font = Font::load(&testdata_dir().join("glyphs3/COLRv0-marks.glyphs")).unwrap();
        let (font, _) = split_color_glyphs(font).unwrap();
        let category = |name: &str| {
            let glyph = &font.glyphs[name];
            (glyph.category, glyph.sub_category)
        };
        let nonspacing = (Some(Category::Mark), Some(Subcategory::Nonspacing));
        // marks by name, by codepoint, and by explicit category
        for mark in ["circumflexcomb", "mymark", "mymark2"] {
            assert_eq!(category(mark), nonspacing, "{mark}");
            assert_eq!(category(&format!("{mark}.color0")), nonspacing, "{mark}");
        }
    }

    /// Split color glyphs keep every master's geometry and matching brace layers.
    #[test]
    fn colrv0_split_keeps_master_and_intermediate_color_layers() {
        let font =
            Font::load(&testdata_dir().join("glyphs3/COLRv0-2masters-brace.glyphs")).unwrap();
        let brace_shapes = font
            .glyphs
            .get("A")
            .unwrap()
            .layers
            .iter()
            .find(|l| l.layer_id == "cbrace")
            .unwrap()
            .shapes
            .clone();
        let (font, color_glyphs) = split_color_glyphs(font).unwrap();

        assert_eq!(
            color_glyphs.get("A").map(Vec::as_slice),
            Some(["A.color0".into(), "A.color1".into()].as_slice())
        );

        let original = font.glyphs.get("A").unwrap();
        // (split glyph, palette index, id of the Bold master's color layer)
        for (split_name, palette_idx, bold_layer_id) in
            [("A.color0", 1, "c03"), ("A.color1", 0, "c04")]
        {
            // A.color0 also carries the intermediate, checked below.
            let split_glyph = font.glyphs.get(split_name).unwrap();
            let masters: Vec<&Layer> = split_glyph
                .layers
                .iter()
                .filter(|l| l.is_master())
                .collect();
            assert_eq!(masters.len(), 2, "{split_name}");
            for (layer, expected_id) in masters.iter().zip(["m01", "m02"]) {
                assert_eq!(layer.layer_id, expected_id, "{split_name}");
                assert_eq!(
                    layer.attributes.color_palette,
                    Some(palette_idx),
                    "{split_name} {expected_id}"
                );
            }
            // the Bold layer must carry the Bold color layer's geometry
            let expected_shapes = &original
                .layers
                .iter()
                .find(|l| l.layer_id == bold_layer_id)
                .unwrap()
                .shapes;
            assert_eq!(&masters[1].shapes, expected_shapes, "{split_name}");
        }

        // the intermediate has colorPalette = 1, so it belongs to A.color0
        let color0 = font.glyphs.get("A.color0").unwrap();
        let brace = color0
            .layers
            .iter()
            .find(|l| l.is_intermediate())
            .expect("A.color0 should have an intermediate layer");
        assert_eq!(brace.associated_master_id.as_deref(), Some("m01"));
        assert_eq!(
            brace
                .attributes
                .coordinates
                .iter()
                .map(|c| c.0)
                .collect::<Vec<_>>(),
            vec![550.0]
        );
        assert_eq!(brace.shapes, brace_shapes);
        assert_eq!(color0.layers.len(), 3);

        // no intermediate with palette 0, so A.color1 has master layers only
        let color1 = font.glyphs.get("A.color1").unwrap();
        assert!(color1.layers.iter().all(|l| !l.is_intermediate()));
        assert_eq!(color1.layers.len(), 2);
    }

    fn square_path(dx: f64) -> glyphs_reader::Shape {
        let mut path = Path::new(true);
        for (x, y) in [
            (dx, 0.0),
            (dx + 100.0, 0.0),
            (dx + 100.0, 100.0),
            (dx, 100.0),
        ] {
            path.nodes.push(Node {
                pt: (x, y).into(),
                node_type: glyphs_reader::NodeType::Line,
            });
        }
        glyphs_reader::Shape::Path(path)
    }

    fn master_layer(master_id: &str, dx: f64) -> Layer {
        Layer {
            layer_id: master_id.to_string(),
            shapes: vec![square_path(dx)],
            ..Default::default()
        }
    }

    /// A color layer associated with a master; non-empty `coords` makes it an
    /// intermediate (brace) layer
    fn palette_layer(id: &str, master_id: &str, dx: f64, palette: i64, coords: &[f64]) -> Layer {
        Layer {
            layer_id: id.to_string(),
            associated_master_id: Some(master_id.to_string()),
            shapes: vec![square_path(dx)],
            attributes: LayerAttributes {
                color_palette: Some(palette),
                coordinates: coords.iter().map(|c| (*c).into()).collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn insert_glyph(font: &mut Font, name: &str, layers: Vec<Layer>) {
        let glyph = Glyph {
            name: name.into(),
            export: true,
            layers,
            ..Default::default()
        };
        font.glyphs.insert(name.into(), glyph);
        font.glyph_order.push(name.into());
    }

    /// A glyph whose only palette layer is an unmatched intermediate produces
    /// no split glyphs and must stay out of color_glyphs: an empty entry means
    /// "paint the (uncolored) base glyph itself" downstream. Palette
    /// intermediates are stripped from the base glyph either way, including
    /// when they never trip COLRv0 detection (glyph "B2": associated with a
    /// non-default master only).
    #[test]
    fn colrv0_unmatched_intermediate_does_not_make_base_a_color_glyph() {
        let mut font =
            Font::load(&testdata_dir().join("glyphs3/COLRv0-2masters-brace.glyphs")).unwrap();
        let master_id = font.default_master().id.clone();
        insert_glyph(
            &mut font,
            "B",
            vec![
                master_layer(&master_id, 0.0),
                palette_layer("bbrace", &master_id, 10.0, 0, &[550.0]),
            ],
        );
        insert_glyph(
            &mut font,
            "B2",
            vec![
                master_layer(&master_id, 0.0),
                palette_layer("b2brace", "m02", 10.0, 0, &[550.0]),
            ],
        );
        let (font, color_glyphs) = split_color_glyphs(font).unwrap();

        assert!(!color_glyphs.contains_key("B"));
        assert!(!font.glyphs.contains_key("B.color0"));
        for name in ["B", "B2"] {
            assert!(
                font.glyphs
                    .get(name)
                    .unwrap()
                    .layers
                    .iter()
                    .all(|l| !l.is_intermediate()),
                "{name} still carries an intermediate color layer"
            );
        }
    }

    /// A master layer that is itself a palette layer (glyphsLib reuses it as a
    /// color layer painting the base glyph) must keep its COLR entry even when
    /// the COLRv0 split produces no color glyphs.
    #[test]
    fn colrv0_unmatched_intermediate_keeps_colored_master_base() {
        let mut font =
            Font::load(&testdata_dir().join("glyphs3/COLRv0-2masters-brace.glyphs")).unwrap();
        let master_id = font.default_master().id.clone();
        insert_glyph(
            &mut font,
            "D",
            vec![
                Layer {
                    attributes: LayerAttributes {
                        color_palette: Some(0),
                        ..Default::default()
                    },
                    ..master_layer(&master_id, 0.0)
                },
                palette_layer("dbrace", &master_id, 10.0, 0, &[550.0]),
            ],
        );

        let (font, color_glyphs) = split_color_glyphs(font).unwrap();

        // an empty entry means "paint the base glyph itself", correct here
        // because the base is color-valued
        assert_eq!(color_glyphs.get("D"), Some(&vec![]));
        assert!(!font.glyphs.contains_key("D.color0"));
    }

    /// When several color layers share a palette index, the n-th intermediate
    /// with that index pairs with the n-th color layer with that index, in
    /// document order (glyphsLib's seen counter) -- not by index alone.
    #[test]
    fn colrv0_duplicate_palette_indices_pair_intermediates_by_occurrence() {
        let mut font =
            Font::load(&testdata_dir().join("glyphs3/COLRv0-2masters-brace.glyphs")).unwrap();
        let master_id = font.default_master().id.clone();
        insert_glyph(
            &mut font,
            "C",
            vec![
                master_layer(&master_id, 0.0),
                // intermediates listed before the color layers: document order
                // within each group drives the pairing, not adjacency
                palette_layer("brace0", &master_id, 10.0, 1, &[550.0]),
                palette_layer("brace1", &master_id, 20.0, 1, &[550.0]),
                // Global layer position differs from occurrence within palette 1.
                palette_layer("c_", &master_id, 50.0, 0, &[]),
                palette_layer("c0", &master_id, 30.0, 1, &[]),
                palette_layer("c1", &master_id, 40.0, 1, &[]),
            ],
        );

        let (font, color_glyphs) = split_color_glyphs(font).unwrap();

        assert_eq!(
            color_glyphs.get("C").map(Vec::as_slice),
            Some(["C.color0".into(), "C.color1".into(), "C.color2".into()].as_slice())
        );
        for (split_name, color_dx, brace_dx) in [("C.color1", 30.0, 10.0), ("C.color2", 40.0, 20.0)]
        {
            let layers = &font.glyphs.get(split_name).unwrap().layers;
            assert_eq!(layers.len(), 2, "{split_name}");
            assert_eq!(
                layers[0].shapes,
                vec![square_path(color_dx)],
                "{split_name}"
            );
            assert!(layers[1].is_intermediate(), "{split_name}");
            assert_eq!(
                layers[1].shapes,
                vec![square_path(brace_dx)],
                "{split_name}"
            );
        }
    }

    /// Test that COLRv1 glyphs with empty color layers are not added to color_glyphs.
    ///
    /// This is similar to the COLRv0 test but for the COLRv1 code path.
    #[test]
    fn colrv1_glyph_with_empty_color_layer_is_skipped() {
        let mut font = Font::load(&testdata_dir().join("glyphs3/COLRv1-gradient.glyphs")).unwrap();
        let master_id = font.default_master().id.clone();

        // Add a glyph "empty_color" with a color layer but no shapes
        let empty_glyph = Glyph {
            name: "empty_color".into(),
            export: true,
            layers: vec![
                // Default master layer - marked as color but empty shapes
                Layer {
                    layer_id: master_id.clone(),
                    associated_master_id: None,
                    width: 0.0.into(),
                    shapes: vec![], // Empty!
                    anchors: vec![],
                    attributes: LayerAttributes {
                        color: true, // This triggers COLRv1 path
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        font.glyphs.insert("empty_color".into(), empty_glyph);
        font.glyph_order.push("empty_color".into());

        // this would add the glyph incorrectly with old code
        let (_, color_glyphs) = split_color_glyphs(font).unwrap();

        // The glyph should NOT be in color_glyphs because it has no shapes
        assert!(
            !color_glyphs.contains_key("empty_color"),
            "COLRv1 glyph with empty color layer should not be added to color_glyphs"
        );
    }

    /// When multiple user-space values map to the same design-space value
    /// (a many-to-one axis map), the axis max should reflect the largest
    /// user-space value, not the result of a lossy design-to-user round-trip.
    /// https://github.com/googlefonts/ufo2ft/issues/978
    #[test]
    fn many_to_one_axis_map_preserves_max() {
        let font = Font::load(&testdata_dir().join("glyphs3/ManyToOneAxisMap.glyphs")).unwrap();
        let font_info = FontInfo::try_from(font).unwrap();
        let wght_tag = write_fonts::types::Tag::from_str("wght").unwrap();
        let wght = font_info.axes.get(&wght_tag).unwrap();
        // user=900 and user=1000 both map to design=1000;
        // axis max must be 1000 (the largest user value), not 900
        assert_eq!(wght.max, fontdrasil::coords::UserCoord::new(1000.0));
    }

    /// The default master has no user location of its own; it's whatever the
    /// mapping says its design location is, read backwards.
    #[test]
    fn user_coord_reverses_the_mapping() {
        use fontdrasil::coords::{DesignCoord, UserCoord};

        let mappings = [
            (UserCoord::new(300.0), DesignCoord::new(66.0)),
            (UserCoord::new(400.0), DesignCoord::new(86.0)),
            (UserCoord::new(700.0), DesignCoord::new(86.0)),
        ];
        // glyphsLib reverses into a dict keyed by design, so the *last* user
        // value for a repeated design value is the one that survives
        assert_eq!(
            super::to_user_coord(&mappings, DesignCoord::new(86.0)),
            UserCoord::new(700.0)
        );
        // between vertices we interpolate...
        assert_eq!(
            super::to_user_coord(&mappings, DesignCoord::new(76.0)),
            UserCoord::new(500.0)
        );
        // ...and off the end we extrapolate by offset, as fontTools does
        assert_eq!(
            super::to_user_coord(&mappings, DesignCoord::new(65.0)),
            UserCoord::new(299.0)
        );
    }

    /// A Glyphs 2 source that uses the Width axis as an italic toggle leaves
    /// every instance on the default widthClass, so the mapping pins the axis
    /// to one user value and half the masters sit at a design value it never
    /// names.
    ///
    /// glyphsLib writes exactly this axis
    ///
    /// ```xml
    /// <axis tag="wdth" name="Width" minimum="100" maximum="100" default="100">
    ///   <map input="100" output="1"/>
    /// </axis>
    /// ```
    ///
    /// and designspaceLib then hands varLib only the Width=1 sources, with the
    /// Width=1 Regular as the default.
    #[test]
    fn width_axis_pinned_by_instances() {
        use fontdrasil::coords::UserCoord;

        let font =
            Font::load(&testdata_dir().join("glyphs2/WidthPinnedByInstances.glyphs")).unwrap();
        let font_info = FontInfo::try_from(font).unwrap();

        let wdth = font_info
            .axes
            .get(&write_fonts::types::Tag::from_str("wdth").unwrap())
            .unwrap();
        assert_eq!(
            (wdth.min, wdth.default, wdth.max),
            (
                UserCoord::new(100.0),
                UserCoord::new(100.0),
                UserCoord::new(100.0)
            )
        );
        assert_eq!(
            wdth.converter
                .iter()
                .map(|(user, design, _)| (user.to_f64(), design.to_f64()))
                .collect::<Vec<_>>(),
            vec![(100.0, 1.0)]
        );

        // the Weight axis, which nothing pins, is untouched
        let wght = font_info
            .axes
            .get(&write_fonts::types::Tag::from_str("wght").unwrap())
            .unwrap();
        assert_eq!(
            (wght.min, wght.default, wght.max),
            (
                UserCoord::new(400.0),
                UserCoord::new(400.0),
                UserCoord::new(700.0)
            )
        );

        // the Width=0 masters are outside the axis and aren't in the font,
        // and neither are their layers or the instances that sit with them
        assert_eq!(
            font_info
                .font
                .masters
                .iter()
                .map(|master| master.id.as_str())
                .collect::<Vec<_>>(),
            vec!["italic-regular", "italic-bold"]
        );
        assert_eq!(
            font_info
                .font
                .instances
                .iter()
                .map(|instance| instance.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Italic", "Bold Italic"]
        );
        assert_eq!(
            font_info.font.glyphs["hyphen"]
                .layers
                .iter()
                .map(|layer| layer.layer_id.as_str())
                .collect::<Vec<_>>(),
            vec!["italic-regular", "italic-bold"]
        );

        // the default master went with them, so the survivor at the default
        // location takes over
        assert_eq!(font_info.font.default_master().id, "italic-regular");
    }

    /// A Weight axis' user (min, default, max), its converter's (user, design)
    /// points, and the ids of the masters that survived.
    type WghtAxisAndMasters<'a> = ((f64, f64, f64), Vec<(f64, f64)>, Vec<&'a str>);

    /// The Weight axis and masters of a [`FontInfo`], for the tests below.
    fn wght_axis_and_masters(font_info: &FontInfo) -> WghtAxisAndMasters<'_> {
        let wght = font_info
            .axes
            .get(&write_fonts::types::Tag::from_str("wght").unwrap())
            .unwrap();
        (
            (wght.min.to_f64(), wght.default.to_f64(), wght.max.to_f64()),
            wght.converter
                .iter()
                .map(|(user, design, _)| (user.to_f64(), design.to_f64()))
                .collect(),
            font_info
                .font
                .masters
                .iter()
                .map(|master| master.id.as_str())
                .collect(),
        )
    }

    /// A Glyphs 3 source with no "Axis Location" or "Axis Mappings", whose only
    /// exporting instance is a Medium at design 483 with weightClass 500.
    ///
    /// glyphsLib reads that weightClass as the instance's user location, and so
    /// pins the Weight axis to user 500, which neither master (400, 700) is on;
    /// reading the range off that mapping dropped both, and with them the default
    /// master. Glyphs 3.5 and 4.1.1 both export the source as a 400-700 axis, user
    /// == design, with no avar and the Medium at 483, and so do we.
    #[test]
    fn one_exporting_instance_keeps_the_masters() {
        let font = Font::load(&testdata_dir().join("glyphs3/OneExportingInstance.glyphs")).unwrap();
        let font_info = FontInfo::try_from(font).unwrap();

        let (range, converter, masters) = wght_axis_and_masters(&font_info);
        assert_eq!(range, (400.0, 400.0, 700.0));
        // user == design, so there's nothing for avar to say
        assert!(
            converter.iter().all(|(user, design)| user == design),
            "{converter:?}"
        );
        assert_eq!(masters, vec!["m01", "E09E0C54-128D-4FEA-B209-1B70BEFE300B"]);
        assert_eq!(font_info.font.default_master().id, "m01");
        assert_eq!(
            font_info
                .font
                .instances
                .iter()
                .map(|instance| (instance.name.as_str(), instance.active))
                .collect::<Vec<_>>(),
            vec![("Regular", false), ("Medium", true)]
        );
    }

    /// A Glyphs 3 source's "Axis Mappings" can reach past its masters; the axis is
    /// still the masters' span. Glyphs exports this one, a 100-900 mapping over
    /// masters at 400 and 700, as a 400-700 axis with avar 0.3333 -> 0.2767: the
    /// mapping's 500 -> 483, and nothing beyond the masters.
    #[test]
    fn glyphs3_axis_mappings_past_the_masters() {
        let raw =
            std::fs::read_to_string(testdata_dir().join("glyphs3/OneExportingInstance.glyphs"))
                .unwrap()
                .replace(
                    "familyName = WghtVar;",
                    "customParameters = (\n{\nname = \"Axis Mappings\";\nvalue = {\nwght = {\n\
                 100 = 300;\n400 = 400;\n500 = 483;\n700 = 700;\n900 = 800;\n};\n};\n}\n);\n\
                 familyName = WghtVar;",
                );
        let font = Font::load_from_string(&raw).unwrap();
        let font_info = FontInfo::try_from(font).unwrap();

        let (range, converter, masters) = wght_axis_and_masters(&font_info);
        assert_eq!(range, (400.0, 400.0, 700.0));
        assert_eq!(
            converter,
            vec![(400.0, 400.0), (500.0, 483.0), (700.0, 700.0)]
        );
        assert_eq!(masters, vec!["m01", "E09E0C54-128D-4FEA-B209-1B70BEFE300B"]);
        assert_eq!(font_info.font.default_master().id, "m01");
    }

    /// The same design in Glyphs 2 still reads its user space as glyphsLib does,
    /// off the instance's weightClass. That leaves no master at the default, a
    /// source fontmake can't build at all, so rather than fail we take the range
    /// off the masters, read through that mapping, and keep them all.
    #[test]
    fn glyphs2_mapping_that_leaves_no_default_master() {
        let raw = std::fs::read_to_string(testdata_dir().join("glyphs2/WghtVar_Instances.glyphs"))
            .unwrap();
        let start = raw.find("instances = (").unwrap();
        let end = raw[start..].find("\n);\n").unwrap() + start + "\n);\n".len();
        let raw = format!(
            "{}instances = (\n{{\nexports = 0;\ninterpolationWeight = 400;\nname = Regular;\n}},\n\
             {{\ninterpolationWeight = 483;\nname = Medium;\nweightClass = Medium;\n}}\n);\n{}",
            &raw[..start],
            &raw[end..]
        );
        let font = Font::load_from_string(&raw).unwrap();
        assert!(font.is_glyphs2());
        // glyphsLib's reading: one point, user 500 -> design 483
        assert_eq!(
            font.axis_mappings
                .get("Weight")
                .unwrap()
                .iter()
                .map(|(user, design)| (user.into_inner(), design.into_inner()))
                .collect::<Vec<_>>(),
            vec![(500.0, 483.0)]
        );
        let font_info = FontInfo::try_from(font).unwrap();

        let (range, converter, masters) = wght_axis_and_masters(&font_info);
        // the masters' span, read through the mapping
        assert_eq!(range, (417.0, 417.0, 717.0));
        assert_eq!(
            converter,
            vec![(417.0, 400.0), (500.0, 483.0), (717.0, 700.0)]
        );
        assert_eq!(masters, vec!["m01", "E09E0C54-128D-4FEA-B209-1B70BEFE300B"]);
        assert_eq!(font_info.font.default_master().id, "m01");
    }

    /// Dropping an axis and dropping a master meet here: a master's `axesValues`
    /// still has a slot for every axis the source declared, dropped ones included,
    /// so reading one back has to skip the gaps rather than count from the left.
    ///
    /// This source drops the *middle* axis - an inert Width - and keeps the Custom
    /// axis after it, while its Weight axis is pinned by its instances so one master
    /// falls outside it. glyphsLib agrees: Weight 400/400/400 mapped to design 65,
    /// Custom 10/10/10, and no Width at all.
    #[test]
    fn a_dropped_axis_does_not_shift_the_masters_that_outlive_it() {
        use fontdrasil::coords::UserCoord;

        let font =
            Font::load(&testdata_dir().join("glyphs2/WeightPinnedWithCustomAxis.glyphs")).unwrap();
        let font_info = FontInfo::try_from(font).unwrap();

        assert_eq!(
            font_info
                .axes
                .iter()
                .map(|axis| axis.tag.to_string())
                .collect::<Vec<_>>(),
            vec!["wght", "XXXX"],
            "the inert Width between them is gone"
        );
        // read through the gap, Custom is still the 10 the masters state; read past it,
        // it would be the 100 of the Width axis that isn't there any more
        let custom = font_info
            .axes
            .get(&write_fonts::types::Tag::from_str("XXXX").unwrap())
            .unwrap();
        assert_eq!(
            (custom.min, custom.default, custom.max),
            (
                UserCoord::new(10.0),
                UserCoord::new(10.0),
                UserCoord::new(10.0)
            )
        );

        // the mapping only reaches design 65, so the master at 151 is outside the axis
        assert_eq!(
            font_info
                .font
                .masters
                .iter()
                .map(|master| master.id.as_str())
                .collect::<Vec<_>>(),
            vec!["hollow"]
        );
        assert_eq!(font_info.font.default_master().id, "hollow");
    }

    /// Test that a layer with palette index 0xFFFF produces a PaintSolid with color `None`.
    #[test]
    fn palette_index_0xffff() {
        use super::to_ir_paint;
        use fontir::ir::Paint;
        use glyphs_reader::ShapeAttributes;

        let layer = Layer {
            attributes: LayerAttributes {
                color_palette: Some(0xFFFF),
                ..Default::default()
            },
            ..Default::default()
        };
        let attr = ShapeAttributes::default();
        let paint = to_ir_paint(None, "test", &layer, &attr).unwrap();
        match paint {
            Paint::Solid(solid) => {
                assert_eq!(solid.color, None, "expected foreground paint (color: None)");
            }
            other => panic!("expected Paint::Solid, got {other:?}"),
        }
    }
}
