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
) -> Result<ir::FeaturesSource, Error> {
    // Based on https://github.com/googlefonts/glyphsLib/blob/24b4d340e4c82948ba121dcfe563c1450a8e69c9/Lib/glyphsLib/builder/features.py#L74
    // TODO: token expansion
    // TODO: implement notes
    let fea_snippets: Vec<_> = features.iter().filter_map(|f| f.str_if_enabled()).collect();
    Ok(ir::FeaturesSource::Memory {
        fea_content: fea_snippets.join("\n\n"),
        include_dir,
    })
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
    let design_to_user = PiecewiseLinearMap::new(by_design.into_iter().collect());
    UserCoord::new(design_to_user.map(design.into_inner()))
}

/// Convert .glyphs axes to IR axes.
///
///  See <https://github.com/googlefonts/glyphsLib/blob/6f243c1f732ea1092717918d0328f3b5303ffe56/Lib/glyphsLib/builder/axes.py#L155>
fn to_ir_axis(
    font: &Font,
    axis_values: &[OrderedFloat<f64>],
    default_idx: usize,
    axis: &glyphs_reader::Axis,
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

    // A mapped axis takes its user-space extremes from the mapping itself, never from
    // the masters: instances contribute mappings too, so the mapped range can reach
    // past the masters, and a master can sit at a design value the mapping never names.
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L284-L285>
    // <https://github.com/googlefonts/fontc/issues/1991>
    //
    // The default master's user location is then the reverse of its design location,
    // clamped into that range.
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L259-L263>
    // <https://github.com/googlefonts/glyphsLib/blob/6.13.1/Lib/glyphsLib/builder/axes.py#L286>
    let mapped = (!mappings.is_empty()).then(|| {
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
        if let Some((user_min, user_default, user_max)) = mapped {
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

    for master in font.masters.iter().filter(|m| dropped.contains(&m.id)) {
        warn!(
            "Master '{}' is outside the axis ranges the mapping defines; dropping it",
            master.name
        );
    }
    let default_master_id = font.default_master().id.clone();
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

    // If the default master went with them the survivor at the default
    // location takes over, as designspaceLib's `subDoc.findDefault()` does.
    font.default_master_idx = if dropped.contains(&default_master_id) {
        let at_default = |axes_values: &[OrderedFloat<f64>]| {
            axes.iter().zip(axis_indices).all(|(axis, &idx)| {
                axes_values.get(idx).is_some_and(|value| {
                    *value == axis.default.to_design(&axis.converter).into_inner()
                })
            })
        };
        font.masters
            .iter()
            .position(|master| at_default(&master.axes_values))
            .ok_or(Error::NoDefaultMaster)?
    } else {
        #[allow(clippy::unwrap_used)] // it wasn't dropped, so it's still there
        font.masters
            .iter()
            .position(|master| master.id == default_master_id)
            .unwrap()
    };
    Ok(())
}

fn ir_axes(font: &Font) -> Result<(fontdrasil::types::Axes, Vec<usize>), Error> {
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
        let axis = to_ir_axis(font, &axis_values, font.default_master_idx, glyphs_axis)?;
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
        // The axes are read off every master, as glyphsLib does, and only then
        // do the sources the axes can't reach get dropped.
        let (axes, axis_indices) = ir_axes(&font)?;
        drop_sources_outside_axes(&mut font, &axes, &axis_indices)?;

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
    Radial(
        Vec<OrderedFloat<f64>>,
        Vec<OrderedFloat<f64>>,
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
            if gradient.style == "circle" {
                return Colrv1RunType::Radial(
                    gradient.start.clone(),
                    gradient.end.clone(),
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
        ..Default::default()
    };
    *nth += 1;
    new_glyph
}

fn split_colrv0_glyph(
    original: &Glyph,
    default_master_layer: &Layer,
    color_glyphs: &mut IndexMap<SmolStr, Vec<SmolStr>>,
    additions: &mut Vec<(SmolStr, Glyph)>,
) -> Result<(), Error> {
    let glyph_name = &original.name;

    // COLRv0 runs are just consecutive shapes by palette index
    // The original glyph becomes uncolored,
    // each color run becomes a new glyph named [original].color[i]
    let mut nth = 0;
    for layer in original.layers.iter() {
        if layer.shapes.is_empty()
            || layer.attributes.color_palette.is_none()
            || layer.associated_master_id.as_deref() != Some(default_master_layer.layer_id.as_str())
        {
            continue;
        }

        // Every layer associated with the master that has a palette index becomes a new color glyph
        let mut new_glyph = new_color_glyph(original, &mut nth);
        let mut layer = layer.clone();
        layer.layer_id = layer.associated_master_id.take().unwrap();
        new_glyph.layers.push(layer);

        debug!("Add COLRv0 {}", new_glyph.name);

        color_glyphs
            .entry(glyph_name.clone())
            .or_default()
            .push(new_glyph.name.clone());
        additions.push((new_glyph.name.clone(), new_glyph));
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

    let mut additions: Vec<(SmolStr, Glyph)> = Vec::new();
    for glyph in font.glyphs.values_mut() {
        let Some(default_master_layer) = glyph
            .layers
            .iter()
            .find(|l| l.layer_id == default_master_id)
        else {
            continue;
        };

        // If 1..N layers with palette indices are associated this is COLRv0
        // See <https://github.com/googlefonts/glyphsLib/blob/99328059ec4799956ecef3d47ebcc13ae70dacff/Lib/glyphsLib/builder/glyph.py#L289-L292>
        if glyph.layers.iter().any(|l| {
            l.attributes.color_palette.is_some()
                && l.associated_master_id.as_deref() == Some(default_master_layer.layer_id.as_str())
        }) {
            split_colrv0_glyph(
                glyph,
                default_master_layer,
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
        } else {
            // Not color
            continue;
        }

        // For COLRv1 single-run glyphs (i.e. no split glyphs created, shapes in default layer),
        // reserve an entry with empty vec so it gets included in COLR (see ColorGlyphsWork::exec).
        // For COLRv0 and v1 multi-run, an non-empty vec already exists from the split_colr* funcs.
        if !default_master_layer.shapes.is_empty() {
            color_glyphs.entry(glyph.name.clone()).or_default();
        }
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
    use glyphs_reader::{Font, Glyph, Layer, LayerAttributes, Node, Path};
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
