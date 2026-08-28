//! Global font metadata

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    io::Read,
};

use chrono::{DateTime, Utc};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use write_fonts::{
    tables::{gasp::GaspRange, gdef::GlyphClassDef, head, os2::SelectionFlags},
    types::{NameId, Tag},
};

use fontdrasil::{
    coords::{DesignCoord, NormalizedCoord, NormalizedLocation, UserLocation},
    types::{Axes, Axis, GlyphName},
    variations::{VariationModel, VariationModelError},
};

use super::GlobalMetric;
use super::feature_writers::FeatureWriterSpec;
use crate::orchestration::Persistable;

/// Glyph names mapped to postscript names
pub type PostscriptNames = HashMap<GlyphName, GlyphName>;

/// Glyphsapp only: the glyph attributes a FEA glyph predicate token can compare.
///
/// A Glyphs.app source can select glyphs into a FEA class with a predicate
/// token, e.g. `@lc = [ $[category == "Letter" && case == lower] ];`. Every
/// value here is the string the *source* stored (`None` when it stored
/// nothing), because that is what glyphsLib compares: its `TokenExpander` reads
/// plain `GSGlyph` attributes and never fontc's GlyphData-derived fallbacks.
/// See <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/tokens.py>.
// NOTE: no `skip_serializing_if` here. IR is also written with bincode, which
// is not self-describing: a field skipped on the way out is still read on the
// way back in, and the whole stream desynchronizes.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct GlyphPredicateAttrs {
    /// `category = Letter;`
    pub category: Option<SmolStr>,
    /// `subCategory = Ligature;`
    pub sub_category: Option<SmolStr>,
    /// `case = lower;`
    pub case: Option<SmolStr>,
    /// The glyph's first codepoint, `%04X`-formatted like glyphsLib's
    /// `GSGlyph.unicode`.
    pub unicode: Option<SmolStr>,
}

/// Global font info that cannot vary across the design space.
///
/// For example, upem, axis definitions, etc, as distinct from
/// metadata that varies across design space such as ascender/descender.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct StaticMetadata {
    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/head>.
    pub units_per_em: u16,

    /// Every axis used by the font being compiled, including point axes.
    ///
    /// This is relatively rarely what you want.
    pub all_source_axes: Axes,

    /// Every variable (non-point) axis used by the font being compiled.
    ///
    /// If empty this is a static font.
    pub axes: Axes,

    /// Named locations in variation space
    pub named_instances: Vec<NamedInstance>,

    /// A model of how variation space is split into regions that have deltas.
    ///
    /// This copy includes all locations used in the entire font. That is, every
    /// location any glyph has an instance. Use of a location not in the global model
    /// is an error. This model enforces the no delta at the default location constraint
    /// used in things like gvar.
    pub variation_model: VariationModel,
    /// Glyphsapp only; named numbers defined per-master
    pub number_values: HashMap<NormalizedLocation, BTreeMap<SmolStr, OrderedFloat<f64>>>,
    /// Glyphsapp only; what a FEA glyph predicate token may ask about a glyph.
    ///
    /// `None` for every source that is not a Glyphs.app file, so that a
    /// predicate asking about anything but the glyph name is an error there
    /// rather than a silent empty match. `Some` holds one entry per glyph that
    /// sets at least one attribute; a glyph that sets none is simply absent.
    #[serde(default)]
    pub glyph_predicate_attrs: Option<BTreeMap<GlyphName, GlyphPredicateAttrs>>,
    /// PostScript-specific data per master, feeding the CFF table.
    ///
    /// Keyed like [`Self::number_values`]: one entry per master that defines
    /// any, at that master's location. Read it with [`Self::postscript_at`] or
    /// [`Self::postscript_default`] rather than indexing directly; sources that
    /// have no PostScript data at all (fontra) leave the map empty.
    #[serde(default)]
    pub postscript: HashMap<NormalizedLocation, PostscriptSettings>,
    default_location: NormalizedLocation,

    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/name>.
    pub names: HashMap<NameKey, String>,

    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/post> and
    /// <https://github.com/adobe-type-tools/agl-specification>
    pub postscript_names: Option<PostscriptNames>,

    /// Italic angle in counter-clockwise degrees from the vertical. Zero for
    /// upright fonts, negative for right-leaning fonts.
    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/post>.
    pub italic_angle: OrderedFloat<f64>,

    /// Records whether this font contains sufficient non-default vertical data
    /// to warrant building a vhea and vmtx table. (The criteria for Glyphs and
    /// UFO sources is different.)
    pub build_vertical: bool,

    /// Miscellaneous font-wide data that didn't seem worthy of top billing
    pub misc: MiscMetadata,

    /// Feature variation rules
    pub variations: Option<VariableFeature>,
}

/// IR for a named position in variation space
///
/// A variable font uses only [`Self::name`], [`Self::postscript_name`] and
/// [`Self::location`], for `fvar`. The rest is what an instance *built* here is
/// called, which only `--instance` reads. All of it is `#[serde(default)]`
/// because IR is schema-less YAML and additive fields are how it grows.
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct NamedInstance {
    /// The style name: designspace `<instance stylename>`, or a Glyphs
    /// instance's `name`.
    pub name: String,
    pub postscript_name: Option<String>,
    pub location: UserLocation,
    /// The family name of the UFO fontmake would interpolate here.
    ///
    /// The instance's own — designspace `<instance familyname>`, or a Glyphs
    /// instance's `familyNames` property — falling back to the font's, which
    /// is what the instance UFO inherits. Name ID 16 is exactly this: the
    /// instance never inherits `openTypeNamePreferredFamilyName`, so ufo2ft's
    /// fallback chain always lands on `familyName`.
    #[serde(default)]
    pub family_name: Option<String>,
    /// Name ID 1 for this instance, if the source states one.
    ///
    /// designspace `<instance stylemapfamilyname>`; for Glyphs, built by
    /// glyphsLib from the instance's style linking. Absent means "let the
    /// RIBBI fallback decide", which is what [`NameBuilder::build`] already
    /// does.
    ///
    /// [`NameBuilder::build`]: crate::ir::NameBuilder::build
    #[serde(default)]
    pub style_map_family_name: Option<String>,
    /// Name ID 2 for this instance, if the source states one.
    ///
    /// Kept as the source's own string rather than as a [`StyleMapStyle`]:
    /// ufo2ft's instantiator writes a `stylemapstylename` that *isn't* one of
    /// the four through to the instance UFO anyway — it only logs "may cause
    /// problems in some applications" — and the compiler then title-cases it
    /// straight into name id 2 while setting **no** RIBBI bit in `fsSelection`
    /// or `head.macStyle`. Doto's `@default` is exactly that: style map style
    /// `Black`, name id 2 `Black`, `fsSelection` `0x0080`.
    ///
    /// Read it with [`Self::style_map_style`] for the flags and
    /// [`Self::style_map_style_display`] for the name.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L775-L792>
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L404>
    #[serde(default)]
    pub style_map_style_name: Option<String>,
    /// What the instance's *own* Glyphs custom parameters say to override.
    ///
    /// Empty for a source that has none, and for anything but `--instance`:
    /// nothing outside the pin reads it.
    #[serde(default)]
    pub overrides: InstanceOverrides,
}

impl NamedInstance {
    /// The style linking this instance's name id 2 implies, if it implies any.
    ///
    /// `None` for a `styleMapStyleName` that isn't one of the four: ufo2ft
    /// sets no RIBBI bit for it at all.
    pub fn style_map_style(&self) -> Option<StyleMapStyle> {
        StyleMapStyle::parse(self.style_map_style_name.as_deref()?)
    }

    /// `styleMapStyleName` as name id 2 spells it.
    ///
    /// ufo2ft lowercases the UFO attribute on the way in and `.title()`s it on
    /// the way out, so `BLACK`, `black` and `Black` all end up `Black`.
    pub fn style_map_style_display(&self) -> Option<String> {
        self.style_map_style_name.as_deref().map(title_case)
    }
}

/// Python's `str.title()`: capitalise every run of letters, lowercase the rest.
fn title_case(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_word = false;
    for c in raw.chars() {
        if c.is_alphabetic() {
            if in_word {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            in_word = true;
        } else {
            out.push(c);
            in_word = false;
        }
    }
    out
}

/// What an instance's own Glyphs custom parameters and properties override.
///
/// glyphsLib's `apply_instance_data_to_ufo` runs `to_ufo_custom_params` over
/// the *instance's* parameters on the already-interpolated UFO, so every one of
/// them wins over whatever the masters produced — and it runs for
/// `.designspace` sources as much as for `.glyphs` ones, because fontmake
/// stashes the parameters in the designspace `<instance><lib>` and replays them
/// from there. Only [`crate::instance`] reads this.
///
/// Parameters whose effect is on *global metrics* live in
/// [`Self::metrics`]; those are applied where the metrics are pinned, in
/// [`GlobalMetricsBuilder::build_pinned`](crate::ir::GlobalMetricsBuilder::build_pinned).
///
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/instances.py#L454-L470>
/// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/custom_params.py#L314-L448>
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct InstanceOverrides {
    /// `panose` / `openTypeOS2Panose`, which *replaces* the merged-across-masters
    /// PANOSE an interpolated instance would otherwise get.
    pub panose: Option<Panose>,
    /// `fsType` / `openTypeOS2Type`.
    pub fs_type: Option<u16>,
    /// `isFixedPitch` / `postscriptIsFixedPitch`.
    pub is_fixed_pitch: Option<bool>,
    /// `unicodeRanges` / `openTypeOS2UnicodeRanges`.
    pub unicode_range_bits: Option<HashSet<u32>>,
    /// `codePageRanges`.
    pub codepage_range_bits: Option<HashSet<u32>>,
    /// `meta Table`, which becomes `public.openTypeMeta`.
    pub meta_table: Option<MetaTableValues>,
    /// `Use Typo Metrics`, i.e. `fsSelection` bit 7.
    pub use_typo_metrics: Option<bool>,
    /// `Has WWS Names`, i.e. `fsSelection` bit 8.
    pub has_wws_names: Option<bool>,
    /// `Don't use Production Names`, already negated into ufo2ft's
    /// `useProductionNames`.
    pub use_production_names: Option<bool>,
    /// Metrics the instance states outright, overriding the interpolation.
    pub metrics: BTreeMap<GlobalMetric, OrderedFloat<f64>>,
    /// Names the instance states as *fontinfo*, so they also drive the
    /// fallbacks: `preferredFamilyName` (16), `preferredSubfamilyName` (17),
    /// `compatibleFullName` (18), `WWSFamilyName` (21), `WWSSubfamilyName`
    /// (22). Windows/English, like everything `NameBuilder` computes.
    pub names: BTreeMap<NameId, String>,
    /// `Name Table Entry`, i.e. `openTypeNameRecords`.
    ///
    /// Any id on any platform, and applied *after* the table is built — these
    /// override the computed records and never feed them, which is the order
    /// `outlineCompiler.setupTable_name` uses.
    pub name_records: BTreeMap<NameKey, String>,
    /// `postscriptFullName`, i.e. the CFF `FullName` operator.
    ///
    /// Not a name-table record: ufo2ft reads `postscriptFullName` only when it
    /// builds the CFF Top DICT. An interpolated instance never inherits the
    /// masters' — it is neither a `MathInfo` attribute nor on ufo2ft's copy
    /// whitelist — so the instance's own is the only one that can reach a
    /// `--flavor otf` build.
    #[serde(default)]
    pub postscript_full_name: Option<String>,
}

/// The four style-linking styles, i.e. UFO `styleMapStyleName`.
///
/// <https://unifiedfontobject.org/versions/ufo3/fontinfo.plist/#generic-identification-information>
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StyleMapStyle {
    Regular,
    Italic,
    Bold,
    BoldItalic,
}

impl StyleMapStyle {
    /// The `Regular` / `Bold Italic` form, which is what name ID 2 holds.
    ///
    /// ufo2ft `.title()`s `styleMapStyleName` on the way into the name table,
    /// so the record is title case whatever the UFO's casing was.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L404>
    pub fn to_name(self) -> &'static str {
        match self {
            StyleMapStyle::Regular => "Regular",
            StyleMapStyle::Italic => "Italic",
            StyleMapStyle::Bold => "Bold",
            StyleMapStyle::BoldItalic => "Bold Italic",
        }
    }

    /// The `fsSelection` bits this style contributes, which are also macStyle's.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/outlineCompiler.py#L714-L725>
    pub fn selection_flags(self) -> SelectionFlags {
        match self {
            StyleMapStyle::Regular => SelectionFlags::REGULAR,
            StyleMapStyle::Italic => SelectionFlags::ITALIC,
            StyleMapStyle::Bold => SelectionFlags::BOLD,
            StyleMapStyle::BoldItalic => SelectionFlags::BOLD | SelectionFlags::ITALIC,
        }
    }

    /// Parse a UFO `styleMapStyleName`, which ufo2ft lowercases first.
    ///
    /// `None` for anything that isn't one of the four: ufo2ft logs "not one of
    /// ..." and leaves the attribute unset.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L778-L792>
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "regular" => Some(StyleMapStyle::Regular),
            "italic" => Some(StyleMapStyle::Italic),
            "bold" => Some(StyleMapStyle::Bold),
            "bold italic" => Some(StyleMapStyle::BoldItalic),
            _ => None,
        }
    }

    /// glyphsLib's flag-driven derivation, which never looks at the style name.
    ///
    /// <https://github.com/googlefonts/glyphsLib/blob/main/Lib/glyphsLib/builder/names.py#L77-L82>
    pub fn from_flags(is_bold: bool, is_italic: bool) -> Self {
        match (is_bold, is_italic) {
            (true, true) => StyleMapStyle::BoldItalic,
            (true, false) => StyleMapStyle::Bold,
            (false, true) => StyleMapStyle::Italic,
            (false, false) => StyleMapStyle::Regular,
        }
    }
}

/// See <https://learn.microsoft.com/en-us/typography/opentype/spec/name>
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NameKey {
    pub name_id: NameId,
    pub platform_id: u16,
    pub encoding_id: u16,
    pub lang_id: u16,
}

impl NameKey {
    /// Create's a [NameKey] suitable for use with the provided value.
    ///
    /// The value matters because if it uses values from outside the Unicode BMP
    /// the key changes.
    pub fn new(name_id: NameId, value: &str) -> NameKey {
        // The spec offers a Unicode platform but fontmake uses Windows because that's more widely supported.
        // Match that. <https://github.com/googlefonts/ufo2ft/blob/fca66fe3ea1ea88ffb36f8264b21ce042d3afd05/Lib/ufo2ft/outlineCompiler.py#L430-L432>.
        NameKey {
            platform_id: 3, // Windows
            encoding_id: Self::encoding_for(value),
            // https://learn.microsoft.com/en-us/typography/opentype/spec/name#windows-language-ids
            lang_id: 0x409, // English, United States.
            name_id,
        }
    }

    pub fn new_with_lang(name_id: NameId, value: &str, lang_id: u16) -> NameKey {
        NameKey {
            platform_id: 3,
            encoding_id: Self::encoding_for(value),
            lang_id,
            name_id,
        }
    }

    /// The encoding for a Windows-platform (which works everywhere) name.
    ///
    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/name#platform-specific-encoding-and-language-ids-windows-platform-platform-id-3>
    fn encoding_for(value: &str) -> u16 {
        if value.chars().all(|c| (c as u32) < 0xFFFF) {
            1 // Unicode BMP
        } else {
            10 // Unicode full repetoire
        }
    }

    pub fn new_bmp_only(name_id: NameId) -> NameKey {
        Self::new(name_id, "")
    }
}

/// GDEF categories derived from source before anchor propagation.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct PreliminaryGdefCategories {
    /// A map of glyphs to categories from source.
    pub categories: BTreeMap<GlyphName, GlyphClassDef>,
    /// Controls whether final GDEF categories should be inferred from the presence
    /// of anchors (similarly to how glyphsLib does) or used as-is as defined in
    /// the source (as standard in DS+UFO workflows).
    pub infer_from_anchors: bool,
    /// All glyphs whose source category is Mark (any subCategory)
    ///
    /// Used during anchor propagation. Glyphs.app skips component anchor
    /// propagation for any Mark-category glyph that already has anchors,
    /// regardless of subCategory. This differs from `categories` which only
    /// contains GDEF Mark (Nonspacing/SpacingCombining); glyphs like "tilde"
    /// (Mark, Spacing) are not GDEF marks but still need to opt out of
    /// component propagation.
    #[serde(default)]
    pub mark_category_glyphs: BTreeSet<GlyphName>,
}

/// Final GDEF categories after anchor propagation has been applied.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct GdefCategories {
    /// A map of glyphs to categories.
    pub categories: BTreeMap<GlyphName, GlyphClassDef>,
}

/// Metadata primarily feeding the OS/2 table.
///
/// <https://learn.microsoft.com/en-us/typography/opentype/spec/os2>
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MiscMetadata {
    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fstype>
    pub fs_type: Option<u16>,

    /// If set, the value the source file specifically stated. Otherwise compiler can choose.
    ///
    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/post#header>
    pub is_fixed_pitch: Option<bool>,

    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fsselection>
    pub selection_flags: SelectionFlags,

    /// See <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#achvendid>
    pub vendor_id: Tag,

    /// `openTypeOS2VendorID` exactly as the source stated it, if it stated one.
    ///
    /// [`Self::vendor_id`] is `achVendID`, which is four bytes and which `Tag`
    /// pads a short id out to. Name id 3's fallback interpolates the *raw*
    /// attribute instead — ufo2ft only `ljust`s for OS/2 — so a source whose
    /// vendor id is a single space (Geom) gets `1.102; ;Geom-Regular`, which
    /// the padded tag cannot spell. The frontends already hand this string to
    /// [`NameBuilder::build`](crate::ir::NameBuilder::build); a pin that
    /// rebuilds the name table needs it too.
    #[serde(default)]
    pub raw_vendor_id: Option<String>,

    /// UFO appears to allow negative major versions.
    ///
    /// See <https://unifiedfontobject.org/versions/ufo3/fontinfo.plist/#generic-identification-information>
    pub version_major: i32,
    pub version_minor: u32,

    pub head_flags: head::Flags,
    pub lowest_rec_ppm: u16,

    pub created: Option<DateTime<Utc>>,

    // <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#sfamilyclass>
    pub family_class: Option<i16>,

    pub panose: Option<Panose>,

    /// The PANOSE an interpolated *instance* gets, which is not [`Self::panose`].
    ///
    /// ufo2ft merges every source's PANOSE into the instance element-wise,
    /// keeping a digit only when every source that has a PANOSE agrees about
    /// it and zeroing the rest, and dropping the whole thing when nothing
    /// survives — so two masters with different PANOSE produce an instance
    /// with none at all. Only `--instance` reads this; a variable build takes
    /// the default master's, as fontmake does.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L480-L502>
    #[serde(default)]
    pub instance_panose: Option<Panose>,

    // Allows source to explicitly control bits. <https://github.com/googlefonts/fontc/issues/1027>
    pub unicode_range_bits: Option<HashSet<u32>>,

    // Allows source to explicitly control bits. <https://github.com/googlefonts/fontc/issues/1027>
    pub codepage_range_bits: Option<HashSet<u32>>,
    pub meta_table: Option<MetaTableValues>,

    /// <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#usweightclass>
    ///
    /// If empty and there is a weight axis OS/2 will use the weight default
    pub us_weight_class: Option<u16>,
    /// <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#uswidthclass>
    ///
    /// If empty and there is a width axis OS/2 will use the width default
    pub us_width_class: Option<u16>,

    // <https://learn.microsoft.com/en-us/typography/opentype/spec/gasp>
    pub gasp: Vec<GaspRange>,

    /// The `com.github.googlei18n.ufo2ft.featureWriters` config.
    ///
    /// `None` means the key was absent (use the built-in defaults); `Some` fully
    /// replaces the defaults (an empty list disables all automatic features).
    pub feature_generation: Option<Vec<FeatureWriterSpec>>,
}

/// The `postscript*` keys of UFO fontinfo, mostly CFF hinting data.
///
/// For Glyphs sources the equivalent values are derived from alignment zones,
/// stems, and custom parameters, the way glyphsLib fills them into the UFOs
/// it generates.
///
/// Arrays are empty when the source provides none. Values are kept unrounded;
/// CFF compilation rounds them the same way ufo2ft does. One of these is
/// stored per master in [`StaticMetadata::postscript`]; CFF, which is
/// single-master, reads the default master's via
/// [`StaticMetadata::postscript_default`].
///
/// See <https://unifiedfontobject.org/versions/ufo3/fontinfo.plist/#postscript-specific-data>
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct PostscriptSettings {
    pub blue_values: Vec<OrderedFloat<f64>>,
    pub other_blues: Vec<OrderedFloat<f64>>,
    pub family_blues: Vec<OrderedFloat<f64>>,
    pub family_other_blues: Vec<OrderedFloat<f64>>,
    pub blue_scale: Option<OrderedFloat<f64>>,
    pub blue_shift: Option<OrderedFloat<f64>>,
    pub blue_fuzz: Option<OrderedFloat<f64>>,
    pub stem_snap_h: Vec<OrderedFloat<f64>>,
    pub stem_snap_v: Vec<OrderedFloat<f64>>,
    pub force_bold: Option<bool>,
    /// Becomes the CFF TopDict `Weight`; not necessarily a wght axis name.
    pub weight_name: Option<String>,
    /// This master's `openTypeOS2WeightClass`, kept only so that an
    /// interpolated instance can derive [`Self::weight_name`] from it.
    ///
    /// It is *not* the class that reaches OS/2 — for an instance that comes
    /// from the axis, and the two legitimately disagree — but fontMath's
    /// `_processPostscriptWeightName` runs on the interpolated value, so the
    /// per-master values have to survive to the pin. `None` for a `.glyphs`
    /// source: a glyphsLib master UFO states no weight class at all (verified
    /// with `glyphsLib.to_ufos`), so fontMath's answer there is genuinely
    /// absent.
    ///
    /// <https://github.com/robotools/fontMath/blob/0.10.0/Lib/fontMath/mathInfo.py#L154-L169>
    pub os2_weight_class: Option<OrderedFloat<f64>>,
    /// Becomes the CFF TopDict `FullName`.
    ///
    /// This is the source's `postscriptFullName`, which is neither the name
    /// table's full font name (id 4) nor its PostScript name (id 6), and which
    /// nothing but the CFF reads. When it is unset the CFF work falls back the
    /// way ufo2ft does.
    pub full_name: Option<String>,
    /// If set, overrides the computed CFF `defaultWidthX`.
    pub default_width_x: Option<OrderedFloat<f64>>,
    /// If set, overrides the computed CFF `nominalWidthX`.
    pub nominal_width_x: Option<OrderedFloat<f64>>,
}

/// Records that will go in the '[meta]' table.
///
/// This can be used to specify explicit languages a font is designed for,
/// as well as languages it is capable of supporting.
///
/// See [design and supported languages][dlng slng].
///
/// [meta]: https://learn.microsoft.com/en-us/typography/opentype/spec/meta
/// [dlng slng]: https://learn.microsoft.com/en-us/typography/opentype/spec/meta#dlng-and-slng-design-and-supported-languages
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct MetaTableValues {
    /// ScriptLangTags for the design languages
    pub dlng: Vec<SmolStr>,
    /// ScriptLangTags for the supported languages
    pub slng: Vec<SmolStr>,
}

/// PANOSE bytes
///
/// <https://learn.microsoft.com/en-us/typography/opentype/spec/os2#panose>
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Panose {
    pub family_type: u8,
    pub serif_style: u8,
    pub weight: u8,
    pub proportion: u8,
    pub contrast: u8,
    pub stroke_variation: u8,
    pub arm_style: u8,
    pub letterform: u8,
    pub midline: u8,
    pub x_height: u8,
}

impl Panose {
    /// The ten digits, in `OS/2` order.
    pub fn digits(&self) -> [u8; 10] {
        [
            self.family_type,
            self.serif_style,
            self.weight,
            self.proportion,
            self.contrast,
            self.stroke_variation,
            self.arm_style,
            self.letterform,
            self.midline,
            self.x_height,
        ]
    }

    pub fn from_digits(digits: [u8; 10]) -> Panose {
        Panose {
            family_type: digits[0],
            serif_style: digits[1],
            weight: digits[2],
            proportion: digits[3],
            contrast: digits[4],
            stroke_variation: digits[5],
            arm_style: digits[6],
            letterform: digits[7],
            midline: digits[8],
            x_height: digits[9],
        }
    }

    /// The PANOSE an instance interpolated from these masters gets.
    ///
    /// ufo2ft keeps a digit only when every master that *has* a PANOSE agrees
    /// about it, zeroes the rest, and drops the whole thing when nothing
    /// survives — so a family whose masters disagree everywhere produces
    /// instances with no PANOSE at all, and `OS/2` writes zeros. Masters with
    /// no PANOSE don't vote; if none of them has one, there is nothing to
    /// merge.
    ///
    /// <https://github.com/googlefonts/ufo2ft/blob/main/Lib/ufo2ft/instantiator.py#L480-L502>
    pub fn merged_for_instance<'a>(
        masters: impl IntoIterator<Item = &'a Panose>,
    ) -> Option<Panose> {
        let mut shared: Option<[u8; 10]> = None;
        for master in masters {
            let digits = master.digits();
            shared = Some(match shared {
                None => digits,
                Some(shared) => {
                    std::array::from_fn(|i| if shared[i] == digits[i] { shared[i] } else { 0 })
                }
            });
        }
        shared
            .filter(|digits| digits.iter().any(|digit| *digit != 0))
            .map(Panose::from_digits)
    }
}

/// A series of substitution rules to be applied to layout features
/// at specific points in design space.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct VariableFeature {
    /// The features that for which these rules should apply, as part of a
    /// [`FeatureVariations`] table.
    ///
    /// [`FeatureVariations`]: write_fonts::tables::layout::FeatureVariations
    pub features: Vec<Tag>,
    pub rules: Vec<Rule>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// sets of conditions that trigger this rule.
    ///
    /// Only one of these needs to be true for the substitutions to be applied.
    pub conditions: Vec<ConditionSet>,
    /// Substitutions to be applied if a condition matches.
    pub substitutions: Vec<Substitution>,
}

/// A glyph substitution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Substitution {
    /// The glyph to be substituted
    pub replace: GlyphName,
    /// The substitute glyph
    pub with: GlyphName,
}

/// A series of [`Condition`]s.
///
/// All conditions in the set must be true for it to to be applied.
///
/// This type can be constructed with `collect()` from an iterator of `Condition`.
/// The inner conditions are always sorted.
#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConditionSet(Vec<Condition>);

/// A range on an axis.
///
/// One of `min` or `max` must be set.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Condition {
    pub axis: Tag,
    /// The minimum position for this condition, in design space coordinates.
    pub min: Option<DesignCoord>,
    /// The maximum position for this condition, in design space coordinates.
    pub max: Option<DesignCoord>,
}

impl Condition {
    pub fn new(axis: Tag, min: Option<DesignCoord>, max: Option<DesignCoord>) -> Self {
        Self { axis, min, max }
    }
}

impl Rule {
    /// `condition_sets` is a slice of slices of (axis, (min, max))
    #[doc(hidden)]
    pub fn for_test(condition_sets: &[&[(&str, (f64, f64))]], subs: &[(&str, &str)]) -> Rule {
        Rule {
            conditions: condition_sets
                .iter()
                .map(|cond_set| {
                    cond_set
                        .iter()
                        .map(|(tag, (min, max))| Condition {
                            axis: std::str::FromStr::from_str(tag).unwrap(),
                            min: Some(DesignCoord::new(*min)),
                            max: Some(DesignCoord::new(*max)),
                        })
                        .collect()
                })
                .collect(),
            substitutions: subs
                .iter()
                .map(|(a, b)| Substitution {
                    replace: GlyphName::new(a),
                    with: GlyphName::new(b),
                })
                .collect(),
        }
    }
}

impl FromIterator<Condition> for ConditionSet {
    fn from_iter<T: IntoIterator<Item = Condition>>(iter: T) -> Self {
        let mut inner: Vec<_> = iter.into_iter().collect();
        inner.sort();
        Self(inner)
    }
}

impl<'a> IntoIterator for &'a ConditionSet {
    type Item = &'a Condition;

    type IntoIter = std::slice::Iter<'a, Condition>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_slice().iter()
    }
}

impl std::ops::Deref for ConditionSet {
    type Target = [Condition];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The named instances that reach fvar.
///
/// varLib splits a designspace into variable fonts before it builds one, and an
/// instance whose user location falls outside an axis' range belongs to none of them,
/// so it is simply absent from the font. This happens for real: a Glyphs source whose
/// instances disagree about where they sit on an axis leaves some of them off the map
/// the surviving ones defined.
///
/// <https://github.com/fonttools/fonttools/blob/4.63.0/Lib/fontTools/designspaceLib/split.py#L324-L326>
fn fvar_instances<'a>(
    axes: &'a Axes,
    named_instances: &'a [NamedInstance],
) -> impl Iterator<Item = &'a NamedInstance> {
    named_instances.iter().filter(|ni| {
        axes.iter().all(|axis| {
            ni.location
                .get(axis.tag)
                .is_none_or(|pos| axis.min <= pos && pos <= axis.max)
        })
    })
}

impl StaticMetadata {
    const DEFAULT_VENDOR_ID_TAG: Tag = Tag::new(b"NONE");
    // TODO: we could consider a builder or something for this?
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        units_per_em: u16,
        names: HashMap<NameKey, String>,
        axes: Vec<Axis>,
        mut named_instances: Vec<NamedInstance>,
        global_locations: HashSet<NormalizedLocation>,
        postscript_names: Option<PostscriptNames>,
        italic_angle: f64,
        glyphsapp_number_values: Option<
            HashMap<NormalizedLocation, BTreeMap<SmolStr, OrderedFloat<f64>>>,
        >,
        build_vertical: bool,
    ) -> Result<StaticMetadata, VariationModelError> {
        // A point axis (min == default == max) can't vary, but it is still an axis:
        // fontmake writes it to fvar, gives it an identity avar segment, a STAT
        // DesignAxis, a name record, and counts it in every VarStore's region axes.
        // Sources keep such an axis deliberately - a .glyphs "Axes" custom parameter
        // that names it, a mapping that bends it, or a designspace that declares it -
        // and glyphsLib preserves it for exactly those reasons:
        // <https://github.com/googlefonts/glyphsLib/blob/v6.13.1/Lib/glyphsLib/builder/axes.py#L288-L299>
        //
        // What a point axis cannot do is make a font variable. A source whose axes are
        // *all* point axes has nothing to interpolate, so it stays static and gets no
        // fvar at all; that is what an empty `axes` means to the rest of the compiler.
        // <https://github.com/googlefonts/fontc/issues/1990>
        let variable_axes: Axes = if axes.iter().any(|a| !a.is_point()) {
            Axes::new(axes.clone())
        } else {
            Axes::default()
        };

        // Named instances of static fonts are unhelpful <https://github.com/googlefonts/fontc/issues/1008>
        if !variable_axes.is_empty() {
            for instance in &mut named_instances {
                instance.location = instance.location.subset_axes(&variable_axes);
            }
        } else {
            named_instances.clear();
        };

        // Claim names for axes and named instances
        let mut name_id_gen = 255;
        // Spec-reserved names (<= 255) are not allowed in the set of unique reusable strings,
        // with the exception of the default instance's subfamily name which can reuse the
        // existing nameID 2 or 17:
        // https://github.com/googlefonts/fontc/issues/1502
        let mut reusable_names: HashMap<String, NameKey> = names
            .iter()
            .filter(|&(k, _)| k.name_id > 255.into())
            .map(|(k, v)| (v.clone(), *k))
            .collect();

        let default_instance_location: UserLocation =
            variable_axes.iter().map(|a| (a.tag, a.default)).collect();

        let mut register_if_new = |name: &str| {
            reusable_names.entry(name.to_owned()).or_insert_with(|| {
                name_id_gen += 1;
                NameKey::new(name_id_gen.into(), name)
            });
        };

        for axes in variable_axes.iter() {
            register_if_new(axes.ui_label_name());
        }

        for ni in fvar_instances(&variable_axes, &named_instances) {
            let instance_name = ni.name.as_str();
            if ni.location == default_instance_location
                && names
                    .iter()
                    .find_map(|(key, string)| (*string == instance_name).then_some(key.name_id))
                    .is_some_and(|name_id| {
                        name_id == NameId::SUBFAMILY_NAME
                            || name_id == NameId::TYPOGRAPHIC_SUBFAMILY_NAME
                    })
            {
                log::debug!(
                    "Reuse existing subfamily name '{instance_name}' for default instance at {default_instance_location:?}",
                );
            } else {
                register_if_new(instance_name);
            }

            if let Some(ps_name) = ni.postscript_name.as_deref() {
                register_if_new(ps_name);
            }
        }

        let mut names = names;
        names.extend(
            reusable_names
                .into_iter()
                .map(|(string, key)| (key, string)),
        );

        let variation_model = VariationModel::new(global_locations, variable_axes.axis_order());

        let default_location = axes
            .iter()
            .map(|a| (a.tag, NormalizedCoord::new(0.0)))
            .collect();

        Ok(StaticMetadata {
            units_per_em,
            names,
            all_source_axes: Axes::new(axes),
            axes: variable_axes,
            named_instances,
            variation_model,
            default_location,
            postscript_names,
            italic_angle: italic_angle.into(),
            number_values: glyphsapp_number_values.unwrap_or_default(),
            // the Glyphs source sets this after construction; every other
            // source leaves it None
            glyph_predicate_attrs: None,
            postscript: Default::default(),
            build_vertical,
            misc: MiscMetadata {
                fs_type: None, // default is, sigh, inconsistent across source formats
                is_fixed_pitch: None,
                selection_flags: Default::default(),
                vendor_id: Self::DEFAULT_VENDOR_ID_TAG,
                raw_vendor_id: None,
                // https://github.com/googlefonts/ufo2ft/blob/0d2688cd847d003b41104534d16973f72ef26c40/Lib/ufo2ft/fontInfoData.py#L353-L354
                version_major: 0,
                version_minor: 0,
                // <https://github.com/googlefonts/ufo2ft/blob/0d2688cd847d003b41104534d16973f72ef26c40/Lib/ufo2ft/fontInfoData.py#L364>
                lowest_rec_ppm: 6,
                // <https://github.com/googlefonts/ufo2ft/blob/0d2688cd847/Lib/ufo2ft/fontInfoData.py#L365>
                head_flags: head::Flags::LSB_AT_X_0 | head::Flags::BASELINE_AT_Y_0,
                created: None,
                family_class: None,
                panose: None,
                instance_panose: None,
                unicode_range_bits: None,
                codepage_range_bits: None,
                meta_table: None,
                us_weight_class: None,
                us_width_class: None,
                gasp: Vec::new(),
                feature_generation: None,
            },
            variations: None,
        })
    }

    /// The default on all variable axes.
    pub fn default_location(&self) -> &NormalizedLocation {
        &self.default_location
    }

    /// The PostScript settings of the master at `loc`.
    ///
    /// Falls back to the default master's settings when `loc` names no master
    /// (or names one the source gave no PostScript data), and to
    /// [`PostscriptSettings::default`] when the source has none at all.
    pub fn postscript_at(&self, loc: &NormalizedLocation) -> Cow<'_, PostscriptSettings> {
        self.postscript
            .get(loc)
            .or_else(|| self.postscript.get(&self.default_location))
            .map(Cow::Borrowed)
            .unwrap_or_default()
    }

    /// The PostScript settings of the default master.
    ///
    /// This is what a single-master table like CFF wants; a CFF2 writer would
    /// walk [`Self::postscript`] in [`Self::variation_model`] order instead.
    pub fn postscript_default(&self) -> Cow<'_, PostscriptSettings> {
        self.postscript_at(self.default_location())
    }

    pub fn axis(&self, tag: &Tag) -> Option<&Axis> {
        self.axes.iter().find(|a| &a.tag == tag)
    }

    /// The named instances fvar lists, a subset of [`Self::named_instances`].
    ///
    /// See [`fvar_instances`]: an instance outside the axes' ranges is not part of the
    /// variable font. It stays in `named_instances` for callers that want every
    /// instance the source declared, such as building one as a static.
    pub fn fvar_instances(&self) -> impl Iterator<Item = &NamedInstance> {
        fvar_instances(&self.axes, &self.named_instances)
    }

    /// Calculate a mapping of existing name text to the sorted set of name ID(s) that provide it.
    pub fn reverse_names(&self) -> HashMap<&str, BTreeSet<NameId>> {
        // https://github.com/fonttools/fonttools/blob/d5aec1b9/Lib/fontTools/ttLib/tables/_n_a_m_e.py#L326-L329
        self.names
            .iter()
            .fold(HashMap::new(), |mut accum, (key, name)| {
                accum.entry(name).or_default().insert(key.name_id);
                accum
            })
    }
}

impl From<[u8; 10]> for Panose {
    fn from(value: [u8; 10]) -> Self {
        Self {
            family_type: value[0],
            serif_style: value[1],
            weight: value[2],
            proportion: value[3],
            contrast: value[4],
            stroke_variation: value[5],
            arm_style: value[6],
            letterform: value[7],
            midline: value[8],
            x_height: value[9],
        }
    }
}

impl Panose {
    pub fn to_bytes(&self) -> [u8; 10] {
        [
            self.family_type,
            self.serif_style,
            self.weight,
            self.proportion,
            self.contrast,
            self.stroke_variation,
            self.arm_style,
            self.letterform,
            self.midline,
            self.x_height,
        ]
    }
}

impl Persistable for StaticMetadata {
    fn read(from: &mut dyn Read) -> Self {
        serde_yaml::from_reader(from).unwrap()
    }

    fn write(&self, to: &mut dyn std::io::Write) {
        serde_yaml::to_writer(to, self).unwrap();
    }
}

impl Persistable for GdefCategories {
    fn read(from: &mut dyn Read) -> Self {
        serde_yaml::from_reader(from).unwrap()
    }

    fn write(&self, to: &mut dyn std::io::Write) {
        serde_yaml::to_writer(to, self).unwrap();
    }
}

impl Persistable for PreliminaryGdefCategories {
    fn read(from: &mut dyn Read) -> Self {
        serde_yaml::from_reader(from).unwrap()
    }

    fn write(&self, to: &mut dyn std::io::Write) {
        serde_yaml::to_writer(to, self).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use fontdrasil::coords::UserCoord;

    use crate::ir::{FeatureWriterMode, KnownFeatureWriter};

    use super::*;

    fn test_static_metadata() -> StaticMetadata {
        let axis = Axis::for_test("wght");
        let mut point_axis = axis.clone();
        point_axis.min = point_axis.default;
        point_axis.max = point_axis.default;

        StaticMetadata {
            units_per_em: 1000,
            all_source_axes: vec![axis.clone(), point_axis].into(),
            axes: Axes::new(vec![axis.clone()]),
            named_instances: vec![NamedInstance {
                name: "Nobody".to_string(),
                postscript_name: None,
                location: vec![(WGHT, UserCoord::new(100.0))].into(),
                ..Default::default()
            }],
            variation_model: VariationModel::new(
                HashSet::from([
                    vec![(WGHT, NormalizedCoord::new(-1.0))].into(),
                    vec![(WGHT, NormalizedCoord::new(0.0))].into(),
                    vec![(WGHT, NormalizedCoord::new(1.0))].into(),
                ]),
                vec![axis.tag],
            ),
            default_location: vec![(WGHT, NormalizedCoord::new(0.0))].into(),
            names: HashMap::from([
                (
                    NameKey::new_bmp_only(NameId::FAMILY_NAME),
                    "Fam".to_string(),
                ),
                (
                    NameKey::new_bmp_only(NameId::TYPOGRAPHIC_FAMILY_NAME),
                    "Fam".to_string(),
                ),
                (
                    NameKey::new_bmp_only(NameId::new(256)),
                    "Weight".to_string(),
                ),
                (
                    NameKey::new_bmp_only(NameId::new(257)),
                    "Nobody".to_string(),
                ),
            ]),
            postscript_names: Some(HashMap::from([("lhs".into(), "rhs".into())])),
            italic_angle: 0.0.into(),
            misc: MiscMetadata {
                fs_type: None,
                is_fixed_pitch: None,
                selection_flags: SelectionFlags::default(),
                vendor_id: Tag::from_be_bytes(*b"DUCK"),
                raw_vendor_id: None,
                version_major: 42,
                version_minor: 24,
                head_flags: head::Flags::empty(),
                lowest_rec_ppm: 42,
                created: None,
                family_class: None,
                panose: None,
                instance_panose: None,
                unicode_range_bits: None,
                codepage_range_bits: None,
                meta_table: None,
                us_weight_class: None,
                us_width_class: None,
                gasp: Vec::new(),
                feature_generation: Some(vec![FeatureWriterSpec {
                    writer: KnownFeatureWriter::Kern,
                    mode: FeatureWriterMode::Append,
                    features: None,
                }]),
            },
            number_values: Default::default(),
            // a Glyphs source always sets this, and the round-trip tests
            // should carry an entry through
            glyph_predicate_attrs: Some(BTreeMap::from([(
                GlyphName::new("a"),
                GlyphPredicateAttrs {
                    category: Some("Letter".into()),
                    case: Some("lower".into()),
                    unicode: Some("0061".into()),
                    ..Default::default()
                },
            )])),
            // one entry per master, so the round-trip tests exercise a
            // multi-entry map
            postscript: HashMap::from([
                (
                    vec![(WGHT, NormalizedCoord::new(0.0))].into(),
                    PostscriptSettings {
                        blue_values: vec![(-10.0).into(), 0.0.into(), 700.0.into(), 710.0.into()],
                        blue_scale: Some(0.05.into()),
                        weight_name: Some("Chonky".to_string()),
                        ..Default::default()
                    },
                ),
                (
                    vec![(WGHT, NormalizedCoord::new(1.0))].into(),
                    PostscriptSettings {
                        blue_values: vec![(-12.0).into(), 0.0.into(), 720.0.into(), 734.0.into()],
                        stem_snap_v: vec![120.0.into()],
                        weight_name: Some("Chonkier".to_string()),
                        ..Default::default()
                    },
                ),
            ]),
            variations: None,
            build_vertical: false,
        }
    }

    const WGHT: Tag = Tag::from_be_bytes(*b"wght");

    fn assert_yml_round_trip<T>(thing: T)
    where
        for<'a> T: Serialize + Deserialize<'a> + PartialEq + Debug,
    {
        let yml = serde_yaml::to_string(&thing).unwrap();
        assert_eq!(thing, serde_yaml::from_str(&yml).unwrap());
    }

    fn assert_bincode_round_trip<T>(thing: T)
    where
        for<'a> T: Serialize + Deserialize<'a> + PartialEq + Debug,
    {
        let bin = bincode::serialize(&thing).unwrap();
        assert_eq!(thing, bincode::deserialize(&bin).unwrap());
    }

    #[test]
    fn axis_yaml() {
        assert_yml_round_trip(Axis::for_test("wght"));
    }

    #[test]
    fn axis_bincode() {
        assert_bincode_round_trip(Axis::for_test("wght"));
    }

    #[test]
    fn static_metadata_yaml() {
        assert_yml_round_trip(test_static_metadata());
    }

    #[test]
    fn static_metadata_bincode() {
        assert_bincode_round_trip(test_static_metadata());
    }

    #[test]
    fn static_metadata_smallest_id() {
        let static_metadata = test_static_metadata();
        let reverse_names = static_metadata.reverse_names();
        // in a sorted BTreeSet, the first is always the smallest
        assert_eq!(
            reverse_names.get("Fam").unwrap().iter().next().unwrap(),
            &NameId::FAMILY_NAME
        );
    }

    #[test]
    fn postscript_at_master_locations() {
        let static_metadata = test_static_metadata();
        let default = vec![(WGHT, NormalizedCoord::new(0.0))].into();
        let bold = vec![(WGHT, NormalizedCoord::new(1.0))].into();

        assert_eq!(
            static_metadata
                .postscript_at(&default)
                .weight_name
                .as_deref(),
            Some("Chonky")
        );
        assert_eq!(
            static_metadata.postscript_at(&bold).weight_name.as_deref(),
            Some("Chonkier")
        );
        // the default master is what a single-master table gets
        assert_eq!(
            static_metadata.postscript_default().weight_name.as_deref(),
            Some("Chonky")
        );
    }

    #[test]
    fn postscript_at_falls_back() {
        let mut static_metadata = test_static_metadata();
        let unknown: NormalizedLocation = vec![(WGHT, NormalizedCoord::new(-1.0))].into();

        // a location with no entry of its own gets the default master's
        assert_eq!(
            static_metadata
                .postscript_at(&unknown)
                .weight_name
                .as_deref(),
            Some("Chonky")
        );

        // a source with no PostScript data at all (e.g. fontra) gets defaults
        static_metadata.postscript.clear();
        assert_eq!(
            static_metadata.postscript_default().into_owned(),
            PostscriptSettings::default()
        );
    }

    #[test]
    fn condition_set_sorted() {
        let one = Condition::new(Tag::new(b"test"), None, None);
        let two = Condition::new(Tag::new(b"blah"), None, None);
        let tre = Condition::new(Tag::new(b"derp"), None, None);

        assert_eq!(
            [one, two, tre].into_iter().collect::<ConditionSet>(),
            [two, tre, one].into_iter().collect()
        );
    }
}
