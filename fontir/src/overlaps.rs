//! Removing overlaps from a static glyph's outline.
//!
//! fontmake removes overlaps from every glyph of a static font by default
//! (ufo2ft's `RemoveOverlapsFilter`; `--keep-overlaps` opts out), because a
//! renderer that fills with the even-odd rule draws the overlap of two contours
//! as a hole. A variable font keeps its overlaps: a union per master would
//! leave the masters incompatible.
//!
//! The boolean operation itself is [`fontdrasil::overlaps`]; this module is the
//! font-specific policy around it, which follows ufo2ft's filter:
//!
//! - it is a union of the glyph's own *contours* under the non-zero rule.
//!   Components are left alone — ufo2ft's filter iterates contours, and a pure
//!   composite has none — so in a TrueType font a composite's components still
//!   overlap each other. Decomposition, which happens before this, is what
//!   decides which glyphs have contours to union;
//! - the result follows the PostScript convention, outer contours
//!   counter-clockwise and counters clockwise, whatever direction the source
//!   drew them in, as skia-pathops' and booleanOperations' unions do. The
//!   TrueType backend reverses them afterwards, like ufo2ft's cu2qu does;
//! - it runs on the source's curves, before any conversion to quadratics, the
//!   same order as ufo2ft's TrueType preprocessor (remove overlaps, then cu2qu).
//!
//! Beyond that, the result is made to disturb the outline as little as
//! possible, so that a glyph with nothing to remove comes out exactly as it
//! went in:
//!
//! - every point and segment the union merely reproduced is put back exactly
//!   as the source had it. The sweep line is free to move points by a hair
//!   (its accuracy parameter), and a hair is enough to send a coordinate
//!   sitting on .5 to the other side when it is rounded to font units;
//! - like skia-pathops' `keep_starting_points`, a contour starts where the
//!   source contour it came from started, if that point survived;
//! - contours keep the source's order rather than the sweep line's.

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use fontdrasil::overlaps::{self, FillRule, OverlapError};
use kurbo::{BezPath, CubicBez, Line, ParamCurve, PathEl, PathSeg, Point, QuadBez, Shape};

use crate::{error::BadGlyph, ir::Glyph};

/// How close a point of the union has to be to a point of the source to be
/// taken for it.
///
/// A few times the sweep line's accuracy parameter (1e-6 at font-unit
/// magnitudes), and far below anything that survives rounding to font units:
/// two distinct points of a real outline are never this close.
const SAME_POINT: f64 = 1e-5;

/// How close the off-curve points of a segment of the union have to be to a
/// source segment between the same two points to be taken for it.
///
/// Looser than [`SAME_POINT`]: a curve the union did not cut comes back as a
/// re-parameterised copy of itself, whose off-curves carry the error of a
/// numerically solved parameter. A curve with the same ends that is genuinely
/// different is many units away.
const SAME_CURVE: f64 = 1e-3;

/// The area, in units², below which a contour of the union is a sliver; see
/// [`is_a_sliver`].
const NEGLIGIBLE_AREA: f64 = 0.25;

/// Remove the overlaps of every glyph in `glyphs` that has contours.
///
/// Returns the glyphs that changed (every glyph with contours, at every
/// location), in the order given. ufo2ft's filter skips a glyph without
/// contours, so a composite keeps its overlapping components.
///
/// Each glyph is independent of every other, so they are spread over as many
/// threads as the platform offers; where it offers none (wasm), they run here.
pub fn remove_glyph_overlaps<G>(glyphs: &[G]) -> Result<Vec<Glyph>, BadGlyph>
where
    G: AsRef<Glyph> + Sync,
{
    let todo: Vec<&Glyph> = glyphs
        .iter()
        .map(AsRef::as_ref)
        .filter(|glyph| {
            glyph
                .sources()
                .values()
                .any(|inst| !inst.contours.is_empty())
        })
        .collect();
    map_in_parallel(&todo, |glyph| {
        let name = glyph.name.clone();
        let mut glyph = (*glyph).clone();
        for inst in glyph.sources_mut().values_mut() {
            inst.contours =
                remove_overlaps(&inst.contours).map_err(|e| BadGlyph::new(name.clone(), e))?;
        }
        Ok(glyph)
    })
    .into_iter()
    .collect()
}

/// `items.iter().map(f).collect()`, on all the threads there are.
///
/// Threads take the next item as they finish the last, rather than a fixed
/// share, because glyphs vary a lot in how long their union takes.
fn map_in_parallel<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, U)>> = Mutex::new(Vec::with_capacity(items.len()));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut mine = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else {
                        break;
                    };
                    mine.push((i, f(item)));
                }
                results.lock().unwrap().extend(mine);
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_unstable_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, result)| result).collect()
}

/// The union of `contours` under the non-zero rule, as contours.
///
/// Every contour is taken to be closed: a font has no open contours, a
/// renderer closes them, and ufo2ft's union backends close them too. A contour
/// without a single segment (a lone point) has no area and does not survive.
pub fn remove_overlaps(contours: &[BezPath]) -> Result<Vec<BezPath>, OverlapError> {
    let closed: Vec<BezPath> = contours.iter().map(closed).collect();
    let union = overlaps::remove_overlaps(&overlaps::combine_paths(&closed), FillRule::NonZero)?;
    let source = SourceGeometry::new(&closed);
    let union = union
        .iter()
        .map(|contour| source.restore(contour))
        .filter(|segments| !is_a_sliver(segments))
        .collect();
    Ok(restore_source_order(union, contours))
}

/// Is the contour drawing `segments` a sliver, too thin to have any area?
///
/// A union can leave a hair-thin loop where a source contour doubles back on
/// itself (a curve overshooting the line it ends on), or where two contours
/// touch to within the sweep line's accuracy. Such a contour fills nothing a
/// renderer could show, and rounded to font units it collapses into a contour
/// of one or two points — or, in CFF, a bare moveto. ufo2ft's booleanOperations
/// backend never emits one, since it works on an integer grid.
///
/// Across the twenty-odd sources this was checked on, every sliver had an area
/// under 0.001 units² and every genuine contour more than 4, so the threshold
/// has room either way.
fn is_a_sliver(segments: &[PathSeg]) -> bool {
    to_path(segments).is_none_or(|path| path.area().abs() < NEGLIGIBLE_AREA)
}

/// `contour`, closed if it was not.
fn closed(contour: &BezPath) -> BezPath {
    let mut closed = contour.clone();
    if !matches!(closed.elements().last(), Some(PathEl::ClosePath) | None) {
        closed.close_path();
    }
    closed
}

/// The source's points and segments, to recognise them in the union.
struct SourceGeometry {
    /// On-curve points, bucketed by a grid much coarser than [`SAME_POINT`].
    points: HashMap<(i64, i64), Vec<Point>>,
    /// Segments, by their exact end points.
    segments: HashMap<(PointKey, PointKey), Vec<PathSeg>>,
}

/// A point's exact bits, to look it up by.
type PointKey = (u64, u64);

fn point_key(p: Point) -> PointKey {
    // + 0.0 turns -0.0 into 0.0, which compare equal but have other bits
    ((p.x + 0.0).to_bits(), (p.y + 0.0).to_bits())
}

/// The side of a [`SourceGeometry::points`] grid cell.
const GRID: f64 = 1.0 / 64.0;

fn grid_cell(p: Point) -> (i64, i64) {
    ((p.x / GRID).floor() as i64, (p.y / GRID).floor() as i64)
}

impl SourceGeometry {
    fn new(contours: &[BezPath]) -> Self {
        let mut points: HashMap<(i64, i64), Vec<Point>> = HashMap::new();
        let mut segments: HashMap<(PointKey, PointKey), Vec<PathSeg>> = HashMap::new();
        for seg in contours.iter().flat_map(|contour| contour.segments()) {
            for p in [seg.start(), seg.end()] {
                let cell = points.entry(grid_cell(p)).or_default();
                if !cell.contains(&p) {
                    cell.push(p);
                }
            }
            segments
                .entry((point_key(seg.start()), point_key(seg.end())))
                .or_default()
                .push(seg);
        }
        SourceGeometry { points, segments }
    }

    /// The source point `p` is a copy of, or `p` if it is a new one.
    fn point(&self, p: Point) -> Point {
        let (cx, cy) = grid_cell(p);
        let mut best: Option<(f64, Point)> = None;
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for q in self.points.get(&(x, y)).into_iter().flatten() {
                    let d = q.distance(p);
                    if d <= SAME_POINT && best.is_none_or(|(best_d, _)| d < best_d) {
                        best = Some((d, *q));
                    }
                }
            }
        }
        best.map_or(p, |(_, q)| q)
    }

    /// The source segment `seg` is a copy of (either way round), or `seg` with
    /// its ends put back on the source points they are copies of.
    fn segment(&self, seg: PathSeg) -> PathSeg {
        let seg = with_ends(seg, self.point(seg.start()), self.point(seg.end()));
        let (start, end) = (point_key(seg.start()), point_key(seg.end()));
        let forwards = self.segments.get(&(start, end)).into_iter().flatten();
        let backwards = self
            .segments
            .get(&(end, start))
            .into_iter()
            .flatten()
            .map(|s| s.reverse());
        forwards
            .copied()
            .chain(backwards)
            .find(|candidate| same_curve(*candidate, seg))
            .unwrap_or(seg)
    }

    /// One contour of the union, as segments restored to the source's,
    /// leaving out any segment the restoration collapsed to a point.
    fn restore(&self, contour: &BezPath) -> Vec<PathSeg> {
        contour
            .segments()
            .map(|seg| self.segment(seg))
            .filter(|seg| !is_a_point(*seg))
            .collect()
    }
}

/// `seg` moved to start at `start` and end at `end`, off-curves untouched.
fn with_ends(seg: PathSeg, start: Point, end: Point) -> PathSeg {
    match seg {
        PathSeg::Line(_) => PathSeg::Line(Line::new(start, end)),
        PathSeg::Quad(q) => PathSeg::Quad(QuadBez::new(start, q.p1, end)),
        PathSeg::Cubic(c) => PathSeg::Cubic(CubicBez::new(start, c.p1, c.p2, end)),
    }
}

/// Are `a` and `b` the same kind of segment, with the same ends and
/// off-curves within [`SAME_CURVE`] of each other?
fn same_curve(a: PathSeg, b: PathSeg) -> bool {
    let near = |p: Point, q: Point| p.distance(q) <= SAME_CURVE;
    a.start() == b.start()
        && a.end() == b.end()
        && match (a, b) {
            (PathSeg::Line(_), PathSeg::Line(_)) => true,
            (PathSeg::Quad(a), PathSeg::Quad(b)) => near(a.p1, b.p1),
            (PathSeg::Cubic(a), PathSeg::Cubic(b)) => near(a.p1, b.p1) && near(a.p2, b.p2),
            _ => false,
        }
}

/// Does `seg` draw nothing but a point?
fn is_a_point(seg: PathSeg) -> bool {
    let p = seg.start();
    match seg {
        PathSeg::Line(l) => l.p1 == p,
        PathSeg::Quad(q) => q.p1 == p && q.p2 == p,
        PathSeg::Cubic(c) => c.p1 == p && c.p2 == p && c.p3 == p,
    }
}

/// Start each contour of the union where its source contour started, and put
/// them back in source order.
///
/// A source contour claims the first contour of the union, in the union's
/// order, that passes through its start point; the claimed contours then come
/// out in the order of the source contours that claimed them, followed by any
/// contour nobody claimed (one whose source start point was swallowed by the
/// union), in the union's order.
fn restore_source_order(mut segments: Vec<Vec<PathSeg>>, source: &[BezPath]) -> Vec<BezPath> {
    let mut claimed_by: Vec<Option<usize>> = vec![None; segments.len()];

    for (source_idx, start) in
        source
            .iter()
            .enumerate()
            .filter_map(|(i, c)| match c.elements().first() {
                Some(PathEl::MoveTo(p)) => Some((i, *p)),
                _ => None,
            })
    {
        let found = segments.iter().enumerate().find_map(|(i, segs)| {
            if claimed_by[i].is_some() {
                return None;
            }
            segs.iter()
                .position(|seg| seg.start().distance(start) <= SAME_POINT)
                .map(|at| (i, at))
        });
        if let Some((i, at)) = found {
            segments[i].rotate_left(at);
            claimed_by[i] = Some(source_idx);
        }
    }

    let mut ordered: Vec<(usize, Vec<PathSeg>)> = segments
        .into_iter()
        .zip(claimed_by)
        .enumerate()
        .map(|(union_idx, (segs, claimed))| (claimed.unwrap_or(source.len() + union_idx), segs))
        .collect();
    ordered.sort_by_key(|(order, _)| *order);
    ordered
        .into_iter()
        .filter_map(|(_, segs)| to_path(&segs))
        .collect()
}

/// A closed contour drawing `segments`, with an explicit closing segment.
///
/// The explicit closing line is how fontc's frontends write a closed contour
/// (fontTools' `outputImpliedClosingLine=True`), so the union reads like any
/// other contour to the backends.
fn to_path(segments: &[PathSeg]) -> Option<BezPath> {
    let first = segments.first()?;
    let mut path = BezPath::new();
    path.move_to(first.start());
    for seg in segments {
        path.push(seg.as_path_el());
    }
    if path.elements().last().and_then(|el| el.end_point()) != Some(first.start()) {
        path.line_to(first.start());
    }
    path.close_path();
    Some(path)
}

#[cfg(test)]
mod tests {
    use kurbo::Circle;

    use super::*;

    /// Is the shoelace area of `contour` positive, i.e. is it counter-clockwise
    /// in a y-up coordinate system?
    fn is_counter_clockwise(contour: &BezPath) -> bool {
        contour.area() > 0.0
    }

    /// The first point of `contour`.
    fn start_point(contour: &BezPath) -> Option<Point> {
        match contour.elements().first() {
            Some(PathEl::MoveTo(p)) => Some(*p),
            _ => None,
        }
    }

    /// A rectangle drawn counter-clockwise (y-up) from its bottom-left corner,
    /// with the explicit closing line fontc's frontends write.
    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        let mut path = BezPath::new();
        path.move_to((x0, y0));
        path.line_to((x1, y0));
        path.line_to((x1, y1));
        path.line_to((x0, y1));
        path.line_to((x0, y0));
        path.close_path();
        path
    }

    fn area(contours: &[BezPath]) -> f64 {
        contours.iter().map(|c| c.area()).sum()
    }

    #[test]
    fn two_overlapping_squares_become_one_contour() {
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), rect(5., 5., 15., 15.)]).unwrap();
        assert_eq!(union.len(), 1, "{union:?}");
        assert_eq!(area(&union), 175.0);
        assert!(is_counter_clockwise(&union[0]));
    }

    #[test]
    fn the_union_starts_where_the_source_did() {
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), rect(5., 5., 15., 15.)]).unwrap();
        assert_eq!(start_point(&union[0]), Some(Point::new(0., 0.)));
        // explicitly closed, like every other contour in the IR
        let elements = union[0].elements();
        assert_eq!(elements.last(), Some(&PathEl::ClosePath));
        assert_eq!(
            elements[elements.len() - 2].end_point(),
            Some(Point::new(0., 0.))
        );
    }

    #[test]
    fn nothing_to_remove_is_a_no_op() {
        // an 'o': a counter-clockwise outer, a clockwise counter, and a
        // separate dot, out of sweep-line order
        let dot = rect(20., 20., 22., 22.);
        let outer = rect(0., 0., 10., 10.);
        let counter = rect(2., 2., 8., 8.).reverse_subpaths();
        let source = vec![dot, outer, counter];
        let union = remove_overlaps(&source).unwrap();
        assert_eq!(union, source);
    }

    #[test]
    fn curves_with_nothing_to_remove_come_back_unchanged() {
        let circle = Circle::new((0.0, 0.0), 100.0).to_path(0.01);
        // kurbo's circle ends its last curve on the start point, so it is
        // already in the IR's explicitly-closed form
        let start = circle.elements().first().and_then(|el| el.end_point());
        let n = circle.elements().len();
        assert_eq!(circle.elements()[n - 2].end_point(), start);
        let union = remove_overlaps(std::slice::from_ref(&circle)).unwrap();
        // exactly: the union's re-parameterised copies of the curves are
        // replaced by the source's own
        assert_eq!(union, vec![circle]);
    }

    #[test]
    fn a_point_on_a_rounding_tie_stays_put() {
        // a '<' whose tip, interpolated, landed a hair under .5: moving it by
        // the sweep line's accuracy would round it to the other font unit
        let tip = 61.5 - 1e-9;
        let mut less = BezPath::new();
        less.move_to((504., 394.));
        less.line_to((504., 496.));
        less.line_to((tip, 241.));
        less.line_to((504., -14.));
        less.line_to((504., 87.));
        less.line_to((241., 241.));
        less.line_to((504., 394.));
        less.close_path();
        // and something to overlap, so the union has work to do
        let bar = rect(400., 200., 600., 280.);
        let union = remove_overlaps(&[less, bar]).unwrap();
        let points: Vec<Point> = union
            .iter()
            .flat_map(|c| c.elements().iter().filter_map(|el| el.end_point()))
            .collect();
        assert!(points.contains(&Point::new(tip, 241.)), "{union:?}");
    }

    #[test]
    fn a_clockwise_outer_comes_out_counter_clockwise() {
        let union = remove_overlaps(&[rect(0., 0., 10., 10.).reverse_subpaths()]).unwrap();
        assert_eq!(union.len(), 1);
        assert!(is_counter_clockwise(&union[0]));
        assert_eq!(start_point(&union[0]), Some(Point::new(0., 0.)));
    }

    #[test]
    fn a_counter_drawn_like_the_outer_is_filled() {
        // non-zero: the "counter" has winding 2, so it is inside
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), rect(2., 2., 8., 8.)]).unwrap();
        assert_eq!(union, vec![rect(0., 0., 10., 10.)]);
    }

    #[test]
    fn a_duplicate_contour_is_dropped() {
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), rect(0., 0., 10., 10.)]).unwrap();
        assert_eq!(union, vec![rect(0., 0., 10., 10.)]);
    }

    #[test]
    fn an_open_contour_is_closed() {
        let mut open = BezPath::new();
        open.move_to((0., 0.));
        open.line_to((10., 0.));
        open.line_to((10., 10.));
        let union = remove_overlaps(&[open]).unwrap();
        assert_eq!(union.len(), 1, "{union:?}");
        assert_eq!(area(&union), 50.0);
    }

    #[test]
    fn a_lone_point_does_not_survive() {
        let mut lone = BezPath::new();
        lone.move_to((5., 5.));
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), lone]).unwrap();
        assert_eq!(union, vec![rect(0., 0., 10., 10.)]);
    }

    #[test]
    fn a_swallowed_start_point_comes_last() {
        // the second square's start, (5, 5), is inside the first, so the
        // union cannot start there; the first square still claims it
        let union = remove_overlaps(&[rect(0., 0., 10., 10.), rect(5., 5., 15., 15.)]).unwrap();
        assert_eq!(union.len(), 1);
        // and a disjoint third contour keeps its place after it
        let union = remove_overlaps(&[
            rect(5., 5., 15., 15.),
            rect(0., 0., 10., 10.),
            rect(30., 30., 40., 40.),
        ])
        .unwrap();
        assert_eq!(union.len(), 2);
        assert_eq!(start_point(&union[0]), Some(Point::new(0., 0.)));
        assert_eq!(union[1], rect(30., 30., 40., 40.));
    }

    #[test]
    fn overlapping_curves_lose_their_overlap() {
        let a = Circle::new((0.0, 0.0), 100.0).to_path(0.01);
        let b = Circle::new((80.0, 0.0), 100.0).to_path(0.01);
        let union = remove_overlaps(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(union.len(), 1);
        // two r=100 circles 80 apart: their lens counts once
        let r: f64 = 100.0;
        let d: f64 = 80.0;
        let lens = 2.0 * r * r * (d / (2.0 * r)).acos() - d / 2.0 * (4.0 * r * r - d * d).sqrt();
        let expected = 2.0 * std::f64::consts::PI * r * r - lens;
        let got = area(&union);
        assert!(
            (got - expected).abs() / expected < 1e-3,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn a_curve_doubling_back_leaves_no_sliver() {
        // from Josefin Sans' lira: a curve that overshoots the stem it joins
        // and comes back down onto it
        let mut path = BezPath::new();
        path.move_to((200., 93.));
        path.line_to((230., 93.));
        path.curve_to((252., 112.), (251., 188.), (251., 178.));
        path.line_to((251., 504.));
        path.line_to((200., 504.));
        path.line_to((200., 93.));
        path.close_path();
        let union = remove_overlaps(std::slice::from_ref(&path)).unwrap();
        for contour in &union {
            assert!(contour.area().abs() >= NEGLIGIBLE_AREA, "{union:?}");
        }
        let (got, want) = (area(&union), path.area());
        assert!((got - want).abs() < 1.0, "{got} vs {want}");
    }
}
