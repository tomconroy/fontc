//! Evaluation of Glyphs.app glyph predicate tokens (`$[...]`).
//!
//! Glyphs.app sources can use a predicate token inside a FEA glyph class to
//! select glyphs by an `NSPredicate`-style expression, e.g.
//! `@ss01 = [ $[name endswith 'ss01'] ];`. At compile time we expand the token
//! into the matching glyphs, in glyph order.
//!
//! What is supported (<https://github.com/googlefonts/fontc/issues/92>, and
//! its phase 2 <https://github.com/googlefonts/fontc/issues/2052>): the
//! attributes `name`, `category`, `subCategory`, `case` and `unicode`, compared
//! with the operators `beginswith`, `endswith`, `contains`, `like`, `==`/`=`,
//! `!=`/`<>`, and (for `name` only) `<`, `<=`, `>`, `>=`, against a quoted
//! string or a bare word, joined by a flat chain of either `and`/`&&` or
//! `or`/`||` (but not a mix of the two). Anything else -- other attributes, the
//! `matches` operator, `not`, parentheses, or a mix of `and` and `or` -- is
//! rejected with a diagnostic naming it. Nothing is quietly ignored: an
//! unsupported predicate never silently selects everything or nothing.
//!
//! The grammar builds a typed [`typed::GlyphsAppPredicate`] AST; validation
//! (`compile::validate`) enforces that subset with diagnostics attached to the
//! offending child, and [`evaluate_predicate`] runs the already-validated tree
//! directly. Like the rest of the compiler, evaluation trusts validation: an
//! out-of-scope predicate that reaches it is a bug, and panics.
//!
//! The reference implementation is glyphsLib's `TokenExpander`
//! (`Lib/glyphsLib/builder/tokens.py`); we mirror its semantics: operator
//! keywords are case-insensitive, attribute names and value comparisons are
//! case-sensitive, both single and double quotes are accepted, and matches are
//! emitted in glyph order, de-duplicated. For `or`, glyphsLib accumulates
//! clause-by-clause (all of clause 1's matches in glyph order, then clause 2's
//! new matches, ...) rather than a single glyph-order pass; we reproduce that
//! exactly because the resulting class member order is observable in the
//! compiled tables.
//!
//! Every attribute but `name` is answered by the source (see
//! [`crate::compile::VariationInfo::glyph_predicate_attr`]) with the string the
//! source *stored*, and is missing (`None`) when it stored nothing. glyphsLib
//! compares a missing value as Python `None`: never equal to any string, but
//! stringified to the literal `"None"` by the substring and pattern operators.
//! That is a sharp edge of the reference implementation, and we reproduce it.
//!
//! A bare (unquoted) value is a string here, as it is in glyphsLib's final
//! `\w+` branch -- but glyphsLib types a bare word that *starts with* `yes`,
//! `true`, `no` or `false` as a boolean, and one that starts with a digit as an
//! integer, either of which then compares unequal to every string (or raises).
//! Rather than reproduce that, validation rejects those spellings and tells the
//! author to quote them.

use std::collections::HashSet;
use std::hash::Hash;

use smol_str::SmolStr;

use crate::compile::GlyphPredicateAttr;
use crate::typed;

/// The boolean connective joining the clauses of a predicate, reduced to the
/// single evaluation strategy it implies.
///
/// Only a flat chain of a single connective is supported; mixing `and` and
/// `or` is rejected during validation. A single clause evaluates as `And`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Connective {
    And,
    Or,
}

/// Evaluate a validated predicate against a glyph set.
///
/// `glyphs` yields `(id, name)` pairs and MUST be in glyph (GID) order; the
/// returned ids preserve that order. Results are de-duplicated.
///
/// `attr_value` answers one glyph's value for an attribute other than `name`,
/// returning `None` when the source did not set it. Validation has already
/// rejected every non-`name` attribute when the source cannot answer at all, so
/// a `None` here always means "the source left it unset", never "nobody asked
/// the source".
///
/// Validation (`compile::validate`) has already rejected anything outside the
/// supported subset, so evaluation trusts its input like the rest of the
/// compiler; a predicate that violates that invariant is a bug and panics.
///
/// glyphsLib emits predicate matches in *source* glyph order. We only have
/// the GID-ordered glyph map, which equals source order in the common case
/// but not for a source with a custom `glyphOrder` parameter that reorders
/// glyphs relative to the source; in that case a class whose member order is
/// observable (e.g. a parallel class-to-class substitution) could diverge.
/// Resolving this would require threading source order through to fea-rs.
pub(crate) fn evaluate_predicate<'a, T>(
    node: &typed::GlyphsAppPredicate,
    glyphs: impl IntoIterator<Item = (T, &'a str)>,
    mut attr_value: impl FnMut(&str, GlyphPredicateAttr) -> Option<SmolStr>,
) -> Vec<T>
where
    T: Copy + Eq + Hash,
{
    // Hoist each clause's attribute, operator and value out of the per-glyph
    // loops; `value.text()` allocates, so compute it exactly once per clause.
    // `None` for the attribute is `name`, which we answer ourselves.
    let clauses: Vec<(
        Option<GlyphPredicateAttr>,
        typed::GlyphsAppPredicateOp,
        SmolStr,
    )> = node
        .clauses()
        .map(|clause| {
            let attr = clause.attr();
            // glyphsLib's object regex is case-sensitive: `name` is valid,
            // `NAME` is not.
            let attr = (attr.text() != "name").then(|| {
                GlyphPredicateAttr::from_token(attr.text())
                    .expect("unsupported attributes are rejected by validation")
            });
            (attr, clause.op(), clause.value().text().into())
        })
        .collect();
    assert!(!clauses.is_empty(), "empty predicates are a parse error");

    let connective = node
        .connectives()
        .map(|conn| match conn {
            typed::GlyphsAppPredicateConnective::And(_) => Connective::And,
            typed::GlyphsAppPredicateConnective::Or(_) => Connective::Or,
        })
        .reduce(|prev, this| {
            assert_eq!(prev, this, "mixed connectives are rejected by validation");
            this
        })
        .unwrap_or(Connective::And);

    let glyphs: Vec<(T, &str)> = glyphs.into_iter().collect();
    // one clause tested against one glyph
    let mut matches = |(attr, op, value): &(
        Option<GlyphPredicateAttr>,
        typed::GlyphsAppPredicateOp,
        SmolStr,
    ),
                       name: &str| {
        let got = match attr {
            None => Some(SmolStr::new(name)),
            Some(attr) => attr_value(name, *attr),
        };
        op_matches(op, got.as_deref(), value)
    };

    match connective {
        // glyphsLib appends each clause's matches in turn, so a glyph that
        // matches an earlier clause keeps its earlier position. A single
        // glyph-order pass would re-interleave them; this does not.
        Connective::Or => {
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for clause in &clauses {
                for (id, name) in &glyphs {
                    if matches(clause, name) && seen.insert(*id) {
                        out.push(*id);
                    }
                }
            }
            out
        }
        // A single clause, or an `and` chain: a glyph is included iff every
        // clause matches. Iterating once in glyph order matches glyphsLib's
        // ordering for `and` (which preserves first-clause order) and
        // naturally de-duplicates.
        Connective::And => glyphs
            .iter()
            .filter(|(_, name)| clauses.iter().all(|clause| matches(clause, name)))
            .map(|(id, _)| *id)
            .collect(),
    }
}

/// Whether one glyph satisfies `<attribute> <op> <value>`.
///
/// `got` is the glyph's value for the clause's attribute, `None` when the
/// source did not set it. glyphsLib holds that as Python `None` and compares it
/// without special-casing, which is what the two arms below reproduce: the
/// operators that reach for `str(got)` see the literal `"None"`, and `==`/`!=`
/// see a value that is equal to no string at all
/// (`tokens.py`, `apply_comparators`).
fn op_matches(op: &typed::GlyphsAppPredicateOp, got: Option<&str>, value: &str) -> bool {
    use typed::GlyphsAppPredicateOp as Op;
    // `str(None)` in the operators that stringify
    let text = got.unwrap_or("None");
    match op {
        Op::BeginsWith(_) => text.starts_with(value),
        Op::EndsWith(_) => text.ends_with(value),
        Op::Contains(_) => text.contains(value),
        // glyphsLib's `like` is `fnmatch.fnmatchcase`, i.e. a case-sensitive
        // shell-style pattern anchored at both ends (`tokens.py`, `_like`).
        Op::Like(_) => fnmatch(text, value),
        Op::Eq(_) => got == Some(value),
        Op::Ne(_) => got != Some(value),
        // glyphsLib compares with Python's `<`/`<=`/`>`/`>=`, i.e.
        // lexicographic string ordering. Rust's `str` ordering is UTF-8 byte
        // order, which equals Unicode code point order, matching Python for
        // every valid string. Python raises a TypeError comparing `None` this
        // way, so validation only allows these on `name`, which every glyph
        // has.
        Op::Lt(_) | Op::Le(_) | Op::Gt(_) | Op::Ge(_) => {
            let got = got.expect("ordering on an optional attribute is rejected by validation");
            match op {
                Op::Lt(_) => got < value,
                Op::Le(_) => got <= value,
                Op::Gt(_) => got > value,
                _ => got >= value,
            }
        }
        Op::Matches(_) => unreachable!("'matches' is rejected by validation"),
    }
}

/// Whether `text` matches the shell-style pattern `pat`, as Python's
/// `fnmatch.fnmatchcase` does.
///
/// `*` matches any run of characters, `?` exactly one, everything else itself,
/// and the whole of `text` must match. Character classes (`[abc]`) are rejected
/// by validation, so a `[` here is a literal one -- which is also what Python
/// does with a `[` that opens no class.
///
/// Matching is by `char`, not byte: Python's `?` matches one code point.
fn fnmatch(text: &str, pat: &str) -> bool {
    let text: Vec<char> = text.chars().collect();
    let pat: Vec<char> = pat.chars().collect();
    // the classic linear backtracking match: on a mismatch, resume from one
    // character further along the most recent `*`
    let (mut t, mut p) = (0, 0);
    let (mut star, mut star_t) = (None, 0);
    while t < text.len() {
        if p < pat.len() && (pat[p] == '?' || pat[p] == text[t]) {
            t += 1;
            p += 1;
        } else if p < pat.len() && pat[p] == '*' {
            star = Some(p);
            star_t = t;
            p += 1;
        } else if let Some(star) = star {
            p = star + 1;
            star_t += 1;
            t = star_t;
        } else {
            return false;
        }
    }
    pat[p..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_tree::typed::AstNode;

    /// Parse `$[inner]` through the real lexer + grammar, so these tests
    /// exercise the same path the compiler uses.
    fn parse_predicate(inner: &str) -> typed::GlyphsAppPredicate {
        let src = format!("$[{inner}]");
        let (node, diags, err_str) = crate::parse::grammar::debug_parse_output(&src, |parser| {
            crate::parse::grammar::eat_glyphs_predicate(parser, crate::TokenSet::EMPTY);
        });
        assert!(
            !diags.iter().any(|diag| diag.is_error()),
            "`{inner}` produced parse errors that would stop real compilation, so \
             this evaluator test would be exercising a recovered parse:\n{err_str}"
        );
        typed::GlyphsAppPredicate::cast(&node)
            .unwrap_or_else(|| panic!("`{inner}` did not parse as a predicate"))
    }

    // evaluate against a list of (id, name) pairs given in glyph order.
    fn eval(inner: &str, glyphs: &[(u16, &str)]) -> Vec<u16> {
        eval_with_attrs(inner, glyphs, &[])
    }

    /// evaluate with a source that answers attributes from `attrs`, given as
    /// (glyph name, attribute, value) triples; anything absent is unset.
    fn eval_with_attrs(
        inner: &str,
        glyphs: &[(u16, &str)],
        attrs: &[(&str, GlyphPredicateAttr, &str)],
    ) -> Vec<u16> {
        evaluate_predicate(
            &parse_predicate(inner),
            glyphs.iter().map(|(id, name)| (*id, *name)),
            |glyph, attr| {
                attrs
                    .iter()
                    .find(|(g, a, _)| *g == glyph && *a == attr)
                    .map(|(_, _, value)| SmolStr::new(value))
            },
        )
    }

    fn names<'a>(inner: &str, glyphs: &[(u16, &'a str)]) -> Vec<&'a str> {
        let ids = eval(inner, glyphs);
        ids.iter()
            .map(|id| glyphs.iter().find(|(g, _)| g == id).unwrap().1)
            .collect()
    }

    fn sample() -> Vec<(u16, &'static str)> {
        // a small glyph order with arabic-ish suffixes plus some plain glyphs
        [
            "A",
            "A.sc",
            "B",
            "behDotless-ar.init",
            "behDotless-ar.init.fbeh2",
            "behDotless-ar.medi",
            "meem-ar.init",
            "meem-ar.medi",
            "ss01.a",
            "x.ss01",
        ]
        .iter()
        .enumerate()
        .map(|(i, n)| (i as u16, *n))
        .collect()
    }

    #[test]
    fn endswith_single_quote() {
        // DynaPuff form
        let glyphs = sample();
        assert_eq!(names("name endswith 'ss01'", &glyphs), vec!["x.ss01"]);
    }

    #[test]
    fn contains_double_quote() {
        let glyphs = sample();
        assert_eq!(
            names("name contains \"meem-ar\"", &glyphs),
            vec!["meem-ar.init", "meem-ar.medi"]
        );
    }

    #[test]
    fn beginswith() {
        let glyphs = sample();
        assert_eq!(
            names("name beginswith \"behDotless\"", &glyphs),
            vec![
                "behDotless-ar.init",
                "behDotless-ar.init.fbeh2",
                "behDotless-ar.medi"
            ]
        );
    }

    #[test]
    fn flat_and_with_not_equal() {
        // Noto Nastaliq Urdu form: contains X and name != Y and name != Z
        let glyphs = sample();
        assert_eq!(
            names(
                "name contains \"behDotless-ar.init\" and name != \"behDotless-ar.init.fbeh2\"",
                &glyphs
            ),
            vec!["behDotless-ar.init"]
        );
    }

    #[test]
    fn flat_or() {
        let glyphs = sample();
        assert_eq!(
            names(
                "name contains \"meem-ar.init\" or name contains \"meem-ar.medi\"",
                &glyphs
            ),
            vec!["meem-ar.init", "meem-ar.medi"]
        );
    }

    #[test]
    fn or_preserves_clause_order_not_glyph_order() {
        // A clause-2 match precedes a clause-1 match in glyph order. glyphsLib
        // emits clause-1 matches first, THEN clause-2's new matches -- NOT pure
        // glyph order. Glyph order here is [medi(0), init(1)].
        let glyphs = [(0u16, "x.medi"), (1u16, "x.init")];
        // clause 1 = init, clause 2 = medi -> expect [init, medi], not [medi, init]
        assert_eq!(
            eval(
                "name endswith \".init\" or name endswith \".medi\"",
                &glyphs
            ),
            vec![1, 0]
        );
    }

    #[test]
    fn or_dedups() {
        // a glyph matching both clauses appears once, at its first-clause position
        let glyphs = [(0u16, "ab"), (1u16, "ba")];
        assert_eq!(
            eval("name contains \"a\" or name contains \"b\"", &glyphs),
            vec![0, 1]
        );
    }

    #[test]
    fn empty_result_is_empty() {
        let glyphs = sample();
        assert!(eval("name endswith \"zzzz\"", &glyphs).is_empty());
    }

    #[test]
    fn operator_keywords_case_insensitive() {
        let glyphs = sample();
        assert_eq!(names("name ENDSWITH 'ss01'", &glyphs), vec!["x.ss01"]);
        assert_eq!(
            names(
                "name contains \"meem-ar.init\" OR name contains \"meem-ar.medi\"",
                &glyphs
            ),
            vec!["meem-ar.init", "meem-ar.medi"]
        );
    }

    #[test]
    fn value_case_sensitive() {
        let glyphs = [(0u16, "A.sc"), (1u16, "a.sc")];
        assert_eq!(eval("name beginswith \"A\"", &glyphs), vec![0]);
    }

    #[test]
    fn symbolic_aliases() {
        let glyphs = [(0u16, "a"), (1u16, "b")];
        assert_eq!(eval("name = \"a\"", &glyphs), vec![0]);
        assert_eq!(eval("name == \"a\"", &glyphs), vec![0]);
        assert_eq!(eval("name != \"a\"", &glyphs), vec![1]);
        assert_eq!(eval("name <> \"a\"", &glyphs), vec![1]);
    }

    #[test]
    fn relational_operators_are_lexicographic() {
        // glyphsLib compares the name string with Python's relational
        // operators, i.e. lexicographic ordering.
        let glyphs = [(0u16, "a"), (1u16, "m"), (2u16, "z")];
        assert_eq!(eval("name < \"m\"", &glyphs), vec![0]);
        assert_eq!(eval("name <= \"m\"", &glyphs), vec![0, 1]);
        assert_eq!(eval("name > \"m\"", &glyphs), vec![2]);
        assert_eq!(eval("name >= \"m\"", &glyphs), vec![1, 2]);
    }

    // -- attributes other than `name` --

    /// (glyph, category, case) triples flattened into the attribute table
    fn letters() -> Vec<(&'static str, GlyphPredicateAttr, &'static str)> {
        use GlyphPredicateAttr::*;
        vec![
            ("a", Category, "Letter"),
            ("a", Case, "lower"),
            ("A", Category, "Letter"),
            ("A", Case, "upper"),
            ("one", Category, "Number"),
            // `space` sets nothing at all
        ]
    }

    fn attr_sample() -> Vec<(u16, &'static str)> {
        vec![(0, "a"), (1, "A"), (2, "one"), (3, "space")]
    }

    #[test]
    fn borel_predicate() {
        // Borel's: every lowercase letter with a codepoint. `unicode != nil`
        // is glyphsLib's own no-op (see `nil_is_a_string_so_matches_all`).
        let mut attrs = letters();
        attrs.push(("a", GlyphPredicateAttr::Unicode, "0061"));
        attrs.push(("A", GlyphPredicateAttr::Unicode, "0041"));
        assert_eq!(
            eval_with_attrs(
                "category like \"Letter\" && case==lower && unicode != nil",
                &attr_sample(),
                &attrs,
            ),
            vec![0]
        );
    }

    #[test]
    fn unset_attribute_matches_nothing_but_is_not_an_error() {
        // `space` sets no category: glyphsLib compares Python None, which is
        // equal to no string
        assert_eq!(
            eval_with_attrs("category == \"Letter\"", &attr_sample(), &letters()),
            vec![0, 1]
        );
        // ... and unequal to every string, so `space` IS selected here
        assert_eq!(
            eval_with_attrs("category != \"Letter\"", &attr_sample(), &letters()),
            vec![2, 3]
        );
    }

    #[test]
    fn unset_attribute_stringifies_to_none() {
        // the sharp edge we inherit from glyphsLib: the operators that reach
        // for `str(got)` see the literal "None". Only `space` (which sets no
        // category) matches; `one` is a "Number".
        assert_eq!(
            eval_with_attrs("category beginswith \"No\"", &attr_sample(), &letters()),
            vec![3]
        );
        assert_eq!(
            eval_with_attrs("category like \"N*e\"", &attr_sample(), &letters()),
            vec![3]
        );
    }

    #[test]
    fn nil_is_a_string_so_matches_all() {
        // glyphsLib types a bare `nil` as the string "nil" (its `\w+` branch),
        // so `unicode != nil` is true even for a glyph with no codepoint
        let attrs = vec![("a", GlyphPredicateAttr::Unicode, "0061")];
        assert_eq!(
            eval_with_attrs("unicode != nil", &attr_sample(), &attrs),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn bare_value_is_a_string() {
        assert_eq!(
            eval_with_attrs("case == lower", &attr_sample(), &letters()),
            vec![0]
        );
        assert_eq!(
            eval_with_attrs("case == \"lower\"", &attr_sample(), &letters()),
            vec![0]
        );
    }

    // -- the `like` operator --

    #[test]
    fn like_without_wildcards_is_equality() {
        let glyphs = [(0u16, "a.init"), (1u16, "a")];
        assert_eq!(eval("name like \"a\"", &glyphs), vec![1]);
    }

    #[test]
    fn like_wildcards() {
        let glyphs = [
            (0u16, "a.init"),
            (1u16, "a.medi"),
            (2u16, "ab.init"),
            (3u16, "a"),
        ];
        assert_eq!(eval("name like \"*.init\"", &glyphs), vec![0, 2]);
        assert_eq!(eval("name like \"a.*\"", &glyphs), vec![0, 1]);
        assert_eq!(eval("name like \"a?init\"", &glyphs), vec![0]);
        assert_eq!(eval("name like \"*\"", &glyphs), vec![0, 1, 2, 3]);
        assert_eq!(eval("name like \"*.*i*\"", &glyphs), vec![0, 1, 2]);
    }

    #[test]
    fn fnmatch_matches_python() {
        // spot checks against `fnmatch.fnmatchcase`
        for (text, pat, expected) in [
            ("", "", true),
            ("", "*", true),
            ("", "?", false),
            ("abc", "abc", true),
            ("abc", "ABC", false),
            ("abc", "a*", true),
            ("abc", "*c", true),
            ("abc", "*b*", true),
            ("abc", "a?c", true),
            ("abc", "a?", false),
            ("abc", "*d*", false),
            ("abc", "**a**b**c**", true),
            ("a.b.c", "a.*.c", true),
            // a multi-byte char is one `?`
            ("é", "?", true),
            // an unclosed class is a literal, as in Python
            ("[a", "[a", true),
        ] {
            assert_eq!(fnmatch(text, pat), expected, "fnmatch({text:?}, {pat:?})");
        }
    }

    #[test]
    fn quoted_boolean_word_is_a_plain_string() {
        // glyphsLib types a *bare* value starting with yes/true/no/false as a
        // boolean (validation rejects all unquoted values). Quoting bypasses
        // the boolean typing and selects the named glyph, as it does in
        // glyphsLib.
        let glyphs = [(0u16, "noon"), (1u16, "a")];
        assert_eq!(eval("name == \"noon\"", &glyphs), vec![0]);
    }
}
