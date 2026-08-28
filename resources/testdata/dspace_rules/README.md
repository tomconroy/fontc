This directory contains designspace files that are used to test our parsing of
'rules' that describe feature variations.

`Basic`, `CustomFeatures` and `Last` share the `WghtVar-*` masters one level up.
`Chain` has masters of its own (`RuleChain-*.ufo`) because it needs three
glyphs, a composite, kerning and kern groups to pin down what happens when two
rules fire at once: applied in order as swaps on live state, `A -> B` then
`B -> C` is a 3-cycle. Its expected numbers are measured from
`fontmake -m Chain.designspace -i`, which is why it also carries `<instances>`.
