* `resources/testdata/glyphs3/Oswald-AE-comb.glyphs`
   * Oswald with most of the glyphs deleted
   * After deletion go to Info > Features
      * Update language systems
      * Delete features
* `resources/testdata/glyphs4/`
   * The same-named `glyphs3/` fixtures, opened in Glyphs 4.1.1 (build 4108) and saved
     as file format 4 with `font.save(path, formatVersion=4, makeCopy=True)`; a
     `.glyphspackage` path saves a package. `Oswald-AE-comb.glyphspackage` and
     `WghtVarWithStylisticSet.glyphspackage` were saved from `.glyphs` twins.
   * Each compiles to the same font as its twin, but for `head.created`, which comes from
     the save date Glyphs adds, and the palette colors of `COLRv0-2layers.glyphs`, which
     Glyphs saves as grey with the same alpha
