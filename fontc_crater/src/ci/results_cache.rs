//! Caching fontmake's output between runs

use std::path::{Path, PathBuf};

use crate::{Target, args::Flavor};

static CACHE_DIR_NAME: &str = "crater_cached_results";

// the flavor-independent files that we cache for each target
static TTX_FILE: &str = "fontmake.ttx";
static MARKKERN_FILE: &str = "fontmake.markkern.txt";

/// Manages a cache of files on disk
pub(crate) struct ResultsCache {
    base_results_cache_dir: PathBuf,
    // "fontmake.ttf" or "fontmake.otf"; the ttx/markkern names are shared, so
    // each flavor gets its own cache tree (see [`ResultsCache::in_dir`])
    font_file: String,
}

impl ResultsCache {
    /// argument is the directory that will contain the cache dir.
    ///
    /// By convention this is the same directory where we checkout git repos.
    /// The ttf cache lives where it always did; other flavors get a sibling
    /// subdirectory, since the cached ttx/markkern files are named the same
    /// but their contents differ per flavor.
    pub fn in_dir(path: &Path, flavor: Flavor) -> Self {
        let mut base_results_cache_dir = path.join(CACHE_DIR_NAME);
        if flavor != Flavor::Ttf {
            base_results_cache_dir.push(flavor.to_string());
        }
        Self {
            base_results_cache_dir,
            font_file: format!("fontmake.{flavor}"),
        }
    }

    /// Delete any cache contents
    pub fn delete_all(&self) {
        if self.base_results_cache_dir.exists() {
            std::fs::remove_dir_all(&self.base_results_cache_dir).expect("failed to remove cache")
        }
    }

    /// if we have cached files for this target, copy them into the build directory.
    pub fn copy_cached_files_to_build_dir(&self, target: &Target, build_dir: &Path) {
        let target_cache_dir = target.cache_dir(&self.base_results_cache_dir);
        if !target_cache_dir.exists() {
            log::trace!("no cached files for {target}");
            return;
        }

        if self.copy_cache_files(&target_cache_dir, build_dir).unwrap() {
            log::trace!("reused cached files for {target}",);
        }
    }

    /// Copy files generated from a previous run into the permanent cache.
    pub fn save_built_files_to_cache(&self, target: &Target, build_dir: &Path) {
        let target_cache_dir = target.cache_dir(&self.base_results_cache_dir);
        if !target_cache_dir.exists() {
            std::fs::create_dir_all(&target_cache_dir).unwrap();
        }
        // no need to overwrite existing cache
        if [self.font_file.as_str(), TTX_FILE, MARKKERN_FILE]
            .into_iter()
            .all(|p| target_cache_dir.join(p).exists())
        {
            return;
        }
        if self.copy_cache_files(build_dir, &target_cache_dir).unwrap() {
            log::trace!("saved cached files for {target}");
        }
    }

    fn copy_cache_files(&self, from_dir: &Path, to_dir: &Path) -> std::io::Result<bool> {
        let font = from_dir.join(&self.font_file);
        let ttx = from_dir.join(TTX_FILE);
        let markkern = from_dir.join(MARKKERN_FILE);

        if [&font, &ttx, &markkern].into_iter().all(|p| p.exists()) {
            if !to_dir.exists() {
                std::fs::create_dir_all(to_dir)?;
            }
            std::fs::copy(font, to_dir.join(&self.font_file)).map(|_| ())?;
            std::fs::copy(ttx, to_dir.join(TTX_FILE)).map(|_| ())?;
            std::fs::copy(markkern, to_dir.join(MARKKERN_FILE)).map(|_| ())?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
