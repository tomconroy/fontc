//! CLI args

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

// this env var can be set by the runner in order to reuse git checkouts
// between runs.
static GIT_CACHE_DIR_VAR: &str = "CRATER_GIT_CACHE";
const DEFAULT_CACHE_DIR: &str = "~/.fontc_crater_cache";

#[derive(Debug, PartialEq, Parser)]
#[command(about = "compile multiple fonts and report the results")]
pub(super) struct Args {
    #[command(subcommand)]
    pub(super) command: Commands,
}

#[derive(Debug, Subcommand, PartialEq)]
pub(super) enum Commands {
    Ci(CiArgs),
}

#[derive(Debug, PartialEq, clap::Args)]
pub(super) struct CiArgs {
    /// Path to a json list of repos + revs to run.
    pub(super) to_run: PathBuf,
    /// Directory to store font sources and the google/fonts repo.
    ///
    /// Reusing this directory saves us having to clone all the repos on each run.
    ///
    /// This can also be set via the CRATER_GIT_CACHE environment variable,
    /// although the CLI argument takes precedence.
    ///
    /// If no argument is provided, defaults to `~/.fontc_crater_cache`.
    #[arg(short, long = "cache")]
    cache_dir: Option<PathBuf>,

    /// Directory where results are written.
    ///
    /// This should be consistent between runs.
    #[arg(short = 'o', long = "out")]
    pub(super) out_dir: PathBuf,
    /// gftools mode (disable to reduce target count when running locally)
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub(super) gftools: bool,
    /// Which outline flavor to build and compare: 'ttf' (glyf), 'otf' (static
    /// CFF) or 'cff2' (variable CFF2). PostScript flavors are compared at
    /// fontmake's --optimize-cff 1.
    ///
    /// 'otf' and 'cff2' are the same ttx_diff mode -- it writes CFF for a
    /// static output and CFF2 for a variable one -- but they are separate
    /// crater modes because they are run over separate target lists, of
    /// static and of variable sources. Keeping them apart is what stops one
    /// corpus reading the other's cached fontmake output for a source that
    /// appears in both lists. Neither runs gftools builds, so neither runs
    /// the per-instance targets split from them. Use a separate out dir for
    /// each.
    #[arg(long, value_enum, default_value_t)]
    pub(super) flavor: Flavor,
    /// only generate html (for the provided out_dir)
    #[arg(long)]
    pub(super) html_only: bool,
    /// Rebuild every target without reading or writing cached results.
    ///
    /// Source checkouts in the cache dir are still used. This is for testing
    /// changes to python dependencies without invalidating the cache; use it
    /// with a scratch `--out` dir, since the next normal run compares against
    /// the last run recorded there.
    #[arg(long)]
    pub(super) no_results_cache: bool,
    /// File for live progress, published by the CI script.
    ///
    /// Crater rewrites it as targets finish.
    #[arg(long)]
    pub(super) progress_file: Option<PathBuf>,
    /// Where the report reads the live status, as published by the CI script.
    ///
    /// Without it, the report says that status isn't configured.
    #[arg(long)]
    pub(super) status_url: Option<String>,
}

/// Which outline flavor ttx_diff compiles and compares.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Flavor {
    #[default]
    Ttf,
    /// PostScript outlines from a static source: a `CFF ` table.
    Otf,
    /// PostScript outlines from a variable source: a `CFF2` table.
    ///
    /// ttx_diff has no separate mode for this — `--flavor otf` writes
    /// whichever table the output calls for — so this differs from
    /// [`Flavor::Otf`] only in what it names the cache and the report.
    Cff2,
}

impl Flavor {
    /// What ttx_diff calls this flavor. CFF and CFF2 are one mode there.
    pub(crate) fn ttx_diff_arg(self) -> &'static str {
        match self {
            Flavor::Ttf => "ttf",
            Flavor::Otf | Flavor::Cff2 => "otf",
        }
    }

    /// The extension of the font file this flavor produces. A variable CFF2
    /// is still an OpenType/CFF font, so still `.otf`.
    pub(crate) fn font_extension(self) -> &'static str {
        self.ttx_diff_arg()
    }
}

impl std::fmt::Display for Flavor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Flavor::Ttf => write!(f, "ttf"),
            Flavor::Otf => write!(f, "otf"),
            Flavor::Cff2 => write!(f, "cff2"),
        }
    }
}

impl CiArgs {
    /// Determine the directory to use for caching git checkouts.
    ///
    /// This may be passed at the command line or via an environment variable.
    pub(crate) fn cache_dir(&self) -> PathBuf {
        let cache_dir = self
            .cache_dir
            .clone()
            .or_else(|| std::env::var_os(GIT_CACHE_DIR_VAR).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR));

        if cache_dir.components().any(|comp| comp.as_os_str() == "~") {
            resolve_home(&cache_dir)
        } else {
            cache_dir
        }
    }
}

/// The cache/report component naming this run's outline flavor.
///
/// `None` for a ttf run, so that the cache it has always used stays where it
/// is. The distinction between "otf" and "cff2" exists precisely so that the
/// static and variable PostScript corpora cannot share a cache.
pub(crate) fn mode_name(flavor: Flavor) -> Option<String> {
    (flavor != Flavor::Ttf).then(|| flavor.to_string())
}

/// The ttx_diff flags that name this run's build mode.
///
/// A reproduction command has to ask for the same build the report is about;
/// without these it would compare a ttf whatever flavor the run was.
pub(crate) fn mode_flags(flavor: Flavor) -> String {
    match flavor {
        Flavor::Ttf => String::new(),
        _ => format!(" --flavor {}", flavor.ttx_diff_arg()),
    }
}

#[allow(deprecated)]
fn resolve_home(path: &Path) -> PathBuf {
    let Some(home_dir) = std::env::home_dir() else {
        log::warn!("No known home directory, ~ will not be resolved");
        return path.to_path_buf();
    };
    let mut result = PathBuf::new();
    for c in path.components() {
        if c.as_os_str() == "~" {
            result.push(home_dir.clone());
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse_ci(extra: &[&str]) -> CiArgs {
        let mut argv = vec!["fontc_crater", "ci", "targets.json", "-o", "out"];
        argv.extend_from_slice(extra);
        match Args::try_parse_from(argv).unwrap().command {
            Commands::Ci(args) => args,
        }
    }

    #[test]
    fn flavor_defaults_to_ttf() {
        assert_eq!(parse_ci(&[]).flavor, Flavor::Ttf);
    }

    /// CFF and CFF2 are one mode to ttx_diff and two to crater, so that the
    /// static and variable PostScript corpora get separate caches.
    #[test]
    fn cff2_is_the_otf_flavor_under_another_name() {
        let args = parse_ci(&["--flavor", "cff2"]);
        assert_eq!(args.flavor, Flavor::Cff2);
        assert_eq!(args.flavor.ttx_diff_arg(), "otf");
        assert_eq!(args.flavor.font_extension(), "otf");
        assert_ne!(mode_name(Flavor::Cff2), mode_name(Flavor::Otf));
    }

    /// The report is about one mode; its repro command has to name it.
    #[test]
    fn repro_flags_name_the_mode() {
        assert_eq!(mode_flags(Flavor::Ttf), "");
        assert_eq!(mode_flags(Flavor::Otf), " --flavor otf");
        assert_eq!(mode_flags(Flavor::Cff2), " --flavor otf");
    }

    #[test]
    fn mode_names() {
        assert_eq!(mode_name(Flavor::Ttf), None);
        assert_eq!(mode_name(Flavor::Otf).as_deref(), Some("otf"));
        assert_eq!(mode_name(Flavor::Cff2).as_deref(), Some("cff2"));
    }
}
