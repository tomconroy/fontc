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
    /// Which outline flavor to build and compare: 'ttf' (glyf) or 'otf'
    /// (static CFF, compared at fontmake's --optimize-cff 1).
    ///
    /// 'otf' only runs default (non-gftools) builds and expects a target list
    /// of static sources; variable sources are reported as skipped. Use a
    /// separate out dir and expect a separate results cache from ttf runs.
    #[arg(long, value_enum, default_value_t)]
    pub(super) flavor: Flavor,
    /// Build one static instance of each variable source instead of a variable
    /// font, and compare it against `fontmake -i <instance> --keep-overlaps`.
    ///
    /// The value is handed to ttx_diff: '@default' picks the named instance
    /// sitting at the default location (a source without one is reported as
    /// skipped), and anything else is a literal DesignSpace instance name,
    /// which only makes sense for a hand-picked target list.
    ///
    /// Only runs default (non-gftools) builds and expects a target list of
    /// variable sources; static sources are reported as skipped. Composes with
    /// --flavor. Use a separate out dir; results are cached separately.
    #[arg(long, value_name = "INSTANCE")]
    pub(super) instance: Option<String>,
    /// only generate html (for the provided out_dir)
    #[arg(long)]
    pub(super) html_only: bool,
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
    Otf,
}

impl std::fmt::Display for Flavor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Flavor::Ttf => write!(f, "ttf"),
            Flavor::Otf => write!(f, "otf"),
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

/// The cache/report component naming this combination of build modes.
///
/// `None` for a plain variable ttf run, so that the cache it has always used
/// stays where it is; otherwise flavor and instance compose into one name,
/// e.g. "otf", "instance@default", "otf-instance@default".
pub(crate) fn mode_name(flavor: Flavor, instance: Option<&str>) -> Option<String> {
    let mut name = String::new();
    if flavor != Flavor::Ttf {
        name.push_str(&flavor.to_string());
    }
    if let Some(instance) = instance {
        if !name.is_empty() {
            name.push('-');
        }
        name.push_str("instance");
        // an instance can be named anything at all ('Bona Nova/Bold'), and this
        // ends up as a directory name
        name.extend(instance.chars().map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        }));
    }
    (!name.is_empty()).then_some(name)
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
    fn instance_defaults_to_off() {
        let args = parse_ci(&[]);
        assert_eq!(args.instance, None);
        assert_eq!(args.flavor, Flavor::Ttf);
    }

    #[test]
    fn instance_is_a_global_flag() {
        // deliberately not per-target: a Target serializes as a string used as
        // a json map key, so it has no room for build options
        let args = parse_ci(&["--instance", "@default"]);
        assert_eq!(args.instance.as_deref(), Some("@default"));
    }

    #[test]
    fn instance_composes_with_flavor() {
        let args = parse_ci(&["--instance", "@default", "--flavor", "otf"]);
        assert_eq!(args.instance.as_deref(), Some("@default"));
        assert_eq!(args.flavor, Flavor::Otf);
    }

    #[test]
    fn mode_names() {
        assert_eq!(mode_name(Flavor::Ttf, None), None);
        assert_eq!(mode_name(Flavor::Otf, None).as_deref(), Some("otf"));
        assert_eq!(
            mode_name(Flavor::Ttf, Some("@default")).as_deref(),
            Some("instance@default")
        );
        assert_eq!(
            mode_name(Flavor::Otf, Some("@default")).as_deref(),
            Some("otf-instance@default")
        );
        // an instance name is arbitrary text; it cannot escape the cache dir
        assert_eq!(
            mode_name(Flavor::Ttf, Some("../Bona Nova Bold")).as_deref(),
            Some("instance.._Bona_Nova_Bold")
        );
    }
}
