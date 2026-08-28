use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("'{0}' exists but is not a directory")]
    ExpectedDirectory(PathBuf),
    #[error("io failed for '{path}': '{source}'")]
    FileIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write to stdout or stderr: '{0}'")]
    StdioWriteFail(#[source] io::Error),
    #[error("Unrecognized source {0}")]
    UnrecognizedSource(PathBuf),
    #[error(transparent)]
    YamlSerError(#[from] serde_yaml::Error),
    #[error(transparent)]
    FontIrError(#[from] fontir::error::Error),
    #[error(transparent)]
    Backend(#[from] fontbe::error::Error),
    #[error("Missing file '{0}'")]
    FileExpected(PathBuf),
    #[error("Unable to proceed; {0} jobs stuck pending")]
    UnableToProceed(usize),
    #[error("No output file specified")]
    NoOutputFile,
    #[error("A task panicked: '{0}'")]
    Panic(String),
    // --instance. The wording of these is load-bearing: ttx_diff and
    // fontc_crater classify a source they cannot compare by matching on it, the
    // way they already do for "--flavor otf requires a static source". The
    // resolution errors live in fontir because the global metrics work resolves
    // the pin too, long before the pin barrier runs.
    #[error(transparent)]
    Pin(#[from] fontir::instance::PinError),
    #[error(
        "--instance cannot apply the feature variation rule substituting '{replace}': there is no glyph '{with}' to swap it with"
    )]
    InstanceRuleSubstituteMissing { replace: String, with: String },
    #[error("--instance does not yet support a feature file with a conditionset")]
    InstanceOfSourceWithFeaConditionSet,
    #[error("Unable to interpolate the instance: {0}")]
    InstanceDeltaError(#[from] fontdrasil::variations::DeltaError),
}
