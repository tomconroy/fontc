use std::{io, path::PathBuf};

use thiserror::Error;
use write_fonts::types::Tag;

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
    // way they already do for "--flavor otf requires a static source".
    #[error("--instance requires a variable source; this source has no variable axes")]
    InstanceRequiresVariableSource,
    #[error("--instance does not know axis '{tag}'; this source has {available}")]
    UnknownInstanceAxis { tag: Tag, available: String },
    #[error("--instance puts '{tag}' at {pos}, outside its range {min}..={max}")]
    InstanceAxisOutOfRange {
        tag: Tag,
        pos: f64,
        min: f64,
        max: f64,
    },
    #[error("--instance does not know an instance named '{name}'; this source has {available}")]
    UnknownInstance { name: String, available: String },
    #[error("--instance cannot resolve that location: {0}")]
    InstanceLocation(String),
    #[error("--instance does not yet support a source with feature variation rules")]
    InstanceOfSourceWithRules,
    #[error("--instance does not yet support a feature file with a conditionset")]
    InstanceOfSourceWithFeaConditionSet,
    #[error("Unable to interpolate the instance: {0}")]
    InstanceDeltaError(#[from] fontdrasil::variations::DeltaError),
}
