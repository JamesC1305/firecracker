use std::path::PathBuf;

use serde::Deserialize;

/// Stores options for configuring checkpoint creation.
#[derive(Debug, Deserialize, Eq, PartialEq)]
pub struct CreateCheckpointParams {
    /// Path that the handler listens on for checkpoint operations.
    pub uffd_path: PathBuf,
}
