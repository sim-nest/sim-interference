//! Stable public diagnostics for interference-domain validation.

use std::fmt;

/// An invalid value at an interference-domain boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterferenceError {
    /// A quantity violated its finite, sign, or zero contract.
    InvalidQuantity {
        /// Stable public quantity name.
        name: &'static str,
        /// Rejected caller-supplied value.
        value: f64,
    },
}

impl fmt::Display for InterferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuantity { name, value } => {
                write!(formatter, "invalid quantity `{name}`: {value:?}")
            }
        }
    }
}

impl std::error::Error for InterferenceError {}
