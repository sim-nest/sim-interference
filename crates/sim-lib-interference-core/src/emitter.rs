//! Coherent scalar emitters and their canonical source collection.

use crate::{FieldAmplitude, InterferenceError, Point3M, Radians, UnitVector3};

/// A coherent scalar-wave emitter.
///
/// Emitters deliberately carry no frequency. One frequency is owned by the
/// complete interference problem so unlike phasors cannot be mixed.
#[derive(Clone, Debug, PartialEq)]
pub enum Emitter {
    /// An outgoing spherical point source.
    Point {
        /// Stable source identity.
        id: String,
        /// Source position.
        position: Point3M,
        /// Field amplitude stated at
        /// [`crate::POINT_SOURCE_REFERENCE_DISTANCE_METRES`].
        amplitude_at_reference: FieldAmplitude,
        /// Source phase offset.
        phase: Radians,
    },
    /// A plane wave admitted only in its forward half-space.
    ForwardPlane {
        /// Stable source identity.
        id: String,
        /// A point on the zero-phase plane.
        through: Point3M,
        /// Forward propagation direction.
        direction: UnitVector3,
        /// Field amplitude on the zero-phase plane.
        amplitude: FieldAmplitude,
        /// Source phase offset.
        phase: Radians,
    },
}

impl Emitter {
    /// Returns the source's stable identity.
    pub fn id(&self) -> &str {
        match self {
            Self::Point { id, .. } | Self::ForwardPlane { id, .. } => id,
        }
    }
}

/// A non-empty collection of emitters in canonical stable-id order.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceSet {
    sources: Vec<Emitter>,
}

impl SourceSet {
    /// Validates source identities and sorts the sources by their id bytes.
    pub fn new(mut sources: Vec<Emitter>) -> Result<Self, InterferenceError> {
        if sources.is_empty() {
            return Err(InterferenceError::EmptySourceSet);
        }
        if sources.iter().any(|source| source.id().is_empty()) {
            return Err(InterferenceError::EmptySourceId);
        }

        sources.sort_unstable_by(|left, right| left.id().cmp(right.id()));
        if let Some(pair) = sources.windows(2).find(|pair| pair[0].id() == pair[1].id()) {
            return Err(InterferenceError::DuplicateSourceId {
                id: pair[0].id().to_owned(),
            });
        }

        Ok(Self { sources })
    }

    /// Returns the number of sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Returns whether the set is empty.
    ///
    /// This is always false for a successfully constructed `SourceSet`.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Returns the canonically ordered sources.
    pub fn as_slice(&self) -> &[Emitter] {
        &self.sources
    }

    /// Iterates over sources in canonical stable-id order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Emitter> {
        self.sources.iter()
    }
}

impl<'a> IntoIterator for &'a SourceSet {
    type Item = &'a Emitter;
    type IntoIter = std::slice::Iter<'a, Emitter>;

    fn into_iter(self) -> Self::IntoIter {
        self.sources.iter()
    }
}
