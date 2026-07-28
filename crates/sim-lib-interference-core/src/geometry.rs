//! Finite three-dimensional points and checked directions.

use crate::{InterferenceError, Metres};

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// A point in three-dimensional Cartesian space measured in metres.
///
/// Each coordinate is a [`Metres`], so a `Point3M` cannot contain NaN or an
/// infinity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3M {
    x: Metres,
    y: Metres,
    z: Metres,
}

impl Point3M {
    /// Constructs a point from already checked coordinates.
    pub fn new(x: Metres, y: Metres, z: Metres) -> Self {
        Self { x, y, z }
    }

    /// Checks three raw coordinates and constructs a point.
    pub fn from_metres(x: f64, y: f64, z: f64) -> Result<Self, InterferenceError> {
        Ok(Self::new(Metres::new(x)?, Metres::new(y)?, Metres::new(z)?))
    }

    /// Returns the x coordinate.
    pub fn x(self) -> Metres {
        self.x
    }

    /// Returns the y coordinate.
    pub fn y(self) -> Metres {
        self.y
    }

    /// Returns the z coordinate.
    pub fn z(self) -> Metres {
        self.z
    }

    /// Returns the Cartesian coordinates as metres.
    pub fn coordinates_metres(self) -> [f64; 3] {
        [self.x.get(), self.y.get(), self.z.get()]
    }

    /// Returns the Euclidean distance to another point in metres.
    ///
    /// Extremely separated finite coordinates can have a distance beyond the
    /// finite `f64` range. Propagation entry points reject that derived value.
    pub fn distance_to(self, other: Self) -> f64 {
        let [x, y, z] = self.coordinates_metres();
        let [other_x, other_y, other_z] = other.coordinates_metres();
        (x - other_x).hypot(y - other_y).hypot(z - other_z)
    }
}

/// A finite, normalized direction in three-dimensional Cartesian space.
///
/// Construction accepts any finite non-zero vector and normalizes it with a
/// scaled norm, avoiding intermediate overflow for large finite components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitVector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl UnitVector3 {
    /// Checks and normalizes a direction.
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, InterferenceError> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(InterferenceError::InvalidDirection { x, y, z });
        }
        let scale = x.abs().max(y.abs()).max(z.abs());
        if scale == 0.0 {
            return Err(InterferenceError::InvalidDirection { x, y, z });
        }

        let scaled_x = x / scale;
        let scaled_y = y / scale;
        let scaled_z = z / scale;
        let norm = scaled_x.hypot(scaled_y).hypot(scaled_z);

        Ok(Self {
            x: canonical_zero(scaled_x / norm),
            y: canonical_zero(scaled_y / norm),
            z: canonical_zero(scaled_z / norm),
        })
    }

    /// Returns the normalized Cartesian components.
    pub fn components(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Projects the displacement from `from` to `to` onto this direction.
    ///
    /// The result is a signed distance in metres. Propagation entry points
    /// reject a non-finite derived value and negative forward-plane distances.
    pub fn signed_distance_metres(self, from: Point3M, to: Point3M) -> f64 {
        let [from_x, from_y, from_z] = from.coordinates_metres();
        let [to_x, to_y, to_z] = to.coordinates_metres();
        self.x * (to_x - from_x) + self.y * (to_y - from_y) + self.z * (to_z - from_z)
    }
}
