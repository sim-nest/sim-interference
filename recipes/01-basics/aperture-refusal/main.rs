use sim_lib_interference_core::{
    FieldAmplitude, Hertz, MetresPerSecond, NepersPerMetre, Point3M, PositiveMetres, Radians,
    ScalarMedium, UnitVector3,
};
use sim_lib_interference_solve::{
    AperturePolicy, STRICT_MAX_SPACING_WAVELENGTHS, ScenarioBuilder, ScenarioError,
    ScenarioLimits,
};

fn main() {
    let builder = ScenarioBuilder::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(2.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        PositiveMetres::new(0.01).unwrap(),
        ScenarioLimits::default(),
    );
    let result = builder.phased_array(
        Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        2,
        PositiveMetres::new(1.1).unwrap(),
        FieldAmplitude::new(1.0).unwrap(),
        Radians::new(0.0).unwrap(),
        Radians::new(0.0).unwrap(),
        AperturePolicy::Strict,
    );
    let Err(ScenarioError::SparseAperture {
        axis,
        spacing_wavelengths,
        ..
    }) = result
    else {
        unreachable!("strict sparse aperture must be refused");
    };
    println!(
        "strict=refused axis={axis} spacing={spacing_wavelengths:.3} wavelengths maximum={STRICT_MAX_SPACING_WAVELENGTHS:.3}"
    );
}
