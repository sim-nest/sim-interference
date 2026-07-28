use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingCertificate, SamplingPlane, SamplingPolicy,
    SamplingThresholds, SamplingVerdict, ScalarMedium, SourceSet, UnitVector3,
};

fn plane(extent_m: f64, cells_per_axis: usize) -> SamplingPlane {
    SamplingPlane::new(
        Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(extent_m).unwrap(),
        PositiveMetres::new(extent_m).unwrap(),
        cells_per_axis,
        cells_per_axis,
    )
    .unwrap()
}

fn problem(frequency_hz: f64, sources: Vec<Emitter>) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(frequency_hz).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(sources).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

fn forward_plane() -> Emitter {
    Emitter::ForwardPlane {
        id: "plane".to_owned(),
        through: Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        amplitude: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    }
}

fn point(z_m: f64) -> Emitter {
    Emitter::Point {
        id: "point".to_owned(),
        position: Point3M::from_metres(0.5, 0.5, z_m).unwrap(),
        amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    }
}

#[test]
fn certificate_records_carrier_and_doubled_power_bandwidth() {
    let certificate =
        SamplingCertificate::measure(&problem(40_000.0, vec![forward_plane()]), &plane(2.0, 200))
            .unwrap();

    assert_eq!(certificate.wavelength_m, 343.0 / 40_000.0);
    assert_eq!(
        certificate.samples_per_power_fringe_u,
        certificate.samples_per_wavelength_u / 2.0
    );
    assert_eq!(
        certificate.samples_per_power_fringe_v,
        certificate.samples_per_wavelength_v / 2.0
    );
    assert_eq!(certificate.nearest_point_source_distance_m, None);
    assert_eq!(certificate.max_envelope_fraction_per_cell, 0.0);
    assert_eq!(certificate.verdict, SamplingVerdict::Aliased);
}

#[test]
fn nearest_point_source_and_conservative_envelope_are_measured() {
    let certificate =
        SamplingCertificate::measure(&problem(1_000.0, vec![point(1.0)]), &plane(1.0, 100))
            .unwrap();

    assert_eq!(certificate.nearest_point_source_distance_m, Some(1.0));
    let diagonal = 0.01_f64.hypot(0.01);
    let expected = diagonal;
    assert_eq!(certificate.max_envelope_fraction_per_cell, expected);
    assert_eq!(certificate.verdict, SamplingVerdict::Resolved);
}

#[test]
fn coarse_near_source_sampling_has_a_finite_conservative_bound() {
    let certificate =
        SamplingCertificate::measure(&problem(1_000.0, vec![point(0.1)]), &plane(1.0, 1)).unwrap();

    assert_eq!(certificate.nearest_point_source_distance_m, Some(0.1));
    assert_eq!(
        certificate.max_envelope_fraction_per_cell,
        1.0_f64.hypot(1.0) / 0.1
    );
    assert_eq!(certificate.verdict, SamplingVerdict::Aliased);
}

#[test]
fn a_point_source_touching_the_plane_has_no_finite_envelope_certificate() {
    assert!(matches!(
        SamplingCertificate::measure(&problem(1_000.0, vec![point(0.0)]), &plane(1.0, 100)),
        Err(
            sim_lib_interference_core::InterferenceError::UnboundedSamplingEnvelope {
                nearest_point_source_distance_m: 0.0,
                ..
            }
        )
    ));
}

#[test]
fn explicit_threshold_records_control_all_three_verdicts() {
    let problem = problem(40_000.0, vec![forward_plane()]);
    let marginal = SamplingCertificate::measure(&problem, &plane(2.0, 1_000)).unwrap();
    let resolved = SamplingCertificate::measure(&problem, &plane(2.0, 2_000)).unwrap();

    assert_eq!(
        SamplingThresholds::default().resolved_min_samples_per_wavelength,
        8.0
    );
    assert_eq!(
        SamplingThresholds::default().marginal_min_samples_per_wavelength,
        4.0
    );
    assert_eq!(marginal.verdict, SamplingVerdict::Marginal);
    assert_eq!(resolved.verdict, SamplingVerdict::Resolved);

    let relaxed = SamplingThresholds::new(4.0, 2.0, 0.10, 0.20).unwrap();
    let reclassified =
        SamplingCertificate::measure_with_thresholds(&problem, &plane(2.0, 1_000), relaxed)
            .unwrap();
    assert_eq!(reclassified.thresholds, relaxed);
    assert_eq!(reclassified.verdict, SamplingVerdict::Resolved);
}

#[test]
fn thresholds_are_validated_and_strict_is_fail_closed() {
    assert!(SamplingThresholds::new(4.0, 8.0, 0.05, 0.10).is_err());
    assert!(SamplingThresholds::new(8.0, 4.0, 0.20, 0.10).is_err());
    assert!(SamplingThresholds::new(f64::NAN, 4.0, 0.05, 0.10).is_err());
    let invalid_record = SamplingThresholds {
        resolved_min_samples_per_wavelength: f64::NAN,
        ..SamplingThresholds::default()
    };
    assert!(
        SamplingCertificate::measure_with_thresholds(
            &problem(1_000.0, vec![forward_plane()]),
            &plane(1.0, 100),
            invalid_record,
        )
        .is_err()
    );

    let aliased =
        SamplingCertificate::measure(&problem(40_000.0, vec![forward_plane()]), &plane(2.0, 200))
            .unwrap();
    assert_eq!(SamplingPolicy::default(), SamplingPolicy::Strict);
    assert_eq!(
        SamplingPolicy::Strict.admit(&aliased),
        Err(
            sim_lib_interference_core::InterferenceError::SamplingRefused {
                certificate: aliased,
            }
        )
    );
    assert_eq!(SamplingPolicy::Annotate.admit(&aliased), Ok(()));
}
