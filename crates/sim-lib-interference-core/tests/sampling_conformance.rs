use sim_lib_interference_core::{
    InterferenceError, Point3M, PositiveMetres, SamplingPlane, UnitVector3,
};

fn plane(
    u_axis: UnitVector3,
    v_axis: UnitVector3,
    rows: usize,
    columns: usize,
) -> Result<SamplingPlane, InterferenceError> {
    SamplingPlane::new(
        Point3M::from_metres(-1.0, -2.0, 3.0).unwrap(),
        u_axis,
        v_axis,
        PositiveMetres::new(2.0).unwrap(),
        PositiveMetres::new(4.0).unwrap(),
        rows,
        columns,
    )
}

#[test]
fn pixel_centres_and_right_handed_normal_are_exactly_defined() {
    let plane = plane(
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        2,
        2,
    )
    .unwrap();

    assert_eq!(plane.cell_count(), 4);
    assert_eq!(plane.cell_size_u_m(), 1.0);
    assert_eq!(plane.cell_size_v_m(), 2.0);
    assert_eq!(plane.normal().components(), [0.0, 0.0, 1.0]);
    assert_eq!(
        plane.point_at(0, 0).unwrap().coordinates_metres(),
        [-0.5, -1.0, 3.0]
    );
    assert_eq!(
        plane.point_at(1, 1).unwrap().coordinates_metres(),
        [0.5, 1.0, 3.0]
    );
}

#[test]
fn invalid_frames_dimensions_and_indices_fail_closed() {
    let x = UnitVector3::new(1.0, 0.0, 0.0).unwrap();
    let y = UnitVector3::new(0.0, 1.0, 0.0).unwrap();
    let skew = UnitVector3::new(1.0, 1.0, 0.0).unwrap();

    assert!(matches!(
        plane(x, skew, 2, 2),
        Err(InterferenceError::NonOrthogonalSamplingAxes { .. })
    ));
    assert_eq!(
        plane(x, y, 0, 2),
        Err(InterferenceError::ZeroSamplingDimension { name: "rows" })
    );
    assert_eq!(
        plane(x, y, 2, 0),
        Err(InterferenceError::ZeroSamplingDimension { name: "columns" })
    );

    let valid = plane(x, y, 2, 2).unwrap();
    assert_eq!(
        valid.point_at(2, 0),
        Err(InterferenceError::SamplingCellOutOfBounds {
            row: 2,
            column: 0,
            rows: 2,
            columns: 2,
        })
    );
}

#[test]
fn cell_count_overflow_is_rejected_during_construction() {
    let error = plane(
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        usize::MAX,
        2,
    )
    .unwrap_err();

    assert_eq!(
        error,
        InterferenceError::SamplingCellCountOverflow {
            rows: usize::MAX,
            columns: 2,
        }
    );
}
// conformance: sampling tests prove physical grid geometry and resolution classification.
