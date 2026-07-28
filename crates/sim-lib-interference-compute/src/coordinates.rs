//! Prepared local-coordinate and scalar f32 Tensors.

use std::sync::Arc;

use sim_lib_interference_core::SamplingPlane;
use sim_lib_numbers_tensor::{Tensor, TypedTensorStorage, domains};

use crate::{LoweringError, PlaneTile, PreflightCheck};

pub(crate) struct LocalCoordinates {
    pub(crate) exact: Vec<[f64; 3]>,
    pub(crate) lowered: Vec<[f32; 3]>,
    pub(crate) tensors: [Tensor; 3],
}

pub(crate) fn prepare_local_coordinates(
    plane: SamplingPlane,
    tile: PlaneTile,
) -> Result<LocalCoordinates, LoweringError> {
    let element_count = tile.rows() * tile.columns();
    let mut exact = Vec::with_capacity(element_count);
    let mut lowered_coordinates = Vec::with_capacity(element_count);
    let mut components = [
        Vec::with_capacity(element_count),
        Vec::with_capacity(element_count),
        Vec::with_capacity(element_count),
    ];
    let [ux, uy, uz] = plane.u_axis().components();
    let [vx, vy, vz] = plane.v_axis().components();
    let center_column = (tile.columns().saturating_sub(1) as f64) * 0.5;
    let center_row = (tile.rows().saturating_sub(1) as f64) * 0.5;
    for row in 0..tile.rows() {
        let offset_v = (row as f64 - center_row) * plane.cell_size_v_m();
        for column in 0..tile.columns() {
            let offset_u = (column as f64 - center_column) * plane.cell_size_u_m();
            let offset = [
                offset_u * ux + offset_v * vx,
                offset_u * uy + offset_v * vy,
                offset_u * uz + offset_v * vz,
            ];
            let lowered = [
                local_component("x", offset[0])?,
                local_component("y", offset[1])?,
                local_component("z", offset[2])?,
            ];
            exact.push(offset);
            lowered_coordinates.push(lowered);
            for axis in 0..3 {
                components[axis].push(lowered[axis]);
            }
        }
    }
    let shape = tile.shape().to_vec();
    let [x, y, z] = components;
    Ok(LocalCoordinates {
        exact,
        lowered: lowered_coordinates,
        tensors: [
            tensor(shape.clone(), x)?,
            tensor(shape.clone(), y)?,
            tensor(shape, z)?,
        ],
    })
}

pub(crate) fn scalar(value: f32) -> Result<Tensor, LoweringError> {
    if !value.is_finite() {
        return Err(LoweringError::new(
            PreflightCheck::Constant,
            format!("scalar Tensor constant {value} is not finite"),
        ));
    }
    tensor(Vec::new(), vec![value])
}

fn tensor(shape: Vec<usize>, cells: Vec<f32>) -> Result<Tensor, LoweringError> {
    Tensor::from_storage(
        shape,
        domains::f32(),
        Arc::new(TypedTensorStorage::<f32>::new(cells)),
    )
    .map_err(|error| {
        LoweringError::new(
            PreflightCheck::Shape,
            format!("cannot construct local f32 Tensor: {error}"),
        )
    })
}

fn local_component(axis: &str, value: f64) -> Result<f32, LoweringError> {
    let lowered = value as f32;
    if value.is_finite() && lowered.is_finite() {
        Ok(lowered)
    } else {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            format!("tile-local {axis} offset {value} is not representable as finite f32"),
        ))
    }
}
