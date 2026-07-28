use std::f64::consts::PI;

use sim_lib_interference_core::{
    FieldAmplitude, Hertz, Metres, MetresPerSecond, NepersPerMetre, PositiveMetres, Radians,
};

fn main() {
    println!("metres={}", Metres::new(-2.5).unwrap().get());
    println!(
        "positive-metres={}",
        PositiveMetres::new(0.25).unwrap().get()
    );
    println!("frequency-hz={}", Hertz::new(440.0).unwrap().get());
    println!("speed-m-s={}", MetresPerSecond::new(343.0).unwrap().get());
    println!(
        "attenuation-np-m={}",
        NepersPerMetre::new(0.0).unwrap().get()
    );
    println!("phase-rad={}", Radians::new(PI).unwrap().get());
    println!(
        "field-amplitude={}",
        FieldAmplitude::new(0.0).unwrap().get()
    );
}
