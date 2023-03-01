#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use blstrs::Bls12;
use ec_gpu::GpuName;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder};
use ec_gpu_gen::{
    multiexp::MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker, EcError,
};
use ff::{Field, PrimeField};
use group::Curve;
use group::{prime::PrimeCurveAffine, Group};
use pairing::Engine;

fn multiexp_gpu<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Curve, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + GpuName,
    S: SourceBuilder<G>,
{
    let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
    let (bss, skip) = bases.get();
    kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
}

#[test]
fn gpu_multiexp_consistency() {
    fil_logger::maybe_init();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();

    for samples in [131076643] {
        println!("Testing Multiexp for {} elements...", samples);
        let g_fixed = <Bls12 as Engine>::G1::generator().to_affine();
        let g = Arc::new(vec![g_fixed; samples]);

        let v_fixed = <Bls12 as Engine>::Fr::one().to_repr();
        let v = Arc::new(vec![v_fixed; samples]);

        let mut now = Instant::now();
        let gpu = multiexp_gpu(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        let result_fixed = blstrs::G1Projective::from_uncompressed(&[
            16, 223, 43, 198, 141, 83, 114, 254, 199, 111, 184, 184, 17, 144, 191, 52, 235, 206,
            97, 113, 34, 144, 203, 77, 60, 61, 34, 141, 74, 164, 42, 0, 213, 114, 216, 222, 133,
            210, 144, 107, 174, 42, 246, 191, 150, 247, 233, 168, 7, 115, 255, 199, 213, 203, 136,
            98, 120, 81, 240, 38, 198, 45, 172, 243, 200, 145, 55, 239, 134, 48, 211, 57, 119, 84,
            42, 148, 41, 9, 120, 114, 102, 174, 79, 252, 100, 165, 4, 86, 188, 96, 255, 177, 7,
            123, 38, 114,
        ])
        .unwrap();
        assert_eq!(gpu, result_fixed);

        now = Instant::now();
        let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
            .wait()
            .unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu, gpu);
        assert_eq!(cpu, result_fixed);

        println!("============================");
    }
}
