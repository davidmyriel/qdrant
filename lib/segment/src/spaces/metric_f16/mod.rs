pub mod simple_cosine;
pub mod simple_dot;
pub mod simple_euclid;
pub mod simple_manhattan;

#[cfg(target_arch = "x86_64")]
pub mod avx;

#[cfg(all(target_arch = "aarch64", not(windows)))]
pub mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse;

use half::f16;
use half::slice::HalfFloatSliceExt as _;

/// Number of elements converted from `f16` to `f32` at once by [`lanewise_sum`].
const CONVERT_BLOCK: usize = 64;

/// Number of independent accumulators used by [`lanewise_sum`].
const LANES: usize = 8;

/// Sums `op(a, b)` over the element pairs of `v1` and `v2`.
///
/// Converting `f16` one element at a time costs a CPU feature check and a call per element,
/// and a single `f32` accumulator forces a serial chain of additions. Instead, convert in
/// blocks and accumulate into independent lanes, so the compiler can vectorize the loop.
#[inline]
pub(super) fn lanewise_sum(v1: &[f16], v2: &[f16], op: impl Fn(f32, f32) -> f32) -> f32 {
    let len = v1.len().min(v2.len());
    let mut buf1 = [0f32; CONVERT_BLOCK];
    let mut buf2 = [0f32; CONVERT_BLOCK];
    let mut lanes = [0f32; LANES];
    let mut tail = 0f32;

    for (c1, c2) in v1[..len]
        .chunks(CONVERT_BLOCK)
        .zip(v2[..len].chunks(CONVERT_BLOCK))
    {
        let (b1, b2) = (&mut buf1[..c1.len()], &mut buf2[..c2.len()]);
        c1.convert_to_f32_slice(b1);
        c2.convert_to_f32_slice(b2);

        let ((l1, r1), (l2, r2)) = (b1.as_chunks::<LANES>(), b2.as_chunks::<LANES>());
        for (a, b) in r1.iter().zip(r2) {
            tail += op(*a, *b);
        }
        for (a, b) in l1.iter().zip(l2) {
            for ((lane, a), b) in lanes.iter_mut().zip(a).zip(b) {
                *lane += op(*a, *b);
            }
        }
    }

    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanewise_sum_matches_naive() {
        // Lengths around the lane and conversion block boundaries.
        for len in [0, 1, 7, 8, 9, 63, 64, 65, 129, 1000] {
            let v1: Vec<f16> = (0..len)
                .map(|i| f16::from_f32((i % 13) as f32 * 0.25 - 1.5))
                .collect();
            let v2: Vec<f16> = (0..len)
                .map(|i| f16::from_f32((i % 7) as f32 * 0.5 - 1.0))
                .collect();

            let ops: [fn(f32, f32) -> f32; 3] =
                [|a, b| a * b, |a, b| (a - b).powi(2), |a, b| (a - b).abs()];
            for op in ops {
                let naive: f32 = v1
                    .iter()
                    .zip(&v2)
                    .map(|(a, b)| op(a.to_f32(), b.to_f32()))
                    .sum();
                let lanewise = lanewise_sum(&v1, &v2, op);
                assert!(
                    (naive - lanewise).abs() <= 1e-3 * naive.abs().max(1.0),
                    "len {len}: naive {naive}, lanewise {lanewise}",
                );
            }
        }
    }
}
