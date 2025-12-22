use faultforge::prelude::*;
use numpy::PyReadwriteArrayDyn;
use pyo3::{exceptions::PyValueError, prelude::*};

use rayon::prelude::*;

use faultforge::encoding::embedded_parity::{decode, encode};

struct Scheme<const N: usize, const M: usize>([([usize; N], usize); M]);

/// 3 data bits, 1 parity bit for 32 bit buffers.
const B32D3P1: Scheme<3, 8> = Scheme([
    // data bits on the left and parity bits on the right.
    // 0bHHHG_GGFF_FEEE_DDDC_CCBB_BAAA_HGFE_DCBA
    ([31, 30, 29], 7), // H
    ([28, 27, 26], 6), // G
    ([25, 24, 23], 5), // F
    ([22, 21, 20], 4), // E
    ([19, 18, 17], 3), // D
    ([16, 15, 14], 2), // C
    ([13, 12, 11], 1), // B
    ([10, 9, 8], 0),   // A
]);

/// 7 data bits, 1 parity bit for 32 bit buffers.
const B32D7P1: Scheme<7, 4> = Scheme([
    // data bits on the left and parity bits on the right.
    // 0bDDDD_DDDC_CCCC_CCBB_BBBB_BAAA_AAAA_DCBA
    ([31, 30, 29, 28, 27, 26, 25], 3), // D
    ([24, 23, 22, 21, 20, 19, 18], 2), // C
    ([17, 16, 15, 14, 13, 12, 11], 1), // B
    ([10, 9, 8, 7, 6, 5, 4], 0),       // A
]);

/// 15 data bits, 1 parity bit for 32 bit buffers.
const B32D15P1: Scheme<15, 2> = Scheme([
    (
        [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17],
        1,
    ),
    ([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], 0),
]);

/// 3 data bits, 1 parity bit for 16 bit buffers.
const B16D3P1: Scheme<3, 4> = Scheme([
    // data bits on the left and parity bits on the right. 0bDDDC_CCBB_BAAA_DCBA
    ([15, 14, 13], 3), // D
    ([12, 11, 10], 2), // C
    ([9, 8, 7], 1),    // B
    ([6, 5, 4], 0),    // A
]);

/// 7 data bits, 1 parity bit for 16 bit buffers.
const B16D7P1: Scheme<7, 2> = Scheme([
    // data bits on the left and parity bits on the right. 0bBBBB_BBBA_AAAA_AABA
    ([15, 14, 13, 12, 11, 10, 9], 1), // B
    ([8, 7, 6, 5, 4, 3, 2], 0),       // A
]);

/// 15 data bits, 1 parity bit for 32 bit buffers.
const B16D15P1: Scheme<15, 1> = Scheme([([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 0)]);

impl<const N: usize, const M: usize> Scheme<N, M> {
    fn encode<B>(&self, item: &mut B)
    where
        B: BitBuffer,
    {
        for (source, dest) in self.0 {
            encode(source, dest, item).unwrap_or_else(|err| {
                panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
            })
        }
    }

    fn decode<B>(&self, item: &mut B)
    where
        B: BitBuffer,
    {
        for (source, dest) in self.0 {
            decode(source, dest, item).unwrap_or_else(|err| {
                panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
            })
        }
    }
}

fn par_map_array<B, F>(mut arr: PyReadwriteArrayDyn<B>, f: F) -> PyResult<()>
where
    B: BitBuffer + numpy::Element,
    F: std::marker::Sync + Fn(&mut B),
{
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| f(item));

    Ok(())
}

#[pyclass(eq)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpScheme {
    D3P1,
    D7P1,
    D15P1,
}

#[pyfunction]
pub fn embedded_parity_encode_f32(
    arr: PyReadwriteArrayDyn<f32>,
    scheme: &EpScheme,
) -> PyResult<()> {
    par_map_array(
        arr,
        match scheme {
            EpScheme::D3P1 => |item: &mut f32| B32D3P1.encode(item),
            EpScheme::D7P1 => |item: &mut f32| B32D7P1.encode(item),
            EpScheme::D15P1 => |item: &mut f32| B32D15P1.encode(item),
        },
    )
}

#[pyfunction]
pub fn embedded_parity_decode_f32(
    arr: PyReadwriteArrayDyn<f32>,
    scheme: &EpScheme,
) -> PyResult<()> {
    let f = match scheme {
        EpScheme::D3P1 => |item: &mut f32| B32D3P1.decode(item),
        EpScheme::D7P1 => |item: &mut f32| B32D7P1.decode(item),
        EpScheme::D15P1 => |item: &mut f32| B32D15P1.decode(item),
    };
    par_map_array(arr, f)
}

#[pyfunction]
pub fn embedded_parity_encode_u16(
    arr: PyReadwriteArrayDyn<u16>,
    scheme: &EpScheme,
) -> PyResult<()> {
    par_map_array(
        arr,
        match scheme {
            EpScheme::D3P1 => |item: &mut u16| B16D3P1.encode(item),
            EpScheme::D7P1 => |item: &mut u16| B16D7P1.encode(item),
            EpScheme::D15P1 => |item: &mut u16| B16D15P1.encode(item),
        },
    )
}

#[pyfunction]
pub fn embedded_parity_decode_u16(
    arr: PyReadwriteArrayDyn<u16>,
    scheme: &EpScheme,
) -> PyResult<()> {
    let f = match scheme {
        EpScheme::D3P1 => |item: &mut u16| B16D3P1.decode(item),
        EpScheme::D7P1 => |item: &mut u16| B16D7P1.decode(item),
        EpScheme::D15P1 => |item: &mut u16| B16D15P1.decode(item),
    };
    par_map_array(arr, f)
}
