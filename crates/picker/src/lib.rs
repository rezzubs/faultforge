use rand::{Rng, RngExt};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("returned value {returned_value} is not in the range 0..{size}")]
pub struct FromReturnedError {
    returned_value: usize,
    size: usize,
}

/// An iterator which returns numbers from 0..n in a random order until all values are consumed.
///
/// Every returned value is unique.
///
/// This is based on the [Fisher-Yates
/// shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) but instead of shuffling
/// the whole sequence we just return the target index for the swap.
pub struct Picker<R> {
    /// Keeps track of the initial size, used in [`Self::reset`].
    initial_size: usize,
    /// Elements from `0..size` will be returned.
    size: usize,
    /// This map is used to keep track of the "current" values at each index. If there's no value
    /// at an index it's assumed to be the same as the index. Each iteration a random index from
    /// 0..size is picked and returned. The returned index will be swapped with `size` and size is
    /// decremented by 1.
    current_values: HashMap<usize, usize>,
    /// The random number generator used for picking indices.
    rng: R,
}

impl<R> Picker<R>
where
    R: rand::Rng,
{
    /// Create a new picker.
    ///
    /// The picker will return elements from `0..size`.
    pub fn new(size: usize, rng: R) -> Self {
        Self {
            initial_size: size,
            size,
            current_values: HashMap::new(),
            rng,
        }
    }

    /// Reset the picker to its initial state.
    pub fn reset(&mut self) {
        self.size = self.initial_size;
        self.current_values.clear();
    }

    /// Return the initial size of the picker.
    ///
    /// This is the size passed to [`Self::new`] or [`Self::from_returned`].
    pub fn initial_size(&self) -> usize {
        self.initial_size
    }

    /// Return the current size of the picker - the number of remaining values.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Reconstruct a picker that will not return any of the `already_returned`
    /// values.
    ///
    /// This is useful when you need to suspend and resume an operation that
    /// uses a picker, and you only have the set of previously returned values
    /// available (not the full internal state). The internal state will differ
    /// from the original picker, but the remaining values returned will be a
    /// valid permutation of `0..initial_size` excluding `already_returned`.
    ///
    /// # How it works
    ///
    /// After `k` values have been returned from a size-`n` picker, the internal
    /// `current_values` map is a bijection from indices `0..n-k` to the
    /// remaining values. Most indices map to themselves (identity), so only
    /// non-identity entries need to be stored.
    ///
    /// To reconstruct this map from `already_returned`:
    ///
    /// 1. Compute `new_size = n - k`.
    /// 2. Find the **holes**: indices in `0..new_size` that were already
    ///    returned — these slots need an explicit entry because their natural
    ///    value has been consumed.
    /// 3. Find the **tails**: remaining values `>= new_size` — these are the
    ///    values that can't sit at their natural index (which is now out of range)
    ///    and must fill the holes.
    /// 4. Pair each hole with a tail value. All other indices map to themselves
    ///    implicitly.
    ///
    /// `|holes| == |tails|` always holds, and both are bounded by `min(k,
    /// n-k)`, so the reconstructed map is no larger than the original would
    /// have been.
    pub fn from_returned(
        initial_size: usize,
        already_returned: &HashSet<usize>,
        rng: R,
    ) -> Result<Self, FromReturnedError> {
        for &v in already_returned {
            if v >= initial_size {
                return Err(FromReturnedError {
                    returned_value: v,
                    size: initial_size,
                });
            }
        }

        let new_size = initial_size - already_returned.len();

        // Indices in 0..new_size that were already returned — these are "holes".
        let mut holes: Vec<usize> = already_returned
            .iter()
            .copied()
            .filter(|&v| v < new_size)
            .collect();
        holes.sort_unstable();

        // Remaining values >= new_size that must fill the holes.
        // The range iterator is ascending so tails is already sorted.
        let tails: Vec<usize> = (new_size..initial_size)
            .filter(|v| !already_returned.contains(v))
            .collect();

        let current_values: HashMap<usize, usize> = holes.into_iter().zip(tails).collect();

        Ok(Self {
            initial_size,
            size: new_size,
            current_values,
            rng,
        })
    }
}

impl<R> Iterator for Picker<R>
where
    R: Rng,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.size == 0 {
            return None;
        }

        let target_element_index_initial = self.rng.random_range(0..self.size);

        let target_element_index_actual = self
            .current_values
            .get(&target_element_index_initial)
            .copied()
            .unwrap_or(target_element_index_initial);

        self.size -= 1;

        if target_element_index_actual == self.size {
            // There is no need to do swap the last element with itself. This
            // element cannot be picked again anyway.
            return Some(target_element_index_actual);
        }

        let last_element_index_initial = self.size;
        let last_element_index_actual = self
            .current_values
            .get(&last_element_index_initial)
            .copied()
            .unwrap_or(last_element_index_initial);
        self.current_values
            .insert(target_element_index_initial, last_element_index_actual);

        Some(target_element_index_actual)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<R> ExactSizeIterator for Picker<R> where R: Rng {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_picker(size in 0usize..1024, seed in 0u64..1024) {
            let picker = Picker::new(size, rand::rngs::SmallRng::seed_from_u64(seed));
            let mut values: Vec<usize> = picker.into_iter().collect();
            values.sort();
            assert_eq!(values, (0..size).collect::<Vec<_>>());
        }

        #[test]
        fn test_picker_from_returned(
            (size, num_to_pick) in (1usize..1024).prop_flat_map(|size| {
            (Just(size), 0..=size)
        }), seed in 0u64..1024) {
            let mut picker = Picker::new(size, rand::rngs::SmallRng::seed_from_u64(seed));

            let already_returned: HashSet<usize> = (0..num_to_pick)
                .map(|_| picker.next().unwrap())
                .collect();

            let picker2 = Picker::from_returned(
                size,
                &already_returned,
                rand::rngs::SmallRng::seed_from_u64(seed),
            ).unwrap();

            let remaining: HashSet<usize> = picker2.collect();

            // Must not contain any already-returned value
            assert!(remaining.is_disjoint(&already_returned));

            // Together they must cover 0..size exactly
            let all: HashSet<usize> = remaining.union(&already_returned).copied().collect();
            assert_eq!(all, (0..size).collect::<HashSet<_>>());
        }
    }
}
