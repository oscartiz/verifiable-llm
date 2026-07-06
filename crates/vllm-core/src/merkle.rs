//! Binary Merkle tree over 32-byte leaves, RFC 6962 tree shape (split at the
//! largest power of two < n), BLAKE3 with domain-separated leaf/node hashing.
//! Proof generation is O(n log n) hashes — fine for the tree sizes we commit
//! to (hundreds of tensors, tens of thousands of trace cells).

use crate::Hash32;

const LEAF_DOMAIN: &[u8] = b"vllm/merkle-leaf/v1";
const NODE_DOMAIN: &[u8] = b"vllm/merkle-node/v1";

fn leaf_node(leaf: &Hash32) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(LEAF_DOMAIN);
    h.update(&leaf.0);
    h.finalize().into()
}

fn inner_node(left: &Hash32, right: &Hash32) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(NODE_DOMAIN);
    h.update(&left.0);
    h.update(&right.0);
    h.finalize().into()
}

/// Largest power of two strictly less than n (n >= 2).
fn split_point(n: usize) -> usize {
    debug_assert!(n >= 2);
    1 << (usize::BITS - 1 - (n - 1).leading_zeros())
}

/// Merkle root of a non-empty leaf slice. Returns None for empty input.
pub fn root(leaves: &[Hash32]) -> Option<Hash32> {
    match leaves {
        [] => None,
        [one] => Some(leaf_node(one)),
        _ => {
            let k = split_point(leaves.len());
            Some(inner_node(&root(&leaves[..k])?, &root(&leaves[k..])?))
        }
    }
}

/// Inclusion proof for `leaves[index]`, ordered bottom-up (sibling closest to
/// the leaf first). Returns None if index is out of range.
pub fn prove(leaves: &[Hash32], index: usize) -> Option<Vec<Hash32>> {
    if index >= leaves.len() {
        return None;
    }
    if leaves.len() == 1 {
        return Some(Vec::new());
    }
    let k = split_point(leaves.len());
    let mut path = if index < k {
        let mut p = prove(&leaves[..k], index)?;
        p.push(root(&leaves[k..])?);
        p
    } else {
        let mut p = prove(&leaves[k..], index - k)?;
        p.push(root(&leaves[..k])?);
        p
    };
    path.shrink_to_fit();
    Some(path)
}

/// Check that `leaf` sits at `index` in a tree of `n_leaves` with the given
/// root, using a bottom-up `path` as produced by [`prove`].
pub fn verify(
    leaf: &Hash32,
    index: usize,
    n_leaves: usize,
    path: &[Hash32],
    expected_root: &Hash32,
) -> bool {
    match root_from_path(leaf, index, n_leaves, path) {
        Some(r) => r == *expected_root,
        None => false,
    }
}

fn root_from_path(leaf: &Hash32, index: usize, n: usize, path: &[Hash32]) -> Option<Hash32> {
    if index >= n {
        return None;
    }
    if n == 1 {
        return path.is_empty().then(|| leaf_node(leaf));
    }
    let (&sibling, rest) = path.split_last()?;
    let k = split_point(n);
    if index < k {
        Some(inner_node(&root_from_path(leaf, index, k, rest)?, &sibling))
    } else {
        Some(inner_node(
            &sibling,
            &root_from_path(leaf, index - k, n - k, rest)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Hash32> {
        (0..n)
            .map(|i| {
                let mut h = blake3::Hasher::new();
                h.update(b"test-leaf");
                h.update(&(i as u64).to_le_bytes());
                h.finalize().into()
            })
            .collect()
    }

    #[test]
    fn empty_tree_has_no_root() {
        assert!(root(&[]).is_none());
        assert!(prove(&[], 0).is_none());
    }

    #[test]
    fn single_leaf_root_is_domain_tagged() {
        let l = leaves(1);
        let mut h = blake3::Hasher::new();
        h.update(LEAF_DOMAIN);
        h.update(&l[0].0);
        assert_eq!(root(&l).unwrap(), Hash32::from(h.finalize()));
    }

    #[test]
    fn proofs_verify_for_all_indices_and_sizes() {
        for n in 1..=17 {
            let l = leaves(n);
            let r = root(&l).unwrap();
            for i in 0..n {
                let path = prove(&l, i).unwrap();
                assert!(verify(&l[i], i, n, &path, &r), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn tampering_is_detected() {
        let l = leaves(7);
        let r = root(&l).unwrap();
        let path = prove(&l, 3).unwrap();
        // Wrong leaf, wrong index, truncated path, mutated path element.
        assert!(!verify(&l[4], 3, 7, &path, &r));
        assert!(!verify(&l[3], 4, 7, &path, &r));
        assert!(!verify(&l[3], 3, 7, &path[..path.len() - 1], &r));
        let mut bad = path.clone();
        bad[0].0[0] ^= 1;
        assert!(!verify(&l[3], 3, 7, &bad, &r));
        // Index out of range must not panic.
        assert!(!verify(&l[3], 9, 7, &path, &r));
    }

    #[test]
    fn root_depends_on_every_leaf_and_on_order() {
        let l = leaves(6);
        let r = root(&l).unwrap();
        for i in 0..6 {
            let mut m = l.clone();
            m[i].0[31] ^= 1;
            assert_ne!(root(&m).unwrap(), r, "leaf {i} did not affect root");
        }
        let mut swapped = l.clone();
        swapped.swap(1, 2);
        assert_ne!(root(&swapped).unwrap(), r);
    }

    /// Golden vector: pins the exact tree construction (domains, shape).
    /// If this changes, every existing commitment breaks — bump the domain
    /// version strings instead of silently changing the scheme.
    #[test]
    fn golden_root_is_stable() {
        let r = root(&leaves(5)).unwrap();
        assert_eq!(
            r.to_hex(),
            golden_expected(),
            "Merkle construction changed; this breaks existing commitments"
        );
    }

    fn golden_expected() -> String {
        // Computed once with blake3 1.8; must never change for v1 domains.
        "3191e548cf1c7a39ed371b50e6e724a97fbe46154140ef47e56db987fd164417".into()
    }
}
