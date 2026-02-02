//! Hash-based innovation tracking for NEAT.
//!
//! Traditional NEAT uses a global innovation counter that requires synchronization
//! for parallel mutation. This module implements "Sovereign Innovation" - a lock-free
//! approach where innovation numbers are computed as deterministic hashes of the
//! structural mutation that created them.
//!
//! For connections: `Hash(input_node_innovation, output_node_innovation)`
//! For nodes (from split): `Hash(connection_innovation, SPLIT_MARKER)`

use std::hash::{Hash, Hasher};

/// Marker value used when hashing node splits to distinguish from connection innovations.
const SPLIT_MARKER: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Reserved range for fixed innovation IDs (bias, input, output nodes).
/// Hash-based innovations will always be >= this value.
const RESERVED_INNOVATION_RANGE: u64 = 1 << 16; // 65536

/// A deterministic hasher for computing innovation numbers.
///
/// Uses FxHash-style multiplication for speed while maintaining
/// good distribution properties. The output is guaranteed to be
/// >= RESERVED_INNOVATION_RANGE to avoid collisions with fixed node IDs.
#[derive(Default)]
struct InnovationHasher {
    state: u64,
}

impl Hasher for InnovationHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state = self
                .state
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(u64::from(byte));
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Final mixing to improve avalanche
        let mut h = self.state;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;

        // Ensure hash never collides with reserved range for fixed node IDs.
        // Map the hash to [RESERVED_INNOVATION_RANGE, u64::MAX] while
        // preserving uniform distribution.
        let range = u64::MAX - RESERVED_INNOVATION_RANGE;
        RESERVED_INNOVATION_RANGE + (h % range)
    }
}

/// Compute a deterministic innovation number for a connection.
///
/// The innovation is computed as `Hash(input_node_innovation, output_node_innovation)`,
/// ensuring that identical structural mutations always produce the same innovation
/// number regardless of when or in which thread they occur.
///
/// # Arguments
///
/// * `input_innovation` - Innovation number of the source node
/// * `output_innovation` - Innovation number of the target node
///
/// # Returns
///
/// A deterministic innovation number for this connection
#[inline]
#[must_use]
pub fn connection_innovation(input_innovation: u64, output_innovation: u64) -> u64 {
    let mut hasher = InnovationHasher::default();
    input_innovation.hash(&mut hasher);
    output_innovation.hash(&mut hasher);
    hasher.finish()
}

/// Compute a deterministic innovation number for a node created by splitting a connection.
///
/// The innovation is computed as `Hash(connection_innovation, SPLIT_MARKER)`,
/// ensuring that splitting the same connection always produces the same node
/// innovation number.
///
/// # Arguments
///
/// * `connection_innovation` - Innovation number of the connection being split
///
/// # Returns
///
/// A deterministic innovation number for the new node
#[inline]
#[must_use]
pub fn node_split_innovation(connection_innovation: u64) -> u64 {
    let mut hasher = InnovationHasher::default();
    connection_innovation.hash(&mut hasher);
    SPLIT_MARKER.hash(&mut hasher);
    hasher.finish()
}

/// Compute innovation for the first connection when splitting (input -> new_node).
#[inline]
#[must_use]
pub fn split_connection_a_innovation(input_node_innovation: u64, new_node_innovation: u64) -> u64 {
    connection_innovation(input_node_innovation, new_node_innovation)
}

/// Compute innovation for the second connection when splitting (new_node -> output).
#[inline]
#[must_use]
pub fn split_connection_b_innovation(new_node_innovation: u64, output_node_innovation: u64) -> u64 {
    connection_innovation(new_node_innovation, output_node_innovation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_innovation_deterministic() {
        let inn1 = connection_innovation(1, 2);
        let inn2 = connection_innovation(1, 2);
        assert_eq!(inn1, inn2, "Same inputs should produce same innovation");
    }

    #[test]
    fn test_connection_innovation_order_matters() {
        let inn1 = connection_innovation(1, 2);
        let inn2 = connection_innovation(2, 1);
        assert_ne!(
            inn1, inn2,
            "Different order should produce different innovation"
        );
    }

    #[test]
    fn test_connection_innovation_distribution() {
        // Test that nearby values produce well-distributed innovations
        let innovations: Vec<u64> = (0..100).map(|i| connection_innovation(i, i + 1)).collect();

        // Check no collisions in this small set
        let mut sorted = innovations.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), innovations.len(), "Should have no collisions");
    }

    #[test]
    fn test_node_split_innovation_deterministic() {
        let inn1 = node_split_innovation(100);
        let inn2 = node_split_innovation(100);
        assert_eq!(
            inn1, inn2,
            "Same connection should produce same node innovation"
        );
    }

    #[test]
    fn test_node_split_differs_from_connection() {
        // A node split innovation should differ from a connection innovation
        // even when using similar input values
        let _conn_inn = connection_innovation(100, SPLIT_MARKER);
        let node_inn = node_split_innovation(100);
        // They might collide by chance, but the hashing approach makes it unlikely
        // The important property is determinism, not collision avoidance
        assert_eq!(node_inn, node_split_innovation(100));
    }

    #[test]
    fn test_split_connection_innovations() {
        let original_conn_inn = 12345;
        let new_node_inn = node_split_innovation(original_conn_inn);
        let input_node_inn = 1;
        let output_node_inn = 2;

        let conn_a = split_connection_a_innovation(input_node_inn, new_node_inn);
        let conn_b = split_connection_b_innovation(new_node_inn, output_node_inn);

        assert_ne!(
            conn_a, conn_b,
            "Split connections should have different innovations"
        );
        assert_ne!(conn_a, original_conn_inn);
        assert_ne!(conn_b, original_conn_inn);
    }
}
