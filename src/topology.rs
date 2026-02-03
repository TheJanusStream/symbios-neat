//! Graph topology analysis using CSR format.
//!
//! This module provides efficient graph algorithms for NEAT genomes using
//! Compressed Sparse Row (CSR) format. CSR eliminates per-call Vec<Vec<usize>>
//! allocations that cause allocator pressure in evolutionary loops.
//!
//! ## Determinism
//!
//! Edges are sorted by innovation number before CSR construction to ensure
//! bit-identical floating-point results regardless of mutation history.
//! This matters because floating-point addition is not associative:
//! `(a + b) + c ≠ a + (b + c)` in general.

use crate::gene::{ConnectionId, NodeId};
use crate::genome::NeatGenome;
use std::collections::VecDeque;

/// CSR-format graph topology for efficient graph algorithms.
///
/// This struct holds a snapshot of the genome's enabled connections in a
/// cache-friendly format. It supports both forward traversal (for evaluation)
/// and reverse traversal (for cycle detection from a target node).
#[derive(Debug, Clone)]
pub struct GraphTopology {
    /// Number of nodes in the graph.
    node_count: usize,
    /// Maps NodeId to dense index (0..node_count).
    node_to_idx: Vec<(NodeId, usize)>,
    /// Maps dense index back to NodeId.
    idx_to_node: Vec<NodeId>,
    /// CSR offsets for forward edges (outgoing). Length = node_count + 1.
    fwd_offsets: Vec<usize>,
    /// CSR targets for forward edges. fwd_targets[fwd_offsets[i]..fwd_offsets[i+1]] are successors of node i.
    fwd_targets: Vec<usize>,
    /// CSR connection IDs for forward edges, parallel to fwd_targets.
    fwd_conn_ids: Vec<ConnectionId>,
    /// CSR offsets for reverse edges (incoming). Length = node_count + 1.
    rev_offsets: Vec<usize>,
    /// CSR sources for reverse edges. rev_sources[rev_offsets[i]..rev_offsets[i+1]] are predecessors of node i.
    rev_sources: Vec<usize>,
    /// CSR connection IDs for reverse edges, parallel to rev_sources.
    #[allow(dead_code)] // Reserved for future use (e.g., reverse edge lookup)
    rev_conn_ids: Vec<ConnectionId>,
}

impl GraphTopology {
    /// Build topology from a genome's enabled connections.
    ///
    /// Edges are sorted by innovation number for deterministic iteration order,
    /// ensuring bit-identical floating-point results across equivalent topologies.
    #[must_use]
    pub fn from_genome(genome: &NeatGenome) -> Self {
        // Collect node IDs in deterministic order (sorted by innovation).
        let mut node_entries: Vec<(NodeId, u64)> = genome
            .nodes
            .iter()
            .map(|(id, node)| (id, node.innovation))
            .collect();
        node_entries.sort_by_key(|(_, inn)| *inn);

        let node_count = node_entries.len();
        let idx_to_node: Vec<NodeId> = node_entries.iter().map(|(id, _)| *id).collect();

        // Build node_to_idx lookup as a sorted vec for binary search.
        // This avoids HashMap allocation while providing O(log n) lookup.
        let mut node_to_idx: Vec<(NodeId, usize)> = idx_to_node
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        // SlotMap keys implement Ord, so we can sort directly
        node_to_idx.sort_by_key(|(id, _)| *id);

        // Collect enabled connections sorted by innovation for determinism.
        let mut edges: Vec<(ConnectionId, NodeId, NodeId, u64)> = genome
            .connections
            .iter()
            .filter(|(_, c)| c.enabled)
            .map(|(id, c)| (id, c.input, c.output, c.innovation))
            .collect();
        edges.sort_by_key(|(_, _, _, inn)| *inn);

        // Build forward CSR (outgoing edges).
        let mut fwd_counts = vec![0usize; node_count];
        for &(_, from, _, _) in &edges {
            if let Some(idx) = lookup_idx(&node_to_idx, from) {
                fwd_counts[idx] += 1;
            }
        }

        let mut fwd_offsets = Vec::with_capacity(node_count + 1);
        fwd_offsets.push(0);
        for &count in &fwd_counts {
            fwd_offsets.push(fwd_offsets.last().unwrap() + count);
        }

        let total_edges = *fwd_offsets.last().unwrap();
        let mut fwd_targets = vec![0usize; total_edges];
        let mut fwd_conn_ids: Vec<ConnectionId> = vec![ConnectionId::default(); total_edges];
        let mut fwd_write_pos = fwd_offsets[..node_count].to_vec();

        for &(conn_id, from, to, _) in &edges {
            if let (Some(from_idx), Some(to_idx)) =
                (lookup_idx(&node_to_idx, from), lookup_idx(&node_to_idx, to))
            {
                let pos = fwd_write_pos[from_idx];
                fwd_targets[pos] = to_idx;
                fwd_conn_ids[pos] = conn_id;
                fwd_write_pos[from_idx] += 1;
            }
        }

        // Build reverse CSR (incoming edges).
        let mut rev_counts = vec![0usize; node_count];
        for &(_, _, to, _) in &edges {
            if let Some(idx) = lookup_idx(&node_to_idx, to) {
                rev_counts[idx] += 1;
            }
        }

        let mut rev_offsets = Vec::with_capacity(node_count + 1);
        rev_offsets.push(0);
        for &count in &rev_counts {
            rev_offsets.push(rev_offsets.last().unwrap() + count);
        }

        let mut rev_sources = vec![0usize; total_edges];
        let mut rev_conn_ids: Vec<ConnectionId> = vec![ConnectionId::default(); total_edges];
        let mut rev_write_pos = rev_offsets[..node_count].to_vec();

        for &(conn_id, from, to, _) in &edges {
            if let (Some(from_idx), Some(to_idx)) =
                (lookup_idx(&node_to_idx, from), lookup_idx(&node_to_idx, to))
            {
                let pos = rev_write_pos[to_idx];
                rev_sources[pos] = from_idx;
                rev_conn_ids[pos] = conn_id;
                rev_write_pos[to_idx] += 1;
            }
        }

        Self {
            node_count,
            node_to_idx,
            idx_to_node,
            fwd_offsets,
            fwd_targets,
            fwd_conn_ids,
            rev_offsets,
            rev_sources,
            rev_conn_ids,
        }
    }

    /// Get the dense index for a NodeId.
    #[inline]
    pub fn node_index(&self, id: NodeId) -> Option<usize> {
        lookup_idx(&self.node_to_idx, id)
    }

    /// Get the NodeId for a dense index.
    #[inline]
    pub fn node_id(&self, idx: usize) -> Option<NodeId> {
        self.idx_to_node.get(idx).copied()
    }

    /// Number of nodes in the topology.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Iterate over successors of a node (forward edges).
    #[inline]
    pub fn successors(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        let start = self.fwd_offsets[idx];
        let end = self.fwd_offsets[idx + 1];
        self.fwd_targets[start..end].iter().copied()
    }

    /// Iterate over predecessors of a node (reverse edges).
    #[inline]
    pub fn predecessors(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        let start = self.rev_offsets[idx];
        let end = self.rev_offsets[idx + 1];
        self.rev_sources[start..end].iter().copied()
    }

    /// Check if adding edge from `from_id` to `to_id` would create a cycle.
    ///
    /// Uses reverse BFS from `from_id` to check if `to_id` is reachable
    /// (i.e., if there's already a path from `to_id` to `from_id`).
    #[must_use]
    pub fn would_create_cycle(&self, from_id: NodeId, to_id: NodeId) -> bool {
        let (Some(from_idx), Some(to_idx)) = (self.node_index(from_id), self.node_index(to_id))
        else {
            return false;
        };

        // If from == to, it's a self-loop (cycle)
        if from_idx == to_idx {
            return true;
        }

        // BFS from to_idx following forward edges to see if we can reach from_idx.
        // If so, adding from->to would create a cycle.
        let mut visited = vec![false; self.node_count];
        let mut queue = VecDeque::new();
        queue.push_back(to_idx);
        visited[to_idx] = true;

        while let Some(current) = queue.pop_front() {
            for succ in self.successors(current) {
                if succ == from_idx {
                    return true;
                }
                if !visited[succ] {
                    visited[succ] = true;
                    queue.push_back(succ);
                }
            }
        }

        false
    }

    /// Detect if the graph contains any cycle using Kahn's algorithm.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut in_degree: Vec<usize> = vec![0; self.node_count];

        // Count in-degrees from reverse CSR
        for (idx, deg) in in_degree.iter_mut().enumerate() {
            *deg = self.rev_offsets[idx + 1] - self.rev_offsets[idx];
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(idx);
            }
        }

        let mut processed = 0;
        while let Some(u) = queue.pop_front() {
            processed += 1;
            for v in self.successors(u) {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        processed != self.node_count
    }

    /// Compute depths for all nodes using Kahn's algorithm (longest path).
    ///
    /// Returns a vector of depths indexed by dense node index, or None if cycle detected.
    #[must_use]
    pub fn compute_depths(&self) -> Option<Vec<u32>> {
        let mut in_degree: Vec<usize> = vec![0; self.node_count];
        let mut depths = vec![0u32; self.node_count];

        for (idx, deg) in in_degree.iter_mut().enumerate() {
            *deg = self.rev_offsets[idx + 1] - self.rev_offsets[idx];
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(idx);
            }
        }

        let mut processed = 0;
        while let Some(u) = queue.pop_front() {
            processed += 1;
            for v in self.successors(u) {
                let new_depth = depths[u].saturating_add(1);
                if new_depth > depths[v] {
                    depths[v] = new_depth;
                }

                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        if processed != self.node_count {
            None // Cycle detected
        } else {
            Some(depths)
        }
    }

    /// Find a back edge (cycle-causing edge) using iterative DFS.
    ///
    /// Returns the ConnectionId of the back edge with lowest absolute weight
    /// among all back edges found, to minimize signal disruption.
    pub fn find_back_edge(&self, genome: &NeatGenome) -> Option<ConnectionId> {
        // DFS coloring: 0=white, 1=gray (in path), 2=black (done)
        let mut color = vec![0u8; self.node_count];
        let mut back_edges: Vec<ConnectionId> = Vec::new();

        // Stack: (node_idx, edge_iter_offset, is_entering)
        let mut stack: Vec<(usize, usize, bool)> = Vec::with_capacity(self.node_count);

        for start in 0..self.node_count {
            if color[start] != 0 {
                continue;
            }

            stack.push((start, 0, true));

            while let Some((node, edge_offset, is_entering)) = stack.pop() {
                if is_entering {
                    color[node] = 1; // Gray
                }

                let start_pos = self.fwd_offsets[node];
                let end_pos = self.fwd_offsets[node + 1];
                let num_edges = end_pos - start_pos;

                let mut found_unvisited = false;
                for offset in edge_offset..num_edges {
                    let actual_idx = start_pos + offset;
                    let neighbor = self.fwd_targets[actual_idx];
                    let conn_id = self.fwd_conn_ids[actual_idx];

                    if color[neighbor] == 1 {
                        // Back edge found
                        back_edges.push(conn_id);
                    } else if color[neighbor] == 0 {
                        // Push current node to resume at next edge
                        stack.push((node, offset + 1, false));
                        // Push neighbor to visit
                        stack.push((neighbor, 0, true));
                        found_unvisited = true;
                        break;
                    }
                }

                if !found_unvisited {
                    color[node] = 2; // Black
                }
            }
        }

        if back_edges.is_empty() {
            return None;
        }

        // Choose edge with lowest absolute weight
        back_edges.into_iter().min_by(|&a, &b| {
            let weight_a = genome
                .connections
                .get(a)
                .map(|c| c.weight.abs())
                .unwrap_or(0.0);
            let weight_b = genome
                .connections
                .get(b)
                .map(|c| c.weight.abs())
                .unwrap_or(0.0);
            weight_a
                .partial_cmp(&weight_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get iterator over node indices in topological order.
    ///
    /// Returns None if the graph has cycles.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let mut in_degree: Vec<usize> = vec![0; self.node_count];
        for (idx, deg) in in_degree.iter_mut().enumerate() {
            *deg = self.rev_offsets[idx + 1] - self.rev_offsets[idx];
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(idx);
            }
        }

        let mut order = Vec::with_capacity(self.node_count);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for v in self.successors(u) {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        if order.len() != self.node_count {
            None
        } else {
            Some(order)
        }
    }

    /// Get CSR data for forward edges (for evaluator construction).
    ///
    /// Returns (offsets, sources, weights) where for node i, incoming edges
    /// are at indices [offsets[i]..offsets[i+1]].
    pub fn get_csr_for_evaluation(
        &self,
        genome: &NeatGenome,
        node_id_to_eval_idx: &std::collections::HashMap<NodeId, usize>,
    ) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let eval_node_count = node_id_to_eval_idx.len();

        // Count incoming edges per eval node
        let mut counts = vec![0usize; eval_node_count];

        // Collect edges sorted by innovation for determinism
        let mut edges: Vec<(usize, usize, f32, u64)> = Vec::new();
        for (_, conn) in &genome.connections {
            if !conn.enabled {
                continue;
            }
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_id_to_eval_idx.get(&conn.input),
                node_id_to_eval_idx.get(&conn.output),
            ) {
                edges.push((from_idx, to_idx, conn.weight, conn.innovation));
                counts[to_idx] += 1;
            }
        }

        // Sort by innovation for deterministic summation order
        edges.sort_by_key(|(_, _, _, inn)| *inn);

        // Build offsets
        let mut offsets = Vec::with_capacity(eval_node_count + 1);
        offsets.push(0);
        for &count in &counts {
            offsets.push(offsets.last().unwrap() + count);
        }

        // Allocate flat arrays
        let total = *offsets.last().unwrap();
        let mut sources = vec![0usize; total];
        let mut weights = vec![0.0f32; total];
        let mut write_pos = offsets[..eval_node_count].to_vec();

        for (from_idx, to_idx, weight, _) in edges {
            let pos = write_pos[to_idx];
            sources[pos] = from_idx;
            weights[pos] = weight;
            write_pos[to_idx] += 1;
        }

        (offsets, sources, weights)
    }
}

/// Binary search for NodeId in sorted vec.
fn lookup_idx(sorted: &[(NodeId, usize)], id: NodeId) -> Option<usize> {
    sorted
        .binary_search_by_key(&id, |(k, _)| *k)
        .ok()
        .map(|pos| sorted[pos].1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::NeatConfig;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn test_topology_basic() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let topo = GraphTopology::from_genome(&genome);

        // 2 inputs + 1 output = 3 nodes
        assert_eq!(topo.node_count(), 3);

        // Should be acyclic
        assert!(!topo.has_cycle());
    }

    #[test]
    fn test_topology_depths() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let topo = GraphTopology::from_genome(&genome);
        let depths = topo.compute_depths().expect("Should compute depths");

        // Inputs should have depth 0, output should have depth 1
        assert_eq!(depths.len(), 3);
    }

    #[test]
    fn test_would_create_cycle() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let topo = GraphTopology::from_genome(&genome);

        let input_id = genome.input_ids[0];
        let output_id = genome.output_ids[0];

        // Adding output -> input would create a cycle (since input -> output exists)
        assert!(topo.would_create_cycle(output_id, input_id));

        // Adding input -> output again wouldn't create cycle (already exists, but check is for new edge)
        // Actually this should not create a cycle since it's the same direction
        assert!(!topo.would_create_cycle(input_id, output_id));
    }

    #[test]
    fn test_topological_order() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let topo = GraphTopology::from_genome(&genome);
        let order = topo.topological_order().expect("Should have topo order");

        assert_eq!(order.len(), 3);
    }
}
