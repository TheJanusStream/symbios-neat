//! CPPN evaluator for NEAT genomes.
//!
//! This module provides the [`CppnEvaluator`] which executes a NEAT genome as a
//! Compositional Pattern Producing Network. CPPNs can be queried with spatial
//! coordinates to generate patterns, geometries, and weights for morphogenetic
//! engineering applications.

use crate::gene::{NodeId, NodeType};
use crate::genome::NeatGenome;
use crate::topology::GraphTopology;

/// A compiled, evaluation-ready representation of a NEAT genome.
///
/// The evaluator pre-computes topological order and organizes data for
/// cache-efficient forward propagation. Uses Compressed Sparse Row (CSR) format
/// for O(N+E) evaluation with optimal cache locality.
#[derive(Debug, Clone)]
pub struct CppnEvaluator {
    /// Node activations in topological order.
    activations: Vec<f32>,
    /// Node biases.
    biases: Vec<f32>,
    /// Activation function indices.
    activation_fns: Vec<crate::activation::Activation>,
    // CSR format for incoming connections - contiguous memory for cache locality.
    // For node i, incoming connections are at indices [csr_offsets[i]..csr_offsets[i+1]).
    /// CSR: source node indices for all connections (flat array).
    csr_sources: Vec<usize>,
    /// CSR: weights for all connections (flat array, parallel to csr_sources).
    csr_weights: Vec<f32>,
    /// CSR: offsets into csr_sources/csr_weights for each node (len = num_nodes + 1).
    csr_offsets: Vec<usize>,
    /// Indices of input nodes in the activations array.
    input_indices: Vec<usize>,
    /// Indices of output nodes in the activations array.
    output_indices: Vec<usize>,
    /// Index of bias node if present.
    bias_index: Option<usize>,
    /// Evaluation order (indices into activations, excluding inputs).
    eval_order: Vec<usize>,
}

/// Error type for evaluator construction failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluatorError {
    /// The genome contains cycles in its enabled connections.
    ///
    /// Feedforward evaluation requires an acyclic graph. Use `NeatGenome::break_cycles()`
    /// or `NeatGenome::has_cycle()` to detect and fix cycles before creating an evaluator.
    CyclicGenome,
}

impl std::fmt::Display for EvaluatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluatorError::CyclicGenome => {
                write!(
                    f,
                    "genome contains cycles; feedforward evaluation requires an acyclic graph"
                )
            }
        }
    }
}

impl std::error::Error for EvaluatorError {}

impl CppnEvaluator {
    /// Compile a NEAT genome into an efficient evaluator.
    ///
    /// This method ensures depth consistency by recomputing depths before
    /// building the evaluator. This guarantees correct evaluation order even
    /// if the genome was modified without calling `update_depths()`.
    ///
    /// # Panics
    ///
    /// Panics if the genome contains cycles. Use [`try_new`](Self::try_new) for
    /// non-panicking construction, or [`NeatGenome::break_cycles`] to fix the genome first.
    #[must_use]
    pub fn new(genome: &NeatGenome) -> Self {
        Self::try_new(genome).expect("genome contains cycles; use try_new() or break_cycles()")
    }

    /// Try to compile a NEAT genome into an efficient evaluator.
    ///
    /// Returns an error if the genome contains cycles, which would make feedforward
    /// evaluation mathematically undefined.
    ///
    /// Uses shared GraphTopology for O(V+E) depth computation with CSR format,
    /// avoiding duplicated Vec<Vec<usize>> allocations. Edges are sorted by
    /// innovation number for deterministic floating-point summation order.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluatorError::CyclicGenome`] if the genome contains cycles.
    pub fn try_new(genome: &NeatGenome) -> Result<Self, EvaluatorError> {
        // Use shared GraphTopology for depth computation - avoids code duplication
        // and ensures deterministic edge ordering by innovation number.
        let topo = GraphTopology::from_genome(genome);

        let depths = topo.compute_depths().ok_or(EvaluatorError::CyclicGenome)?;

        // Build NodeId -> depth mapping for sorting
        let depth_map: std::collections::HashMap<NodeId, u32> = (0..topo.node_count())
            .filter_map(|idx| topo.node_id(idx).map(|id| (id, depths[idx])))
            .collect();

        // Create mapping from NodeId to dense evaluator index
        let mut node_id_to_idx: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();

        // Sort nodes by computed depth for topological order
        let mut nodes: Vec<_> = genome.nodes.iter().collect();
        nodes.sort_by_key(|(id, _)| depth_map.get(id).copied().unwrap_or(0));

        let mut activations = Vec::with_capacity(nodes.len());
        let mut biases = Vec::with_capacity(nodes.len());
        let mut activation_fns = Vec::with_capacity(nodes.len());
        let mut bias_index = None;
        let mut eval_order = Vec::new();

        for (node_id, node) in &nodes {
            let idx = activations.len();
            node_id_to_idx.insert(*node_id, idx);

            activations.push(0.0);
            biases.push(node.bias);
            activation_fns.push(node.activation);

            match node.node_type {
                NodeType::Input => {} // Will be handled via genome.input_ids
                NodeType::Output => {
                    // Add to eval_order but not output_indices (handled via genome.output_ids)
                    eval_order.push(idx);
                }
                NodeType::Hidden => eval_order.push(idx),
                NodeType::Bias => bias_index = Some(idx),
            }
        }

        // Use genome input_ids to preserve semantic input ordering
        // This ensures Input 0 is always the first input, Input 1 is second, etc.
        // regardless of SlotMap iteration order after crossover/deserialization
        let input_indices: Vec<usize> = genome
            .input_ids
            .iter()
            .filter_map(|id| node_id_to_idx.get(id).copied())
            .collect();

        // Use genome output_ids to preserve semantic output ordering
        let output_indices: Vec<usize> = genome
            .output_ids
            .iter()
            .filter_map(|id| node_id_to_idx.get(id).copied())
            .collect();

        // Build CSR format with deterministic edge ordering (sorted by innovation).
        // This ensures bit-identical floating-point results across equivalent topologies.
        let (csr_offsets, csr_sources, csr_weights) =
            topo.get_csr_for_evaluation(genome, &node_id_to_idx);

        Ok(Self {
            activations,
            biases,
            activation_fns,
            csr_sources,
            csr_weights,
            csr_offsets,
            input_indices,
            output_indices,
            bias_index,
            eval_order,
        })
    }

    /// Evaluate the network with given inputs, writing results to a provided buffer.
    ///
    /// This is the allocation-free version for hot paths like CPPN pattern generation.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input values. Must match the number of input nodes.
    /// * `outputs` - Buffer to write output values. Must match the number of output nodes.
    ///
    /// # Panics
    ///
    /// Panics if input or output length doesn't match the network configuration.
    pub fn evaluate_into(&mut self, inputs: &[f32], outputs: &mut [f32]) {
        assert_eq!(
            inputs.len(),
            self.input_indices.len(),
            "Input length mismatch: expected {}, got {}",
            self.input_indices.len(),
            inputs.len()
        );
        assert_eq!(
            outputs.len(),
            self.output_indices.len(),
            "Output length mismatch: expected {}, got {}",
            self.output_indices.len(),
            outputs.len()
        );

        // Reset activations
        for act in &mut self.activations {
            *act = 0.0;
        }

        // Set input values
        for (i, &idx) in self.input_indices.iter().enumerate() {
            self.activations[idx] = inputs[i];
        }

        // Set bias value
        if let Some(bias_idx) = self.bias_index {
            self.activations[bias_idx] = 1.0;
        }

        // Forward propagation in topological order - O(N+E) using CSR format.
        // CSR enables contiguous memory access for optimal cache performance.
        for &node_idx in &self.eval_order {
            // Sum incoming connections from CSR arrays
            let mut sum = self.biases[node_idx];
            let start = self.csr_offsets[node_idx];
            let end = self.csr_offsets[node_idx + 1];
            for i in start..end {
                let from_idx = self.csr_sources[i];
                let weight = self.csr_weights[i];
                sum += self.activations[from_idx] * weight;
            }

            // Apply activation function
            self.activations[node_idx] = self.activation_fns[node_idx].apply(sum);
        }

        // Write outputs to buffer
        for (i, &idx) in self.output_indices.iter().enumerate() {
            outputs[i] = self.activations[idx];
        }
    }

    /// Evaluate the network with given inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input values. Must match the number of input nodes.
    ///
    /// # Returns
    ///
    /// Output values from the network.
    ///
    /// # Panics
    ///
    /// Panics if input length doesn't match the number of input nodes.
    pub fn evaluate(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = vec![0.0; self.output_indices.len()];
        self.evaluate_into(inputs, &mut outputs);
        outputs
    }

    /// Query the CPPN with 2D coordinates.
    ///
    /// Convenience method for 2D pattern generation.
    /// Inputs are: [x, y]
    #[inline]
    pub fn query_2d(&mut self, x: f32, y: f32) -> Vec<f32> {
        self.evaluate(&[x, y])
    }

    /// Query the CPPN with 3D coordinates.
    ///
    /// Convenience method for 3D geometry generation.
    /// Inputs are: [x, y, z]
    #[inline]
    pub fn query_3d(&mut self, x: f32, y: f32, z: f32) -> Vec<f32> {
        self.evaluate(&[x, y, z])
    }

    /// Query the CPPN with 2D coordinates plus distance from center.
    ///
    /// Useful for radial patterns. Inputs are: [x, y, d] where d = sqrt(x² + y²)
    #[inline]
    pub fn query_2d_with_distance(&mut self, x: f32, y: f32) -> Vec<f32> {
        let d = x.hypot(y);
        self.evaluate(&[x, y, d])
    }

    /// Query the CPPN for substrate weight generation.
    ///
    /// Used in HyperNEAT-style indirect encoding.
    /// Inputs are: [x1, y1, x2, y2] (source and target coordinates)
    #[inline]
    pub fn query_substrate(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<f32> {
        self.evaluate(&[x1, y1, x2, y2])
    }

    /// Get the number of input nodes.
    #[must_use]
    pub const fn num_inputs(&self) -> usize {
        self.input_indices.len()
    }

    /// Get the number of output nodes.
    #[must_use]
    pub const fn num_outputs(&self) -> usize {
        self.output_indices.len()
    }

    /// Get the activation function for a specific output node.
    ///
    /// # Panics
    ///
    /// Panics if `output_index` is out of bounds.
    #[must_use]
    pub fn output_activation(&self, output_index: usize) -> crate::activation::Activation {
        self.activation_fns[self.output_indices[output_index]]
    }
}

/// Error type for pattern generation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternError {
    /// The requested output index exceeds the number of network outputs.
    OutputIndexOutOfBounds {
        /// The requested index.
        requested: usize,
        /// The actual number of outputs.
        available: usize,
    },
}

impl std::fmt::Display for PatternError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternError::OutputIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "output_index {} out of bounds for network with {} outputs",
                requested, available
            ),
        }
    }
}

impl std::error::Error for PatternError {}

/// Generate a 2D pattern image from a CPPN.
///
/// # Arguments
///
/// * `evaluator` - The CPPN evaluator (must have 2+ inputs and 1+ outputs)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `output_index` - Which output to use (0 for first output)
///
/// # Returns
///
/// A flattened grayscale image as `f32` values in `[0, 1]`, or an error if
/// the output index is out of bounds.
///
/// # Errors
///
/// Returns [`PatternError::OutputIndexOutOfBounds`] if `output_index` >= number of outputs.
#[allow(clippy::cast_precision_loss)] // Image dimensions are small enough
pub fn generate_pattern(
    evaluator: &mut CppnEvaluator,
    width: usize,
    height: usize,
    output_index: usize,
) -> Result<Vec<f32>, PatternError> {
    // Validate output_index upfront to fail fast
    if output_index >= evaluator.num_outputs() {
        return Err(PatternError::OutputIndexOutOfBounds {
            requested: output_index,
            available: evaluator.num_outputs(),
        });
    }

    let mut pattern = Vec::with_capacity(width * height);

    // For mapping pixel indices to [-1, 1] range:
    // - Single pixel (dim=1): center at 0.0
    // - Multiple pixels: span from -1.0 to +1.0 inclusive
    let x_divisor = if width > 1 { (width - 1) as f32 } else { 1.0 };
    let y_divisor = if height > 1 { (height - 1) as f32 } else { 1.0 };

    // Pre-allocate buffers for allocation-free inner loop
    let mut inputs = [0.0f32; 2];
    let mut outputs = vec![0.0f32; evaluator.num_outputs()];

    for y in 0..height {
        for x in 0..width {
            // Normalize coordinates to [-1, 1]
            // For dim=1: coordinate is 0.0 (centered)
            // For dim>1: maps [0, dim-1] to [-1, 1]
            inputs[0] = if width == 1 {
                0.0
            } else {
                (x as f32 / x_divisor).mul_add(2.0, -1.0)
            };
            inputs[1] = if height == 1 {
                0.0
            } else {
                (y as f32 / y_divisor).mul_add(2.0, -1.0)
            };

            evaluator.evaluate_into(&inputs, &mut outputs);
            // SAFETY: We validated output_index at function entry
            let value = outputs[output_index];

            // Normalize output to [0, 1] based on activation function's range.
            // This correctly handles all activations (ReLU, Identity, etc.),
            // not just bounded ones like Tanh/Sigmoid.
            let (min_val, max_val) = evaluator.output_activation(output_index).output_range();
            let range = max_val - min_val;
            let normalized = if range > 0.0 {
                (value - min_val) / range
            } else {
                0.5 // Degenerate case: single-value range
            };
            pattern.push(normalized.clamp(0.0, 1.0));
        }
    }

    Ok(pattern)
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
    fn test_evaluator_basic() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);

        assert_eq!(evaluator.num_inputs(), 2);
        assert_eq!(evaluator.num_outputs(), 1);

        let outputs = evaluator.evaluate(&[0.5, 0.5]);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_evaluator_deterministic() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);

        let outputs1 = evaluator.evaluate(&[0.5, -0.5]);
        let outputs2 = evaluator.evaluate(&[0.5, -0.5]);

        assert!(
            (outputs1[0] - outputs2[0]).abs() < 1e-6,
            "Evaluation should be deterministic"
        );
    }

    #[test]
    fn test_evaluator_with_hidden_node() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let mut genome = NeatGenome::fully_connected(config, &mut rng);

        // Add a hidden node
        let conn_id = genome.connections.iter().next().unwrap().0;
        genome.add_node(conn_id, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);
        let outputs = evaluator.evaluate(&[1.0, 0.0]);

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finite());
    }

    #[test]
    fn test_query_methods() {
        let config = NeatConfig::cppn(3, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);

        let out_3d = evaluator.query_3d(0.0, 0.0, 0.0);
        assert_eq!(out_3d.len(), 1);
    }

    #[test]
    fn test_generate_pattern() {
        let config = NeatConfig::cppn(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);
        let pattern = generate_pattern(&mut evaluator, 8, 8, 0).unwrap();

        assert_eq!(pattern.len(), 64);
        for &val in &pattern {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    #[should_panic(expected = "Input length mismatch")]
    fn test_evaluator_input_mismatch() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);
        evaluator.evaluate(&[1.0]); // Wrong number of inputs
    }

    #[test]
    fn test_try_new_returns_ok_for_acyclic_genome() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let result = CppnEvaluator::try_new(&genome);
        assert!(result.is_ok(), "try_new should succeed for acyclic genome");
    }

    #[test]
    fn test_evaluator_error_display() {
        let err = EvaluatorError::CyclicGenome;
        let msg = err.to_string();
        assert!(
            msg.contains("cycle"),
            "Error message should mention cycles: {}",
            msg
        );
    }
}
