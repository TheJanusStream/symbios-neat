//! CPPN evaluator for NEAT genomes.
//!
//! This module provides the [`CppnEvaluator`] which executes a NEAT genome as a
//! Compositional Pattern Producing Network. CPPNs can be queried with spatial
//! coordinates to generate patterns, geometries, and weights for morphogenetic
//! engineering applications.

use crate::gene::{NodeId, NodeType};
use crate::genome::NeatGenome;
use crate::network::{FeedforwardNetwork, Scratchpad};
use crate::topology::GraphTopology;

/// Reusable scratchpad for CPPN evaluation.
///
/// This struct holds the mutable activation state during forward propagation,
/// allowing a single immutable `CppnEvaluator` to be shared across multiple
/// threads for parallel evaluation.
///
/// # Thread Safety
///
/// Each thread should have its own `EvalScratchpad`. The `CppnEvaluator` itself
/// is immutable during evaluation, enabling safe parallel pattern generation.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use rayon::prelude::*;
///
/// let evaluator = Arc::new(CppnEvaluator::new(&genome).unwrap());
///
/// let results: Vec<f32> = coordinates
///     .par_iter()
///     .map_init(
///         || evaluator.create_scratchpad(),
///         |scratch, (x, y)| {
///             let mut outputs = [0.0];
///             evaluator.evaluate_into(&[*x, *y], &mut outputs, scratch);
///             outputs[0]
///         }
///     )
///     .collect();
/// ```
pub type EvalScratchpad = Scratchpad;

/// A compiled, evaluation-ready representation of a NEAT genome.
///
/// Wraps a [`FeedforwardNetwork`] with CPPN-specific query helpers
/// ([`query_2d`](Self::query_2d), [`query_3d`](Self::query_3d), etc.).
///
/// # Thread Safety
///
/// The evaluator is immutable during evaluation. Mutable state (activations) is
/// stored in a separate [`EvalScratchpad`], allowing a single evaluator to be
/// shared across threads via `Arc<CppnEvaluator>`.
#[derive(Debug, Clone)]
pub struct CppnEvaluator {
    network: FeedforwardNetwork,
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
    /// Returns an error if the genome contains cycles, which would make feedforward
    /// evaluation mathematically undefined. Callers with cyclic genomes should
    /// use [`NeatGenome::break_cycles`] before constructing the evaluator.
    ///
    /// Uses shared GraphTopology for O(V+E) depth computation with CSR format,
    /// avoiding duplicated `Vec<Vec<usize>>` allocations. Edges are sorted by
    /// innovation number for deterministic floating-point summation order.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluatorError::CyclicGenome`] if the genome contains cycles.
    pub fn new(genome: &NeatGenome) -> Result<Self, EvaluatorError> {
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

        let mut biases = Vec::with_capacity(nodes.len());
        let mut activation_fns = Vec::with_capacity(nodes.len());
        let mut bias_index = None;
        let mut eval_order = Vec::new();
        let mut node_count = 0;

        for (node_id, node) in &nodes {
            let idx = node_count;
            node_id_to_idx.insert(*node_id, idx);

            biases.push(node.bias);
            activation_fns.push(node.activation);
            node_count += 1;

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
            network: FeedforwardNetwork {
                num_nodes: node_count,
                biases,
                activations: activation_fns,
                csr_sources,
                csr_weights,
                csr_offsets,
                input_indices,
                output_indices,
                bias_index,
                eval_order,
            },
        })
    }

    /// Borrow the underlying [`FeedforwardNetwork`].
    ///
    /// Useful for passing the evaluation core to code that doesn't need
    /// CPPN-specific helpers (e.g., HyperNEAT substrate consumers).
    #[must_use]
    pub fn network(&self) -> &FeedforwardNetwork {
        &self.network
    }

    /// Allocate a fresh scratchpad sized for this evaluator's network.
    #[must_use]
    pub fn create_scratchpad(&self) -> EvalScratchpad {
        self.network.create_scratchpad()
    }

    /// Evaluate the network with given inputs into a caller-supplied buffer.
    ///
    /// # Panics
    ///
    /// Panics if input or output length doesn't match the network configuration.
    pub fn evaluate_into(&self, inputs: &[f32], outputs: &mut [f32], scratch: &mut EvalScratchpad) {
        self.network.evaluate_into(inputs, outputs, scratch);
    }

    /// Evaluate the network with given inputs.
    ///
    /// For repeated evaluation prefer
    /// [`evaluate_into`](Self::evaluate_into) with a reusable scratchpad.
    ///
    /// # Panics
    ///
    /// Panics if input length doesn't match the number of input nodes.
    #[must_use]
    pub fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        self.network.evaluate(inputs)
    }

    /// Query the CPPN with 2D coordinates.
    ///
    /// Convenience method for 2D pattern generation.
    /// Inputs are: [x, y]
    #[inline]
    pub fn query_2d(&self, x: f32, y: f32) -> Vec<f32> {
        self.evaluate(&[x, y])
    }

    /// Query the CPPN with 3D coordinates.
    ///
    /// Convenience method for 3D geometry generation.
    /// Inputs are: [x, y, z]
    #[inline]
    pub fn query_3d(&self, x: f32, y: f32, z: f32) -> Vec<f32> {
        self.evaluate(&[x, y, z])
    }

    /// Query the CPPN with 2D coordinates plus distance from center.
    ///
    /// Useful for radial patterns. Inputs are: [x, y, d] where d = sqrt(x² + y²)
    #[inline]
    pub fn query_2d_with_distance(&self, x: f32, y: f32) -> Vec<f32> {
        let d = x.hypot(y);
        self.evaluate(&[x, y, d])
    }

    /// Query the CPPN for substrate weight generation.
    ///
    /// Used in HyperNEAT-style indirect encoding.
    /// Inputs are: [x1, y1, x2, y2] (source and target coordinates)
    #[inline]
    pub fn query_substrate(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<f32> {
        self.evaluate(&[x1, y1, x2, y2])
    }

    /// Get the number of input nodes.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.network.num_inputs()
    }

    /// Get the number of output nodes.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.network.num_outputs()
    }

    /// Get the activation function for a specific output node.
    ///
    /// # Panics
    ///
    /// Panics if `output_index` is out of bounds.
    #[must_use]
    pub fn output_activation(&self, output_index: usize) -> crate::activation::Activation {
        self.network.output_activation(output_index)
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
    /// The CPPN's input arity does not match the pattern dimensionality
    /// (e.g. calling `generate_pattern_2d` on a CPPN that does not accept 2 inputs).
    InputArityMismatch {
        /// The number of inputs expected for this pattern dimensionality.
        expected: usize,
        /// The number of inputs the CPPN actually has.
        actual: usize,
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
            PatternError::InputArityMismatch { expected, actual } => write!(
                f,
                "CPPN must accept {} inputs for this pattern, got {}",
                expected, actual
            ),
        }
    }
}

impl std::error::Error for PatternError {}

impl CppnEvaluator {
    /// Generate a 2D grayscale pattern by querying the CPPN over a grid of
    /// `width × height` pixels in `[-1, 1]²`.
    ///
    /// Returns a flat row-major buffer of `f32` values in `[0, 1]` (length
    /// `width * height`, x-major within each row, y outermost).
    ///
    /// Output is normalized using the activation function's range, so the
    /// result is meaningful for any activation (Tanh, Sigmoid, ReLU, etc.).
    ///
    /// # Errors
    ///
    /// - [`PatternError::InputArityMismatch`] if the CPPN does not accept 2 inputs.
    /// - [`PatternError::OutputIndexOutOfBounds`] if `output_index` is too large.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_pattern_2d(
        &self,
        width: u32,
        height: u32,
        output_index: usize,
    ) -> Result<Vec<f32>, PatternError> {
        validate_inputs(self, 2)?;
        validate_output_index(self, output_index)?;

        let w = width as usize;
        let h = height as usize;
        let mut pattern = Vec::with_capacity(w * h);
        let mut scratch = self.create_scratchpad();
        let mut inputs = [0.0f32; 2];
        let mut outputs = vec![0.0f32; self.num_outputs()];

        let (min_val, max_val) = self.output_activation(output_index).output_range();
        let range = max_val - min_val;

        for y in 0..h {
            for x in 0..w {
                inputs[0] = grid_coord(x, w);
                inputs[1] = grid_coord(y, h);
                self.evaluate_into(&inputs, &mut outputs, &mut scratch);
                pattern.push(normalize(outputs[output_index], min_val, range));
            }
        }
        Ok(pattern)
    }

    /// Generate a 3D voxel grid by querying the CPPN over `dims = [w, h, d]`
    /// points in `[-1, 1]³`.
    ///
    /// Returns a flat row-major buffer of length `w * h * d`, with axis order
    /// **x fastest, then y, then z slowest** — i.e. index `[x, y, z]` lives at
    /// offset `z * (h * w) + y * w + x`.
    ///
    /// # Errors
    ///
    /// - [`PatternError::InputArityMismatch`] if the CPPN does not accept 3 inputs.
    /// - [`PatternError::OutputIndexOutOfBounds`] if `output_index` is too large.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_voxel_grid(
        &self,
        dims: [u32; 3],
        output_index: usize,
    ) -> Result<Vec<f32>, PatternError> {
        validate_inputs(self, 3)?;
        validate_output_index(self, output_index)?;

        let w = dims[0] as usize;
        let h = dims[1] as usize;
        let d = dims[2] as usize;
        let mut grid = Vec::with_capacity(w * h * d);
        let mut scratch = self.create_scratchpad();
        let mut inputs = [0.0f32; 3];
        let mut outputs = vec![0.0f32; self.num_outputs()];

        let (min_val, max_val) = self.output_activation(output_index).output_range();
        let range = max_val - min_val;

        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    inputs[0] = grid_coord(x, w);
                    inputs[1] = grid_coord(y, h);
                    inputs[2] = grid_coord(z, d);
                    self.evaluate_into(&inputs, &mut outputs, &mut scratch);
                    grid.push(normalize(outputs[output_index], min_val, range));
                }
            }
        }
        Ok(grid)
    }

    /// Generate a 2D RGBA image (grayscale → RGBA) from the CPPN.
    ///
    /// Available behind the `image` Cargo feature. Each pixel's grayscale
    /// value `g ∈ [0, 1]` is encoded as `(g, g, g, 1)` after scaling to `u8`.
    ///
    /// # Errors
    ///
    /// Same as [`generate_pattern_2d`](Self::generate_pattern_2d).
    #[cfg(feature = "image")]
    pub fn generate_image(
        &self,
        width: u32,
        height: u32,
        output_index: usize,
    ) -> Result<image::RgbaImage, PatternError> {
        let pattern = self.generate_pattern_2d(width, height, output_index)?;
        let mut img = image::RgbaImage::new(width, height);
        for (i, pixel) in img.pixels_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let v = (pattern[i].clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgba([v, v, v, 255]);
        }
        Ok(img)
    }
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn grid_coord(idx: usize, len: usize) -> f32 {
    if len <= 1 {
        0.0
    } else {
        let t = idx as f32 / (len - 1) as f32;
        t.mul_add(2.0, -1.0)
    }
}

#[inline]
fn normalize(value: f32, min_val: f32, range: f32) -> f32 {
    if range > 0.0 {
        ((value - min_val) / range).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn validate_output_index(eval: &CppnEvaluator, output_index: usize) -> Result<(), PatternError> {
    if output_index >= eval.num_outputs() {
        return Err(PatternError::OutputIndexOutOfBounds {
            requested: output_index,
            available: eval.num_outputs(),
        });
    }
    Ok(())
}

fn validate_inputs(eval: &CppnEvaluator, expected: usize) -> Result<(), PatternError> {
    if eval.num_inputs() != expected {
        return Err(PatternError::InputArityMismatch {
            expected,
            actual: eval.num_inputs(),
        });
    }
    Ok(())
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

        let evaluator = CppnEvaluator::new(&genome).unwrap();

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

        let evaluator = CppnEvaluator::new(&genome).unwrap();

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

        let evaluator = CppnEvaluator::new(&genome).unwrap();
        let outputs = evaluator.evaluate(&[1.0, 0.0]);

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finite());
    }

    #[test]
    fn test_query_methods() {
        let config = NeatConfig::cppn(3, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let out_3d = evaluator.query_3d(0.0, 0.0, 0.0);
        assert_eq!(out_3d.len(), 1);
    }

    #[test]
    fn test_generate_pattern() {
        let config = NeatConfig::cppn(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let evaluator = CppnEvaluator::new(&genome).unwrap();
        let pattern = evaluator.generate_pattern_2d(8, 8, 0).unwrap();

        assert_eq!(pattern.len(), 64);
        for &val in &pattern {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    /// Cheap fingerprint: bit-or all f32 raw representations together.
    /// Two patterns produce identical fingerprints only if every pixel
    /// matches bit-for-bit. Sufficient for regression detection.
    fn pattern_fingerprint(pattern: &[f32]) -> u64 {
        // FNV-1a over the f32 bit patterns — deterministic, zero-allocation,
        // and surfaces single-pixel changes.
        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in pattern {
            for byte in v.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    #[test]
    fn test_generate_pattern_2d_deterministic_checksum() {
        let config = NeatConfig::cppn(2, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let pattern = evaluator.generate_pattern_2d(16, 16, 0).unwrap();
        assert_eq!(pattern.len(), 256);

        // Regenerate with the same seed; pattern must be bit-identical.
        let mut rng2 = ChaCha8Rng::seed_from_u64(12345);
        let genome2 = NeatGenome::fully_connected(NeatConfig::cppn(2, 1), &mut rng2);
        let evaluator2 = CppnEvaluator::new(&genome2).unwrap();
        let pattern2 = evaluator2.generate_pattern_2d(16, 16, 0).unwrap();

        assert_eq!(
            pattern_fingerprint(&pattern),
            pattern_fingerprint(&pattern2),
            "generate_pattern_2d must be deterministic for a fixed seed"
        );
    }

    #[test]
    fn test_generate_voxel_grid_basic() {
        let config = NeatConfig::cppn(3, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let grid = evaluator.generate_voxel_grid([4, 4, 4], 0).unwrap();
        assert_eq!(grid.len(), 64);
        for &v in &grid {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_generate_voxel_grid_deterministic_checksum() {
        let config = NeatConfig::cppn(3, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let grid_a = evaluator.generate_voxel_grid([6, 5, 4], 0).unwrap();
        let grid_b = evaluator.generate_voxel_grid([6, 5, 4], 0).unwrap();
        assert_eq!(pattern_fingerprint(&grid_a), pattern_fingerprint(&grid_b));
    }

    #[test]
    fn test_generate_pattern_input_arity_mismatch() {
        // 3-input CPPN cannot drive a 2D pattern.
        let config = NeatConfig::cppn(3, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let err = evaluator.generate_pattern_2d(4, 4, 0).unwrap_err();
        match err {
            PatternError::InputArityMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            other => panic!("expected InputArityMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_generate_voxel_grid_input_arity_mismatch() {
        // 2-input CPPN cannot drive a 3D voxel grid.
        let config = NeatConfig::cppn(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let err = evaluator.generate_voxel_grid([4, 4, 4], 0).unwrap_err();
        assert!(matches!(
            err,
            PatternError::InputArityMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[cfg(feature = "image")]
    #[test]
    fn test_generate_image_dimensions_and_grayscale() {
        let config = NeatConfig::cppn(2, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let genome = NeatGenome::fully_connected(config, &mut rng);
        let evaluator = CppnEvaluator::new(&genome).unwrap();

        let img = evaluator.generate_image(8, 8, 0).unwrap();
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
        // Every pixel must be R=G=B (grayscale) and alpha=255.
        for pixel in img.pixels() {
            let [r, g, b, a] = pixel.0;
            assert_eq!(r, g);
            assert_eq!(g, b);
            assert_eq!(a, 255);
        }
    }

    #[test]
    #[should_panic(expected = "Input length mismatch")]
    fn test_evaluator_input_mismatch() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let evaluator = CppnEvaluator::new(&genome).unwrap();
        let _ = evaluator.evaluate(&[1.0]); // Wrong number of inputs
    }

    #[test]
    fn test_new_returns_ok_for_acyclic_genome() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let result = CppnEvaluator::new(&genome);
        assert!(result.is_ok(), "new should succeed for acyclic genome");
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
