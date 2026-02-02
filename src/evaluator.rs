//! CPPN evaluator for NEAT genomes.
//!
//! This module provides the [`CppnEvaluator`] which executes a NEAT genome as a
//! Compositional Pattern Producing Network. CPPNs can be queried with spatial
//! coordinates to generate patterns, geometries, and weights for morphogenetic
//! engineering applications.

use crate::gene::{NodeId, NodeType};
use crate::genome::NeatGenome;

/// A compiled, evaluation-ready representation of a NEAT genome.
///
/// The evaluator pre-computes topological order and organizes data for
/// cache-efficient forward propagation.
#[derive(Debug, Clone)]
pub struct CppnEvaluator {
    /// Node activations in topological order.
    activations: Vec<f32>,
    /// Node biases.
    biases: Vec<f32>,
    /// Activation function indices.
    activation_fns: Vec<crate::activation::Activation>,
    /// Connections as `(from_idx, to_idx, weight, enabled)`.
    connections: Vec<(usize, usize, f32, bool)>,
    /// Indices of input nodes in the activations array.
    input_indices: Vec<usize>,
    /// Indices of output nodes in the activations array.
    output_indices: Vec<usize>,
    /// Index of bias node if present.
    bias_index: Option<usize>,
    /// Evaluation order (indices into activations, excluding inputs).
    eval_order: Vec<usize>,
}

impl CppnEvaluator {
    /// Compile a NEAT genome into an efficient evaluator.
    #[must_use]
    pub fn new(genome: &NeatGenome) -> Self {
        // Create mapping from NodeId to dense index
        let mut node_id_to_idx: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();

        // Sort nodes by depth for topological order
        let mut nodes: Vec<_> = genome.nodes.iter().collect();
        nodes.sort_by_key(|(_, n)| n.depth);

        let mut activations = Vec::with_capacity(nodes.len());
        let mut biases = Vec::with_capacity(nodes.len());
        let mut activation_fns = Vec::with_capacity(nodes.len());
        let mut input_indices = Vec::new();
        let mut output_indices = Vec::new();
        let mut bias_index = None;
        let mut eval_order = Vec::new();

        for (node_id, node) in &nodes {
            let idx = activations.len();
            node_id_to_idx.insert(*node_id, idx);

            activations.push(0.0);
            biases.push(node.bias);
            activation_fns.push(node.activation);

            match node.node_type {
                NodeType::Input => input_indices.push(idx),
                NodeType::Output => {
                    output_indices.push(idx);
                    eval_order.push(idx);
                }
                NodeType::Hidden => eval_order.push(idx),
                NodeType::Bias => bias_index = Some(idx),
            }
        }

        // Compile connections
        let connections: Vec<_> = genome
            .connections
            .iter()
            .filter_map(|(_, conn)| {
                let from_idx = node_id_to_idx.get(&conn.input)?;
                let to_idx = node_id_to_idx.get(&conn.output)?;
                Some((*from_idx, *to_idx, conn.weight, conn.enabled))
            })
            .collect();

        Self {
            activations,
            biases,
            activation_fns,
            connections,
            input_indices,
            output_indices,
            bias_index,
            eval_order,
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
        assert_eq!(
            inputs.len(),
            self.input_indices.len(),
            "Input length mismatch: expected {}, got {}",
            self.input_indices.len(),
            inputs.len()
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

        // Forward propagation in topological order
        for &node_idx in &self.eval_order {
            // Sum incoming connections
            let mut sum = self.biases[node_idx];
            for &(from_idx, to_idx, weight, enabled) in &self.connections {
                if enabled && to_idx == node_idx {
                    sum += self.activations[from_idx] * weight;
                }
            }

            // Apply activation function
            self.activations[node_idx] = self.activation_fns[node_idx].apply(sum);
        }

        // Collect outputs
        self.output_indices
            .iter()
            .map(|&idx| self.activations[idx])
            .collect()
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
}

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
/// A flattened grayscale image as `f32` values in `[0, 1]`.
#[allow(clippy::cast_precision_loss)] // Image dimensions are small enough
pub fn generate_pattern(
    evaluator: &mut CppnEvaluator,
    width: usize,
    height: usize,
    output_index: usize,
) -> Vec<f32> {
    let mut pattern = Vec::with_capacity(width * height);

    let width_f = width as f32;
    let height_f = height as f32;

    for y in 0..height {
        for x in 0..width {
            // Normalize coordinates to [-1, 1]
            let nx = (x as f32 / width_f).mul_add(2.0, -1.0);
            let ny = (y as f32 / height_f).mul_add(2.0, -1.0);

            let outputs = evaluator.query_2d(nx, ny);
            let value = outputs.get(output_index).copied().unwrap_or(0.0);

            // Normalize output to [0, 1]
            let normalized = value.mul_add(0.5, 0.5);
            pattern.push(normalized.clamp(0.0, 1.0));
        }
    }

    pattern
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
        let pattern = generate_pattern(&mut evaluator, 8, 8, 0);

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
}
