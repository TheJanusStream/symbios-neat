//! Generic feed-forward network with CSR-format forward propagation.
//!
//! [`FeedforwardNetwork`] is the underlying evaluation engine used by
//! [`CppnEvaluator`](crate::CppnEvaluator) for NEAT genomes and by
//! [`substrate_to_network`](crate::substrate::substrate_to_network) for
//! HyperNEAT-style indirect encoding. It is a topologically-sorted, immutable
//! network whose forward pass is allocation-free given a reusable
//! [`Scratchpad`].
//!
//! Most users will not interact with this type directly — construct a
//! [`CppnEvaluator`](crate::CppnEvaluator) for direct NEAT evaluation, or call
//! [`substrate_to_network`](crate::substrate::substrate_to_network) to derive a
//! network from a CPPN + substrate.

use crate::activation::Activation;

/// Reusable activation buffer for [`FeedforwardNetwork::evaluate_into`].
///
/// Allocate one per thread when running parallel evaluation; the network
/// itself is immutable and `Sync`.
#[derive(Debug, Clone)]
pub struct Scratchpad {
    activations: Vec<f32>,
}

impl Scratchpad {
    pub(crate) fn new(num_nodes: usize) -> Self {
        Self {
            activations: vec![0.0; num_nodes],
        }
    }

    #[inline]
    fn reset(&mut self) {
        for a in &mut self.activations {
            *a = 0.0;
        }
    }
}

/// Topologically-sorted feed-forward network in Compressed Sparse Row format.
///
/// Construction is the responsibility of higher-level builders
/// ([`CppnEvaluator::new`](crate::CppnEvaluator::new),
/// [`substrate_to_network`](crate::substrate::substrate_to_network)). Once
/// built, evaluation is O(N + E) with cache-friendly contiguous reads.
#[derive(Debug, Clone)]
pub struct FeedforwardNetwork {
    pub(crate) num_nodes: usize,
    pub(crate) biases: Vec<f32>,
    pub(crate) activations: Vec<Activation>,
    pub(crate) csr_sources: Vec<usize>,
    pub(crate) csr_weights: Vec<f32>,
    pub(crate) csr_offsets: Vec<usize>,
    pub(crate) input_indices: Vec<usize>,
    pub(crate) output_indices: Vec<usize>,
    pub(crate) bias_index: Option<usize>,
    pub(crate) eval_order: Vec<usize>,
}

impl FeedforwardNetwork {
    /// Number of input nodes the network expects.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.input_indices.len()
    }

    /// Number of output nodes the network produces.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.output_indices.len()
    }

    /// Activation function attached to a specific output node.
    ///
    /// # Panics
    ///
    /// Panics if `output_index` is out of bounds.
    #[must_use]
    pub fn output_activation(&self, output_index: usize) -> Activation {
        self.activations[self.output_indices[output_index]]
    }

    /// Allocate a fresh [`Scratchpad`] sized for this network.
    #[must_use]
    pub fn create_scratchpad(&self) -> Scratchpad {
        Scratchpad::new(self.num_nodes)
    }

    /// Evaluate the network into a caller-supplied output buffer using a
    /// caller-supplied scratchpad. Allocation-free; safe to call repeatedly.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` differs from [`num_inputs`](Self::num_inputs)
    /// or `outputs.len()` differs from [`num_outputs`](Self::num_outputs).
    pub fn evaluate_into(&self, inputs: &[f32], outputs: &mut [f32], scratch: &mut Scratchpad) {
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

        scratch.reset();

        for (i, &idx) in self.input_indices.iter().enumerate() {
            scratch.activations[idx] = inputs[i];
        }

        if let Some(bias_idx) = self.bias_index {
            scratch.activations[bias_idx] = 1.0;
        }

        for &node_idx in &self.eval_order {
            let mut sum: f64 = f64::from(self.biases[node_idx]);
            let start = self.csr_offsets[node_idx];
            let end = self.csr_offsets[node_idx + 1];
            for i in start..end {
                let from_idx = self.csr_sources[i];
                let weight = f64::from(self.csr_weights[i]);
                let activation = f64::from(scratch.activations[from_idx]);
                sum += activation * weight;
            }

            #[allow(clippy::cast_possible_truncation)]
            let sum_f32 = sum as f32;
            scratch.activations[node_idx] = self.activations[node_idx].apply(sum_f32);
        }

        for (i, &idx) in self.output_indices.iter().enumerate() {
            outputs[i] = scratch.activations[idx];
        }
    }

    /// Evaluate the network, allocating output buffer + scratchpad each call.
    ///
    /// For repeated evaluation prefer
    /// [`evaluate_into`](Self::evaluate_into) with a reusable buffer.
    #[must_use]
    pub fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut scratch = self.create_scratchpad();
        let mut outputs = vec![0.0; self.output_indices.len()];
        self.evaluate_into(inputs, &mut outputs, &mut scratch);
        outputs
    }
}
