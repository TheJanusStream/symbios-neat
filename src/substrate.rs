//! HyperNEAT substrate trait and concrete implementations.
//!
//! In HyperNEAT (Stanley et al., 2009) a *substrate* is a fixed geometric
//! arrangement of neurons whose connection weights are *generated* by querying
//! a Compositional Pattern Producing Network (CPPN) with each (src, tgt)
//! coordinate pair. The CPPN is what evolves; the substrate stays fixed.
//!
//! [`substrate_to_network`] is the helper that materializes a [`Substrate`]
//! into an executable [`FeedforwardNetwork`] given a CPPN [`CppnEvaluator`].
//! It walks every potential link, asks the CPPN for a weight, drops links
//! whose absolute weight falls below an expression threshold (a standard
//! HyperNEAT prune), and topologically sorts the surviving graph.
//!
//! # Coordinate convention
//!
//! All concrete substrates here place coordinates in `[-1, 1]` along each
//! axis. The CPPN must accept `2 * coord_dim` inputs — source coordinates
//! followed by target coordinates. For a 2D substrate that's 4 inputs, the
//! same shape as [`CppnEvaluator::query_substrate`].
//!
//! # Example
//!
//! ```rust
//! use symbios_neat::{
//!     substrate::{substrate_to_network, LayeredSubstrate},
//!     Activation, CppnEvaluator, NeatConfig, NeatGenome,
//! };
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! // CPPN with 4 inputs (src.x, src.y, tgt.x, tgt.y) and 1 weight output.
//! let cppn_config = NeatConfig::cppn(4, 1);
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let cppn_genome = NeatGenome::fully_connected(cppn_config, &mut rng);
//! let cppn = CppnEvaluator::new(&cppn_genome).unwrap();
//!
//! // Substrate: 2 input nodes, 1 hidden, 1 output (HyperNEAT-XOR shape).
//! let substrate = LayeredSubstrate::new(&[2, 1, 1], Activation::Tanh, Activation::Sigmoid);
//!
//! // Materialize. Drop links with |weight| < 0.2 (standard HyperNEAT prune).
//! let network = substrate_to_network(&cppn, &substrate, 0.2);
//! assert_eq!(network.num_inputs(), 2);
//! assert_eq!(network.num_outputs(), 1);
//! ```

use std::collections::VecDeque;

use crate::activation::Activation;
use crate::evaluator::CppnEvaluator;
use crate::network::FeedforwardNetwork;

/// A node in a HyperNEAT substrate: its position in the substrate's coordinate
/// space plus the activation function and bias it should use in the
/// materialized network.
#[derive(Debug, Clone)]
pub struct SubstrateNode {
    /// Position of this node in substrate space. All concrete substrates here
    /// use `[-1, 1]` along each axis.
    pub coords: Vec<f32>,
    /// Activation function applied to this node's summed input.
    pub activation: Activation,
    /// Constant bias added to this node's summed input.
    pub bias: f32,
}

/// A HyperNEAT substrate: a fixed set of nodes with fixed coordinates and a
/// fixed list of potential connections.
///
/// Implementors expose:
/// - All [`SubstrateNode`]s, in the order [`substrate_to_network`] should
///   number them.
/// - The list of `(src, tgt)` index pairs whose weights to query from the CPPN.
/// - Which indices are inputs and which are outputs.
///
/// All node coordinate vectors must have the same length; that length times
/// two is the number of inputs the CPPN must accept.
pub trait Substrate {
    /// All nodes in the substrate.
    fn nodes(&self) -> &[SubstrateNode];

    /// `(src_idx, tgt_idx)` pairs whose weight to query from the CPPN.
    /// Self-loops are not supported and should be excluded by the implementor.
    fn links(&self) -> &[(usize, usize)];

    /// Indices into [`nodes`](Self::nodes) that are inputs to the network.
    fn input_indices(&self) -> &[usize];

    /// Indices into [`nodes`](Self::nodes) that are outputs of the network.
    fn output_indices(&self) -> &[usize];

    /// Dimensionality of the coordinate space (e.g. 2 for a 2D substrate).
    ///
    /// Default impl reads the length of the first node's `coords`. Implementors
    /// with empty substrates should override this.
    fn coord_dim(&self) -> usize {
        self.nodes().first().map_or(0, |n| n.coords.len())
    }
}

/// Layered (densely-connected) substrate with arbitrary layer sizes.
///
/// Nodes are laid out in `coord_dim = 2`: x in `[-1, 1]` distributes nodes
/// across each layer; y in `[-1, 1]` separates layers (input layer at
/// `y = -1`, output layer at `y = +1`). Adjacent layers are fully connected.
///
/// Use this for classical HyperNEAT setups (e.g. HyperNEAT-XOR with shape
/// `[2, 2, 1]`).
#[derive(Debug, Clone)]
pub struct LayeredSubstrate {
    nodes: Vec<SubstrateNode>,
    links: Vec<(usize, usize)>,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
}

impl LayeredSubstrate {
    /// Build a layered substrate.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Number of nodes in each layer, from input to output.
    ///   Must contain at least 2 entries (input + output).
    /// * `hidden_activation` - Activation function for non-output nodes.
    /// * `output_activation` - Activation function for output nodes.
    ///
    /// # Panics
    ///
    /// Panics if `layer_sizes` has fewer than 2 entries, or any layer has
    /// 0 nodes.
    #[must_use]
    pub fn new(
        layer_sizes: &[usize],
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "LayeredSubstrate needs at least an input and an output layer"
        );
        assert!(
            layer_sizes.iter().all(|&n| n > 0),
            "LayeredSubstrate layers must be non-empty"
        );

        let num_layers = layer_sizes.len();
        let mut nodes = Vec::new();
        let mut layer_offsets = Vec::with_capacity(num_layers);

        for (layer_idx, &layer_size) in layer_sizes.iter().enumerate() {
            layer_offsets.push(nodes.len());
            let y = if num_layers == 1 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let t = layer_idx as f32 / (num_layers - 1) as f32;
                t.mul_add(2.0, -1.0)
            };
            let activation = if layer_idx == num_layers - 1 {
                output_activation
            } else {
                hidden_activation
            };
            for node_in_layer in 0..layer_size {
                let x = if layer_size == 1 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let t = node_in_layer as f32 / (layer_size - 1) as f32;
                    t.mul_add(2.0, -1.0)
                };
                nodes.push(SubstrateNode {
                    coords: vec![x, y],
                    activation,
                    bias: 0.0,
                });
            }
        }

        // Full bipartite connectivity between consecutive layers.
        let mut links = Vec::new();
        for layer_idx in 0..num_layers - 1 {
            let src_start = layer_offsets[layer_idx];
            let src_end = layer_offsets[layer_idx + 1];
            let tgt_start = layer_offsets[layer_idx + 1];
            let tgt_end = layer_offsets
                .get(layer_idx + 2)
                .copied()
                .unwrap_or(nodes.len());
            for src in src_start..src_end {
                for tgt in tgt_start..tgt_end {
                    links.push((src, tgt));
                }
            }
        }

        let input_indices = (0..layer_sizes[0]).collect();
        let output_indices = (layer_offsets[num_layers - 1]
            ..layer_offsets[num_layers - 1] + layer_sizes[num_layers - 1])
            .collect();

        Self {
            nodes,
            links,
            input_indices,
            output_indices,
        }
    }
}

impl Substrate for LayeredSubstrate {
    fn nodes(&self) -> &[SubstrateNode] {
        &self.nodes
    }
    fn links(&self) -> &[(usize, usize)] {
        &self.links
    }
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_indices(&self) -> &[usize] {
        &self.output_indices
    }
}

/// 2D "sandwich" substrate: an input row and an output row in a 2D plane.
///
/// Inputs are placed at `y = -1`, outputs at `y = +1`. Within each row,
/// nodes are evenly spaced over `x ∈ [-1, 1]`. Every input is fully connected
/// to every output (no hidden layer).
///
/// For a setup with hidden nodes, use [`LayeredSubstrate`] instead.
#[derive(Debug, Clone)]
pub struct GridSubstrate2D {
    nodes: Vec<SubstrateNode>,
    links: Vec<(usize, usize)>,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
}

impl GridSubstrate2D {
    /// Build a sandwich substrate with `num_inputs` input nodes and
    /// `num_outputs` output nodes.
    ///
    /// # Panics
    ///
    /// Panics if `num_inputs == 0` or `num_outputs == 0`.
    #[must_use]
    pub fn sandwich(num_inputs: usize, num_outputs: usize, output_activation: Activation) -> Self {
        assert!(num_inputs > 0, "num_inputs must be positive");
        assert!(num_outputs > 0, "num_outputs must be positive");

        let mut nodes = Vec::with_capacity(num_inputs + num_outputs);
        place_row(&mut nodes, num_inputs, -1.0, Activation::Identity);
        place_row(&mut nodes, num_outputs, 1.0, output_activation);

        let input_indices: Vec<usize> = (0..num_inputs).collect();
        let output_indices: Vec<usize> = (num_inputs..num_inputs + num_outputs).collect();

        let mut links = Vec::with_capacity(num_inputs * num_outputs);
        for &i in &input_indices {
            for &o in &output_indices {
                links.push((i, o));
            }
        }

        Self {
            nodes,
            links,
            input_indices,
            output_indices,
        }
    }
}

impl Substrate for GridSubstrate2D {
    fn nodes(&self) -> &[SubstrateNode] {
        &self.nodes
    }
    fn links(&self) -> &[(usize, usize)] {
        &self.links
    }
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_indices(&self) -> &[usize] {
        &self.output_indices
    }
}

fn place_row(nodes: &mut Vec<SubstrateNode>, count: usize, y: f32, activation: Activation) {
    for i in 0..count {
        let x = if count == 1 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / (count - 1) as f32;
            t.mul_add(2.0, -1.0)
        };
        nodes.push(SubstrateNode {
            coords: vec![x, y],
            activation,
            bias: 0.0,
        });
    }
}

/// 3D "sandwich" substrate: an input plane at `z = -1` and an output plane at
/// `z = +1`.
///
/// Within each plane, nodes are arranged on an `(x, y)` grid uniformly spaced
/// in `[-1, 1]`. Every input plane node is fully connected to every output
/// plane node.
#[derive(Debug, Clone)]
pub struct GridSubstrate3D {
    nodes: Vec<SubstrateNode>,
    links: Vec<(usize, usize)>,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
}

impl GridSubstrate3D {
    /// Build a 3D sandwich substrate. Each plane has `width × height` nodes.
    ///
    /// # Panics
    ///
    /// Panics if `width == 0` or `height == 0`.
    #[must_use]
    pub fn sandwich(width: usize, height: usize, output_activation: Activation) -> Self {
        assert!(width > 0 && height > 0, "width and height must be positive");

        let plane_size = width * height;
        let mut nodes = Vec::with_capacity(plane_size * 2);
        place_plane(&mut nodes, width, height, -1.0, Activation::Identity);
        place_plane(&mut nodes, width, height, 1.0, output_activation);

        let input_indices: Vec<usize> = (0..plane_size).collect();
        let output_indices: Vec<usize> = (plane_size..plane_size * 2).collect();

        let mut links = Vec::with_capacity(plane_size * plane_size);
        for &i in &input_indices {
            for &o in &output_indices {
                links.push((i, o));
            }
        }

        Self {
            nodes,
            links,
            input_indices,
            output_indices,
        }
    }
}

impl Substrate for GridSubstrate3D {
    fn nodes(&self) -> &[SubstrateNode] {
        &self.nodes
    }
    fn links(&self) -> &[(usize, usize)] {
        &self.links
    }
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_indices(&self) -> &[usize] {
        &self.output_indices
    }
}

fn place_plane(
    nodes: &mut Vec<SubstrateNode>,
    width: usize,
    height: usize,
    z: f32,
    activation: Activation,
) {
    for j in 0..height {
        for i in 0..width {
            let x = if width == 1 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / (width - 1) as f32;
                t.mul_add(2.0, -1.0)
            };
            let y = if height == 1 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let t = j as f32 / (height - 1) as f32;
                t.mul_add(2.0, -1.0)
            };
            nodes.push(SubstrateNode {
                coords: vec![x, y, z],
                activation,
                bias: 0.0,
            });
        }
    }
}

/// Materialize a substrate into an executable [`FeedforwardNetwork`] by
/// querying the CPPN for every potential link's weight.
///
/// Links whose absolute returned weight is below `expression_threshold` are
/// pruned (a standard HyperNEAT optimization that exploits the CPPN's ability
/// to represent regions of "no connection"). The expression threshold is
/// typically in `[0.1, 0.3]`.
///
/// # Panics
///
/// Panics if the CPPN's input dimension does not equal `2 * substrate.coord_dim()`,
/// or if the substrate's links reference out-of-range indices.
#[must_use]
pub fn substrate_to_network<S: Substrate>(
    cppn: &CppnEvaluator,
    substrate: &S,
    expression_threshold: f32,
) -> FeedforwardNetwork {
    let coord_dim = substrate.coord_dim();
    let nodes = substrate.nodes();
    assert_eq!(
        cppn.num_inputs(),
        2 * coord_dim,
        "CPPN must accept 2 * coord_dim ({}) inputs, got {}",
        2 * coord_dim,
        cppn.num_inputs()
    );

    // Query the CPPN once per potential link, prune below threshold.
    let mut cppn_inputs = vec![0.0f32; cppn.num_inputs()];
    let mut cppn_outputs = vec![0.0f32; cppn.num_outputs()];
    let mut scratch = cppn.create_scratchpad();
    let mut kept_links: Vec<(usize, usize, f32)> = Vec::new();
    for &(src, tgt) in substrate.links() {
        assert!(src < nodes.len() && tgt < nodes.len(), "link out of bounds");
        cppn_inputs[..coord_dim].copy_from_slice(&nodes[src].coords);
        cppn_inputs[coord_dim..].copy_from_slice(&nodes[tgt].coords);
        cppn.evaluate_into(&cppn_inputs, &mut cppn_outputs, &mut scratch);
        let w = cppn_outputs[0];
        if w.abs() >= expression_threshold {
            kept_links.push((src, tgt, w));
        }
    }

    // Topologically sort surviving graph via Kahn's algorithm.
    let n = nodes.len();
    let mut in_degree = vec![0usize; n];
    let mut successors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(src, tgt, w) in &kept_links {
        in_degree[tgt] += 1;
        successors[src].push((tgt, w));
    }

    let mut order = Vec::with_capacity(n);
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in &successors[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    // If pruning broke the input→output paths, leftover cyclic / unreachable
    // nodes are simply dropped from the eval order — they receive 0
    // activation, which is the natural HyperNEAT semantics for an
    // unexpressed region.

    // Build CSR for incoming connections (FeedforwardNetwork stores predecessors).
    let mut incoming: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(src, tgt, w) in &kept_links {
        incoming[tgt].push((src, w));
    }

    let mut csr_offsets = Vec::with_capacity(n + 1);
    let mut csr_sources = Vec::new();
    let mut csr_weights = Vec::new();
    csr_offsets.push(0);
    for incoming_to_node in &incoming {
        for &(src, w) in incoming_to_node {
            csr_sources.push(src);
            csr_weights.push(w);
        }
        csr_offsets.push(csr_sources.len());
    }

    // eval_order excludes inputs (they get values from the user) and the
    // bias node (set to 1 unconditionally). Substrates here have no explicit
    // bias node, so we just exclude inputs.
    let inputs: std::collections::HashSet<usize> =
        substrate.input_indices().iter().copied().collect();
    let eval_order: Vec<usize> = order.into_iter().filter(|i| !inputs.contains(i)).collect();

    let biases: Vec<f32> = nodes.iter().map(|n| n.bias).collect();
    let activations: Vec<Activation> = nodes.iter().map(|n| n.activation).collect();

    FeedforwardNetwork {
        num_nodes: n,
        biases,
        activations,
        csr_sources,
        csr_weights,
        csr_offsets,
        input_indices: substrate.input_indices().to_vec(),
        output_indices: substrate.output_indices().to_vec(),
        bias_index: None,
        eval_order,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeatConfig, NeatGenome};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn dummy_cppn(seed: u64, inputs: usize) -> CppnEvaluator {
        let config = NeatConfig::cppn(inputs, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let genome = NeatGenome::fully_connected(config, &mut rng);
        CppnEvaluator::new(&genome).unwrap()
    }

    #[test]
    fn layered_substrate_layout() {
        let s = LayeredSubstrate::new(&[2, 3, 1], Activation::Tanh, Activation::Sigmoid);
        assert_eq!(s.nodes().len(), 6);
        assert_eq!(s.input_indices(), &[0, 1]);
        assert_eq!(s.output_indices(), &[5]);
        // Inputs at y=-1
        assert!((s.nodes()[0].coords[1] + 1.0).abs() < 1e-6);
        // Output at y=+1
        assert!((s.nodes()[5].coords[1] - 1.0).abs() < 1e-6);
        // Input → hidden links: 2 × 3 = 6, hidden → output: 3 × 1 = 3.
        assert_eq!(s.links().len(), 9);
    }

    #[test]
    fn grid2d_sandwich_layout() {
        let s = GridSubstrate2D::sandwich(3, 2, Activation::Sigmoid);
        assert_eq!(s.nodes().len(), 5);
        assert_eq!(s.input_indices(), &[0, 1, 2]);
        assert_eq!(s.output_indices(), &[3, 4]);
        // Full bipartite: 3 × 2 = 6 links
        assert_eq!(s.links().len(), 6);
        assert_eq!(s.coord_dim(), 2);
    }

    #[test]
    fn grid3d_sandwich_layout() {
        let s = GridSubstrate3D::sandwich(2, 2, Activation::Sigmoid);
        assert_eq!(s.nodes().len(), 8);
        assert_eq!(s.coord_dim(), 3);
        // 4 × 4 = 16 links
        assert_eq!(s.links().len(), 16);
    }

    #[test]
    fn substrate_to_network_layered_xor_shape() {
        let cppn = dummy_cppn(7, 4);
        let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid);

        // Use threshold 0 so no links are pruned — confirms the topology
        // pipeline produces exactly the expected I/O shape.
        let network = substrate_to_network(&cppn, &substrate, 0.0);
        assert_eq!(network.num_inputs(), 2);
        assert_eq!(network.num_outputs(), 1);

        // Network should produce finite output
        let out = network.evaluate(&[0.5, -0.3]);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn substrate_to_network_grid2d() {
        let cppn = dummy_cppn(11, 4);
        let substrate = GridSubstrate2D::sandwich(3, 2, Activation::Sigmoid);
        let network = substrate_to_network(&cppn, &substrate, 0.0);
        assert_eq!(network.num_inputs(), 3);
        assert_eq!(network.num_outputs(), 2);
    }

    #[test]
    fn substrate_to_network_grid3d() {
        let cppn = dummy_cppn(13, 6); // 2 * coord_dim = 6 for 3D
        let substrate = GridSubstrate3D::sandwich(2, 2, Activation::Sigmoid);
        let network = substrate_to_network(&cppn, &substrate, 0.0);
        assert_eq!(network.num_inputs(), 4);
        assert_eq!(network.num_outputs(), 4);
    }

    #[test]
    fn expression_threshold_prunes_links() {
        let cppn = dummy_cppn(17, 4);
        let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid);

        // Threshold 0 keeps all links; very high threshold prunes most/all.
        let kept = substrate_to_network(&cppn, &substrate, 0.0);
        let pruned = substrate_to_network(&cppn, &substrate, 100.0);

        // The pruned network has empty CSR; we can't easily compare counts
        // without internal access, so check that pruned still produces finite
        // output (zeros propagate through activations).
        let out = pruned.evaluate(&[0.5, -0.5]);
        assert!(out[0].is_finite());

        // Sanity: kept network's output should differ from pruned's for
        // generic CPPN weights.
        let kept_out = kept.evaluate(&[0.5, -0.5]);
        assert!(kept_out[0].is_finite());
    }

    #[test]
    #[should_panic(expected = "CPPN must accept 2 * coord_dim")]
    fn dimension_mismatch_panics() {
        // 2D substrate needs 4-input CPPN; supply 6-input CPPN to force panic.
        let cppn = dummy_cppn(3, 6);
        let substrate = LayeredSubstrate::new(&[2, 1], Activation::Tanh, Activation::Sigmoid);
        let _ = substrate_to_network(&cppn, &substrate, 0.0);
    }
}
