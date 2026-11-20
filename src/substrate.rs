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
//! [`substrate_to_network_with`] does the same with a [`SubstratePainter`]
//! deciding what the CPPN is asked and what its answers mean, for the
//! refinements classic HyperNEAT leaves out: a separate CPPN output per kind
//! of link (multi-spatial substrates), a link-expression output that decides
//! which links exist, CPPN inputs beyond the two positions (such as their
//! distance), and painted node biases. [`ThresholdPainter`] is the classic
//! scheme as a painter.
//!
//! # Coordinate convention
//!
//! All concrete substrates here place coordinates in `[-1, 1]` along each
//! axis. For [`substrate_to_network`] the CPPN must accept `2 * coord_dim`
//! inputs: source coordinates followed by target coordinates. For a 2D
//! substrate that's 4 inputs, the same shape as
//! [`CppnEvaluator::query_substrate`]. With a [`SubstratePainter`], the
//! CPPN takes whatever the painter's queries hold.
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

use std::collections::{HashSet, VecDeque};

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

/// Decides how a CPPN paints a substrate: what to ask it about each
/// potential link and node, and what its answers mean.
/// [`substrate_to_network_with`] does the rest: it evaluates the CPPN, keeps
/// the links the painter expresses and assembles them into a
/// [`FeedforwardNetwork`].
///
/// [`ThresholdPainter`] is classic HyperNEAT: the source's coordinates, then
/// the target's; output 0 as the weight; a magnitude threshold. Write your
/// own painter to
///
/// - give each kind of link its own CPPN output (a multi-spatial substrate,
///   Pugh and Stanley 2013), with the kinds kept however your substrate likes;
/// - decide which links exist with a dedicated output (link expression,
///   Verbancsics and Stanley 2011) rather than by the weight's magnitude;
/// - feed the CPPN more than the two positions, such as their distance;
/// - paint node biases.
///
/// Every query buffer holds one value per CPPN input and starts zeroed.
///
/// # Example
///
/// A painter for a `[2, 2, 1]` layered substrate whose CPPN takes
/// `(x1, y1, x2, y2, distance)` and has four outputs: the input-to-hidden
/// weight, the hidden-to-output weight, link expression, and bias.
///
/// ```rust
/// use symbios_neat::substrate::{
///     substrate_to_network_with, write_endpoints, LayeredSubstrate, Substrate, SubstratePainter,
/// };
/// use symbios_neat::{Activation, CppnEvaluator, NeatConfig, NeatGenome};
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
///
/// struct LeoPainter;
///
/// impl SubstratePainter<LayeredSubstrate> for LeoPainter {
///     fn link_query(&self, substrate: &LayeredSubstrate, link: usize, query: &mut [f32]) {
///         let written = write_endpoints(substrate, link, query);
///         let (from, to) = substrate.links()[link];
///         let (a, b) = (&substrate.nodes()[from].coords, &substrate.nodes()[to].coords);
///         query[written] = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f32>().sqrt();
///     }
///
///     fn link_weight(&self, substrate: &LayeredSubstrate, link: usize, out: &[f32]) -> Option<f32> {
///         if out[2] <= 0.0 {
///             return None; // not expressed
///         }
///         let (from, _) = substrate.links()[link];
///         let from_inputs = substrate.input_indices().contains(&from);
///         Some(3.0 * if from_inputs { out[0] } else { out[1] })
///     }
///
///     fn bias_query(&self, substrate: &LayeredSubstrate, node: usize, query: &mut [f32]) -> bool {
///         let at = &substrate.nodes()[node].coords;
///         query[..2].copy_from_slice(at);
///         query[2..4].copy_from_slice(at);
///         true
///     }
///
///     fn bias(&self, _substrate: &LayeredSubstrate, _node: usize, out: &[f32]) -> f32 {
///         out[3]
///     }
/// }
///
/// let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Tanh);
/// let mut rng = ChaCha8Rng::seed_from_u64(1);
/// let genome = NeatGenome::fully_connected(NeatConfig::cppn(5, 4), &mut rng);
/// let cppn = CppnEvaluator::new(&genome).unwrap();
/// let network = substrate_to_network_with(&cppn, &substrate, &LeoPainter);
/// assert_eq!(network.num_inputs(), 2);
/// assert_eq!(network.num_outputs(), 1);
/// ```
pub trait SubstratePainter<S: Substrate + ?Sized> {
    /// Fills `query` with the CPPN inputs for the potential link
    /// `substrate.links()[link]`.
    fn link_query(&self, substrate: &S, link: usize, query: &mut [f32]);

    /// The weight of link `link`, given the CPPN's `outputs` for its query.
    /// `None` leaves the link out of the network, and so does a weight that
    /// is not finite.
    fn link_weight(&self, substrate: &S, link: usize, outputs: &[f32]) -> Option<f32>;

    /// Fills `query` with the CPPN inputs for node `node`'s bias and returns
    /// `true`, or returns `false` to keep its [`SubstrateNode::bias`], as the
    /// default does. Never asked about input nodes, whose values are given.
    fn bias_query(&self, substrate: &S, node: usize, query: &mut [f32]) -> bool {
        let _ = (substrate, node, query);
        false
    }

    /// The bias of node `node`, given the CPPN's `outputs` for its bias
    /// query; asked only when [`bias_query`](Self::bias_query) returned
    /// `true`. The default keeps the node's own bias, and so does a bias
    /// that is not finite.
    fn bias(&self, substrate: &S, node: usize, outputs: &[f32]) -> f32 {
        let _ = outputs;
        substrate.nodes()[node].bias
    }
}

/// Classic HyperNEAT painting, as [`substrate_to_network`] does it: the CPPN
/// is asked with the source's coordinates followed by the target's, output 0
/// is the link's weight, links weaker than `threshold` are left out, and
/// nodes keep their own biases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdPainter {
    /// Links with `|weight| < threshold` are pruned; typically in
    /// `[0.1, 0.3]`.
    pub threshold: f32,
}

impl<S: Substrate + ?Sized> SubstratePainter<S> for ThresholdPainter {
    fn link_query(&self, substrate: &S, link: usize, query: &mut [f32]) {
        write_endpoints(substrate, link, query);
    }

    fn link_weight(&self, _substrate: &S, _link: usize, outputs: &[f32]) -> Option<f32> {
        let weight = outputs[0];
        (weight.abs() >= self.threshold).then_some(weight)
    }
}

/// Writes the coordinates of link `link`'s source node, then its target
/// node's, to the front of `query`, and returns how many values it wrote, so
/// that a painter can append values of its own after them (such as the two
/// nodes' distance).
///
/// # Panics
///
/// Panics if `query` is too short for both nodes' coordinates.
pub fn write_endpoints<S: Substrate + ?Sized>(
    substrate: &S,
    link: usize,
    query: &mut [f32],
) -> usize {
    let (source, target) = substrate.links()[link];
    let nodes = substrate.nodes();
    let (from, to) = (&nodes[source].coords, &nodes[target].coords);
    query[..from.len()].copy_from_slice(from);
    query[from.len()..from.len() + to.len()].copy_from_slice(to);
    from.len() + to.len()
}

/// Materialize a substrate into an executable [`FeedforwardNetwork`] by
/// querying the CPPN for every potential link's weight.
///
/// Links whose absolute returned weight is below `expression_threshold` are
/// pruned (a standard HyperNEAT optimization that exploits the CPPN's ability
/// to represent regions of "no connection"). The expression threshold is
/// typically in `[0.1, 0.3]`. This is [`substrate_to_network_with`] with a
/// [`ThresholdPainter`].
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
    assert_eq!(
        cppn.num_inputs(),
        2 * coord_dim,
        "CPPN must accept 2 * coord_dim ({}) inputs, got {}",
        2 * coord_dim,
        cppn.num_inputs()
    );
    substrate_to_network_with(
        cppn,
        substrate,
        &ThresholdPainter {
            threshold: expression_threshold,
        },
    )
}

/// Materialize a substrate into an executable [`FeedforwardNetwork`], with
/// `painter` deciding what the CPPN is asked about each potential link and
/// non-input node and what its answers mean (see [`SubstratePainter`]).
///
/// The surviving links are topologically sorted; nodes left in a cycle, or
/// unreachable, receive no activation (the natural HyperNEAT reading of an
/// unexpressed region). The network's inputs and outputs are the
/// substrate's, in its order.
///
/// # Panics
///
/// Panics if the substrate's links reference out-of-range indices, or if the
/// painter writes past the CPPN's inputs or reads past its outputs.
#[must_use]
pub fn substrate_to_network_with<S, P>(
    cppn: &CppnEvaluator,
    substrate: &S,
    painter: &P,
) -> FeedforwardNetwork
where
    S: Substrate + ?Sized,
    P: SubstratePainter<S> + ?Sized,
{
    let nodes = substrate.nodes();
    let mut query = vec![0.0f32; cppn.num_inputs()];
    let mut outputs = vec![0.0f32; cppn.num_outputs()];
    let mut scratch = cppn.create_scratchpad();

    // Ask the CPPN about every potential link; keep the expressed ones.
    let mut kept_links: Vec<(usize, usize, f32)> = Vec::new();
    for (index, &(src, tgt)) in substrate.links().iter().enumerate() {
        assert!(src < nodes.len() && tgt < nodes.len(), "link out of bounds");
        query.fill(0.0);
        painter.link_query(substrate, index, &mut query);
        cppn.evaluate_into(&query, &mut outputs, &mut scratch);
        if let Some(weight) = painter.link_weight(substrate, index, &outputs)
            && weight.is_finite()
        {
            kept_links.push((src, tgt, weight));
        }
    }

    // Paint the biases the painter asks for. Inputs get theirs from the
    // caller, so they are never asked about.
    let inputs: HashSet<usize> = substrate.input_indices().iter().copied().collect();
    let biases: Vec<f32> = (0..nodes.len())
        .map(|node| {
            let own = nodes[node].bias;
            if inputs.contains(&node) {
                return own;
            }
            query.fill(0.0);
            if !painter.bias_query(substrate, node, &mut query) {
                return own;
            }
            cppn.evaluate_into(&query, &mut outputs, &mut scratch);
            let bias = painter.bias(substrate, node, &outputs);
            if bias.is_finite() { bias } else { own }
        })
        .collect();

    assemble(substrate, &kept_links, biases, &inputs)
}

/// Builds the network from the expressed links: Kahn's topological order,
/// then predecessor CSR in link order.
fn assemble<S: Substrate + ?Sized>(
    substrate: &S,
    kept_links: &[(usize, usize, f32)],
    biases: Vec<f32>,
    inputs: &HashSet<usize>,
) -> FeedforwardNetwork {
    let nodes = substrate.nodes();

    // Topologically sort surviving graph via Kahn's algorithm.
    let n = nodes.len();
    let mut in_degree = vec![0usize; n];
    let mut successors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(src, tgt, w) in kept_links {
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
    for &(src, tgt, w) in kept_links {
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
    let eval_order: Vec<usize> = order.into_iter().filter(|i| !inputs.contains(i)).collect();

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

        // Threshold 0 keeps every potential link (2 x 2 + 2 x 1); the high
        // threshold keeps none, and zeros still propagate to a finite output.
        assert_eq!(kept.num_connections(), 6);
        assert_eq!(pruned.num_connections(), 0);
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

    /// `substrate_to_network` as it was before painters, kept to prove the
    /// painter-based version builds exactly the same networks.
    fn classic_substrate_to_network<S: Substrate>(
        cppn: &CppnEvaluator,
        substrate: &S,
        expression_threshold: f32,
    ) -> FeedforwardNetwork {
        let coord_dim = substrate.coord_dim();
        let nodes = substrate.nodes();
        let mut cppn_inputs = vec![0.0f32; cppn.num_inputs()];
        let mut cppn_outputs = vec![0.0f32; cppn.num_outputs()];
        let mut scratch = cppn.create_scratchpad();
        let mut kept_links: Vec<(usize, usize, f32)> = Vec::new();
        for &(src, tgt) in substrate.links() {
            cppn_inputs[..coord_dim].copy_from_slice(&nodes[src].coords);
            cppn_inputs[coord_dim..].copy_from_slice(&nodes[tgt].coords);
            cppn.evaluate_into(&cppn_inputs, &mut cppn_outputs, &mut scratch);
            let w = cppn_outputs[0];
            if w.abs() >= expression_threshold {
                kept_links.push((src, tgt, w));
            }
        }
        let inputs: HashSet<usize> = substrate.input_indices().iter().copied().collect();
        let biases = nodes.iter().map(|n| n.bias).collect();
        assemble(substrate, &kept_links, biases, &inputs)
    }

    /// A CPPN with some hidden structure, from a seed.
    fn evolved_cppn(seed: u64, inputs: usize) -> CppnEvaluator {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut genome = NeatGenome::fully_connected(NeatConfig::cppn(inputs, 1), &mut rng);
        for _ in 0..5 {
            symbios_genetics::Genotype::mutate(&mut genome, &mut rng, 1.0);
        }
        genome.break_cycles();
        CppnEvaluator::new(&genome).unwrap()
    }

    #[test]
    fn the_threshold_painter_builds_exactly_the_classic_network() {
        for seed in 0..6 {
            for threshold in [0.0, 0.2, 0.5] {
                let layered = LayeredSubstrate::new(&[3, 4, 2], Activation::Tanh, Activation::Tanh);
                let grid2 = GridSubstrate2D::sandwich(4, 3, Activation::Sigmoid);
                let grid3 = GridSubstrate3D::sandwich(2, 3, Activation::Tanh);
                let cppn2 = evolved_cppn(seed, 4);
                let cppn3 = evolved_cppn(seed + 100, 6);
                let pairs = [
                    (
                        substrate_to_network(&cppn2, &layered, threshold),
                        classic_substrate_to_network(&cppn2, &layered, threshold),
                    ),
                    (
                        substrate_to_network(&cppn2, &grid2, threshold),
                        classic_substrate_to_network(&cppn2, &grid2, threshold),
                    ),
                    (
                        substrate_to_network(&cppn3, &grid3, threshold),
                        classic_substrate_to_network(&cppn3, &grid3, threshold),
                    ),
                ];
                for (painted, classic) in pairs {
                    assert_eq!(painted.num_connections(), classic.num_connections());
                    let mut rng = ChaCha8Rng::seed_from_u64(seed);
                    for _ in 0..4 {
                        let input: Vec<f32> = (0..painted.num_inputs())
                            .map(|_| rand::Rng::random_range(&mut rng, -1.0..1.0))
                            .collect();
                        assert_eq!(painted.evaluate(&input), classic.evaluate(&input));
                    }
                }
            }
        }
    }

    /// Two inputs and one identity output in 2D, linked by two links of
    /// different kinds.
    struct KindedSubstrate {
        nodes: Vec<SubstrateNode>,
        links: Vec<(usize, usize)>,
        kinds: Vec<usize>,
    }

    impl KindedSubstrate {
        fn new(input_positions: [[f32; 2]; 2], output_bias: f32) -> Self {
            let node = |coords: [f32; 2], activation, bias| SubstrateNode {
                coords: coords.to_vec(),
                activation,
                bias,
            };
            Self {
                nodes: vec![
                    node(input_positions[0], Activation::Identity, 0.0),
                    node(input_positions[1], Activation::Identity, 0.0),
                    node([0.0, 1.0], Activation::Identity, output_bias),
                ],
                links: vec![(0, 2), (1, 2)],
                kinds: vec![0, 1],
            }
        }
    }

    impl Substrate for KindedSubstrate {
        fn nodes(&self) -> &[SubstrateNode] {
            &self.nodes
        }
        fn links(&self) -> &[(usize, usize)] {
            &self.links
        }
        fn input_indices(&self) -> &[usize] {
            &[0, 1]
        }
        fn output_indices(&self) -> &[usize] {
            &[2]
        }
    }

    /// A CPPN with no hidden nodes: output `k` is `tanh(bias_k + sum of
    /// input_weights)` for each `(input, output, weight)` given.
    fn wired_cppn(
        inputs: usize,
        outputs: usize,
        wires: &[(Option<usize>, usize, f32)],
    ) -> CppnEvaluator {
        let mut genome = NeatGenome::minimal(NeatConfig::cppn(inputs, outputs));
        let bias = genome.bias_id.unwrap();
        for (i, &(input, output, weight)) in wires.iter().enumerate() {
            let from = input.map_or(bias, |k| genome.input_ids[k]);
            let to = genome.output_ids[output];
            genome.connections.insert(crate::ConnectionGene::new(
                1000 + i as u64,
                from,
                to,
                weight,
            ));
        }
        CppnEvaluator::new(&genome).unwrap()
    }

    /// Reads output `kind` as each link's weight, a link-expression output,
    /// a distance appended to the endpoints, and a bias output.
    struct KindPainter {
        expression: Option<usize>,
        bias: Option<usize>,
    }

    impl SubstratePainter<KindedSubstrate> for KindPainter {
        fn link_query(&self, substrate: &KindedSubstrate, link: usize, query: &mut [f32]) {
            let written = write_endpoints(substrate, link, query);
            let (from, to) = substrate.links[link];
            let (a, b) = (&substrate.nodes[from].coords, &substrate.nodes[to].coords);
            query[written] = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        }
        fn link_weight(
            &self,
            substrate: &KindedSubstrate,
            link: usize,
            out: &[f32],
        ) -> Option<f32> {
            if self.expression.is_some_and(|leo| out[leo] <= 0.0) {
                return None;
            }
            Some(out[substrate.kinds[link]])
        }
        fn bias_query(&self, substrate: &KindedSubstrate, node: usize, query: &mut [f32]) -> bool {
            let at = &substrate.nodes[node].coords;
            query[..2].copy_from_slice(at);
            query[2..4].copy_from_slice(at);
            self.bias.is_some()
        }
        fn bias(&self, _substrate: &KindedSubstrate, _node: usize, out: &[f32]) -> f32 {
            out[self.bias.unwrap()]
        }
    }

    #[test]
    fn each_kind_of_link_can_read_its_own_cppn_output() {
        let substrate = KindedSubstrate::new([[-1.0, 0.0], [1.0, 0.0]], 0.0);
        let cppn = wired_cppn(5, 2, &[(None, 0, 0.5), (None, 1, -1.0)]);
        let painter = KindPainter {
            expression: None,
            bias: None,
        };
        let network = substrate_to_network_with(&cppn, &substrate, &painter);
        assert_eq!(network.num_connections(), 2);
        assert!((network.evaluate(&[1.0, 0.0])[0] - 0.5f32.tanh()).abs() < 1e-6);
        assert!((network.evaluate(&[0.0, 1.0])[0] - (-1.0f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn a_link_expression_output_decides_which_links_exist() {
        // Expression is tanh(5 * source x): off for the input at x = -1.
        let substrate = KindedSubstrate::new([[-1.0, 0.0], [1.0, 0.0]], 0.0);
        let cppn = wired_cppn(5, 3, &[(None, 0, 0.5), (None, 1, 0.8), (Some(0), 2, 5.0)]);
        let painter = KindPainter {
            expression: Some(2),
            bias: None,
        };
        let network = substrate_to_network_with(&cppn, &substrate, &painter);
        assert_eq!(network.num_connections(), 1);
        assert_eq!(network.evaluate(&[1.0, 0.0])[0], 0.0);
        assert!((network.evaluate(&[0.0, 1.0])[0] - 0.8f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn queries_can_carry_values_beyond_the_two_positions() {
        // Weight tanh(distance): sqrt(2) for the first input, sqrt(0.5) for the second.
        let substrate = KindedSubstrate::new([[-1.0, 0.0], [0.5, 0.5]], 0.0);
        let cppn = wired_cppn(5, 2, &[(Some(4), 0, 1.0), (Some(4), 1, 1.0)]);
        let painter = KindPainter {
            expression: None,
            bias: None,
        };
        let network = substrate_to_network_with(&cppn, &substrate, &painter);
        let near = 0.5f32.sqrt().tanh();
        let far = 2.0f32.sqrt().tanh();
        assert!((network.evaluate(&[1.0, 0.0])[0] - far).abs() < 1e-6);
        assert!((network.evaluate(&[0.0, 1.0])[0] - near).abs() < 1e-6);
    }

    #[test]
    fn painted_biases_replace_a_nodes_own_unless_the_painter_declines() {
        let substrate = KindedSubstrate::new([[-1.0, 0.0], [1.0, 0.0]], 0.25);
        // No link is expressed; output 2 is the bias, tanh(0.7).
        let cppn = wired_cppn(5, 3, &[(None, 1, -5.0), (None, 2, 0.7)]);
        let painted = substrate_to_network_with(
            &cppn,
            &substrate,
            &KindPainter {
                expression: Some(1),
                bias: Some(2),
            },
        );
        assert!((painted.evaluate(&[1.0, 1.0])[0] - 0.7f32.tanh()).abs() < 1e-6);
        let own = substrate_to_network_with(
            &cppn,
            &substrate,
            &KindPainter {
                expression: Some(1),
                bias: None,
            },
        );
        assert_eq!(own.evaluate(&[1.0, 1.0])[0], 0.25);
    }

    #[test]
    fn input_nodes_are_never_asked_for_a_bias() {
        struct Counting(std::cell::Cell<usize>);
        impl SubstratePainter<LayeredSubstrate> for Counting {
            fn link_query(&self, substrate: &LayeredSubstrate, link: usize, query: &mut [f32]) {
                write_endpoints(substrate, link, query);
            }
            fn link_weight(&self, _: &LayeredSubstrate, _: usize, out: &[f32]) -> Option<f32> {
                Some(out[0])
            }
            fn bias_query(&self, substrate: &LayeredSubstrate, node: usize, _: &mut [f32]) -> bool {
                assert!(
                    !substrate.input_indices().contains(&node),
                    "asked about input {node}"
                );
                self.0.set(self.0.get() + 1);
                false
            }
        }
        let substrate = LayeredSubstrate::new(&[3, 2, 1], Activation::Tanh, Activation::Tanh);
        let painter = Counting(std::cell::Cell::new(0));
        let _ = substrate_to_network_with(&dummy_cppn(5, 4), &substrate, &painter);
        assert_eq!(painter.0.get(), 3);
    }

    #[test]
    fn answers_that_are_not_finite_are_ignored() {
        struct Broken;
        impl SubstratePainter<KindedSubstrate> for Broken {
            fn link_query(&self, _: &KindedSubstrate, _: usize, _: &mut [f32]) {}
            fn link_weight(&self, _: &KindedSubstrate, link: usize, _: &[f32]) -> Option<f32> {
                Some(if link == 0 { f32::NAN } else { f32::INFINITY })
            }
            fn bias_query(&self, _: &KindedSubstrate, _: usize, _: &mut [f32]) -> bool {
                true
            }
            fn bias(&self, _: &KindedSubstrate, _: usize, _: &[f32]) -> f32 {
                f32::NAN
            }
        }
        let substrate = KindedSubstrate::new([[-1.0, 0.0], [1.0, 0.0]], 0.25);
        let network = substrate_to_network_with(&dummy_cppn(1, 4), &substrate, &Broken);
        assert_eq!(network.num_connections(), 0);
        assert_eq!(network.evaluate(&[1.0, 1.0])[0], 0.25);
    }

    #[test]
    fn write_endpoints_writes_the_source_then_the_target() {
        let substrate = LayeredSubstrate::new(&[2, 1], Activation::Tanh, Activation::Tanh);
        let mut query = [9.0f32; 6];
        let written = write_endpoints(&substrate, 1, &mut query);
        assert_eq!(written, 4);
        // Link 1 runs from the second input (1, -1) to the output (0, 1).
        assert_eq!(query, [1.0, -1.0, 0.0, 1.0, 9.0, 9.0]);
    }
}
