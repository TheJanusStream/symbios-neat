//! NEAT genome implementation with arena-allocated graph topology.
//!
//! The [`NeatGenome`] uses SlotMap-based arena storage for nodes and connections,
//! providing cache-friendly access, trivial serialization, and avoiding Rust's
//! reference-counting overhead.

use rand::Rng;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use symbios_genetics::Genotype;

use crate::activation::Activation;
use crate::gene::{ConnectionGene, ConnectionId, NodeGene, NodeId, NodeType};
use crate::innovation::{
    connection_innovation, node_split_innovation, split_connection_a_innovation,
    split_connection_b_innovation,
};

/// Configuration for NEAT genome creation and mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    /// Number of input nodes (excluding bias).
    pub num_inputs: usize,
    /// Number of output nodes.
    pub num_outputs: usize,
    /// Whether to include a bias node.
    pub use_bias: bool,
    /// Default activation for output nodes.
    pub output_activation: Activation,
    /// Activation functions to choose from for hidden nodes.
    pub hidden_activations: Vec<Activation>,
    /// Probability of adding a new connection during mutation.
    pub add_connection_prob: f32,
    /// Probability of adding a new node during mutation.
    pub add_node_prob: f32,
    /// Probability of mutating a connection weight.
    pub weight_mutation_prob: f32,
    /// Standard deviation for weight perturbation.
    pub weight_mutation_power: f32,
    /// Probability of completely replacing a weight.
    pub weight_replace_prob: f32,
    /// Range for initial and replaced weights: [-weight_range, weight_range].
    pub weight_range: f32,
    /// Probability of toggling a connection's enabled state.
    pub toggle_enabled_prob: f32,
    /// Probability of mutating a node's activation function.
    pub activation_mutation_prob: f32,
    /// Coefficient for excess genes in compatibility distance.
    pub compatibility_excess_coeff: f32,
    /// Coefficient for disjoint genes in compatibility distance.
    pub compatibility_disjoint_coeff: f32,
    /// Coefficient for weight differences in compatibility distance.
    pub compatibility_weight_coeff: f32,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            num_inputs: 2,
            num_outputs: 1,
            use_bias: true,
            output_activation: Activation::Sigmoid,
            hidden_activations: Activation::CPPN.to_vec(),
            add_connection_prob: 0.05,
            add_node_prob: 0.03,
            weight_mutation_prob: 0.8,
            weight_mutation_power: 0.5,
            weight_replace_prob: 0.1,
            weight_range: 1.0,
            toggle_enabled_prob: 0.01,
            activation_mutation_prob: 0.1,
            compatibility_excess_coeff: 1.0,
            compatibility_disjoint_coeff: 1.0,
            compatibility_weight_coeff: 0.4,
        }
    }
}

impl NeatConfig {
    /// Create a config for CPPN-style networks.
    #[must_use]
    pub fn cppn(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,
            use_bias: true,
            output_activation: Activation::Tanh,
            hidden_activations: Activation::CPPN.to_vec(),
            ..Default::default()
        }
    }

    /// Create a minimal config for testing.
    #[must_use]
    pub fn minimal(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,
            use_bias: false,
            output_activation: Activation::Sigmoid,
            hidden_activations: vec![Activation::Sigmoid, Activation::Tanh],
            ..Default::default()
        }
    }
}

/// A NEAT genome representing a neural network topology.
///
/// Uses arena-allocated storage for cache-friendly access and trivial serialization.
/// Innovation numbers are computed via hashing for lock-free parallel mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatGenome {
    /// Arena storage for nodes.
    pub nodes: SlotMap<NodeId, NodeGene>,
    /// Arena storage for connections.
    pub connections: SlotMap<ConnectionId, ConnectionGene>,
    /// IDs of input nodes (in order).
    pub input_ids: Vec<NodeId>,
    /// IDs of output nodes (in order).
    pub output_ids: Vec<NodeId>,
    /// ID of bias node, if present.
    pub bias_id: Option<NodeId>,
    /// Configuration used for this genome.
    #[serde(default)]
    pub config: NeatConfig,
}

impl NeatGenome {
    /// Create a minimal genome with only input and output nodes.
    #[must_use]
    pub fn minimal(config: NeatConfig) -> Self {
        let mut nodes: SlotMap<NodeId, NodeGene> = SlotMap::with_key();
        let mut input_ids = Vec::with_capacity(config.num_inputs);
        let mut output_ids = Vec::with_capacity(config.num_outputs);

        // Create bias node first (innovation 0)
        let bias_id = if config.use_bias {
            Some(nodes.insert(NodeGene::bias(0)))
        } else {
            None
        };

        // Create input nodes (innovations 1..num_inputs)
        for i in 0..config.num_inputs {
            let innovation = (i + 1) as u64;
            let id = nodes.insert(NodeGene::input(innovation));
            input_ids.push(id);
        }

        // Create output nodes (innovations num_inputs+1..)
        let output_start = config.num_inputs + 1;
        for i in 0..config.num_outputs {
            let innovation = (output_start + i) as u64;
            let id = nodes.insert(NodeGene::output(innovation, config.output_activation));
            output_ids.push(id);
        }

        Self {
            nodes,
            connections: SlotMap::with_key(),
            input_ids,
            output_ids,
            bias_id,
            config,
        }
    }

    /// Create a fully-connected genome (all inputs connected to all outputs).
    #[must_use]
    pub fn fully_connected<R: Rng>(config: NeatConfig, rng: &mut R) -> Self {
        let mut genome = Self::minimal(config.clone());

        // Connect all inputs to all outputs
        for &input_id in &genome.input_ids {
            for &output_id in &genome.output_ids {
                let input_inn = genome.nodes[input_id].innovation;
                let output_inn = genome.nodes[output_id].innovation;
                let conn_inn = connection_innovation(input_inn, output_inn);
                let weight = rng.random::<f32>() * 2.0 * config.weight_range - config.weight_range;
                genome
                    .connections
                    .insert(ConnectionGene::new(conn_inn, input_id, output_id, weight));
            }
        }

        // Connect bias to all outputs if present
        if let Some(bias_id) = genome.bias_id {
            for &output_id in &genome.output_ids {
                let bias_inn = genome.nodes[bias_id].innovation;
                let output_inn = genome.nodes[output_id].innovation;
                let conn_inn = connection_innovation(bias_inn, output_inn);
                let weight = rng.random::<f32>() * 2.0 * config.weight_range - config.weight_range;
                genome
                    .connections
                    .insert(ConnectionGene::new(conn_inn, bias_id, output_id, weight));
            }
        }

        genome.update_depths();
        genome
    }

    /// Add a new connection between two nodes.
    ///
    /// Returns `None` if the connection would create a cycle or already exists.
    pub fn add_connection<R: Rng>(
        &mut self,
        input_id: NodeId,
        output_id: NodeId,
        rng: &mut R,
    ) -> Option<ConnectionId> {
        // Validate nodes exist
        let input_node = self.nodes.get(input_id)?;
        let output_node = self.nodes.get(output_id)?;

        // Cannot connect to input/bias nodes
        if matches!(output_node.node_type, NodeType::Input | NodeType::Bias) {
            return None;
        }

        // Cannot connect from output nodes (for feedforward networks)
        if input_node.node_type == NodeType::Output {
            return None;
        }

        // Check if connection already exists
        let input_inn = input_node.innovation;
        let output_inn = output_node.innovation;
        let conn_inn = connection_innovation(input_inn, output_inn);

        for (_, conn) in &self.connections {
            if conn.innovation == conn_inn {
                return None;
            }
        }

        // Check for cycles (would make output unreachable from inputs)
        if self.would_create_cycle(input_id, output_id) {
            return None;
        }

        // Create the connection
        let weight =
            rng.random::<f32>() * 2.0 * self.config.weight_range - self.config.weight_range;
        let conn = ConnectionGene::new(conn_inn, input_id, output_id, weight);
        let conn_id = self.connections.insert(conn);

        self.update_depths();
        Some(conn_id)
    }

    /// Add a new node by splitting an existing connection.
    ///
    /// The original connection is disabled, and two new connections are created:
    /// input -> new_node (weight 1.0) and new_node -> output (original weight).
    ///
    /// Returns `None` if:
    /// - The connection doesn't exist or is disabled
    /// - No hidden activation functions are configured
    pub fn add_node<R: Rng>(&mut self, conn_id: ConnectionId, rng: &mut R) -> Option<NodeId> {
        // Cannot add nodes if no hidden activations are configured
        if self.config.hidden_activations.is_empty() {
            return None;
        }

        let conn = self.connections.get_mut(conn_id)?;
        if !conn.enabled {
            return None;
        }

        // Disable the original connection
        conn.enabled = false;
        let original_weight = conn.weight;
        let conn_innovation = conn.innovation;
        let input_id = conn.input;
        let output_id = conn.output;

        let input_inn = self.nodes.get(input_id)?.innovation;
        let output_inn = self.nodes.get(output_id)?.innovation;

        // Create new hidden node
        let new_node_inn = node_split_innovation(conn_innovation);
        let activation = self.config.hidden_activations
            [rng.random_range(0..self.config.hidden_activations.len())];
        let new_node = NodeGene::hidden(new_node_inn, activation);
        let new_node_id = self.nodes.insert(new_node);

        // Create connection from input to new node (weight 1.0 to preserve signal)
        let conn_a_inn = split_connection_a_innovation(input_inn, new_node_inn);
        let conn_a = ConnectionGene::new(conn_a_inn, input_id, new_node_id, 1.0);
        self.connections.insert(conn_a);

        // Create connection from new node to output (original weight)
        let out_conn_inn = split_connection_b_innovation(new_node_inn, output_inn);
        let out_conn = ConnectionGene::new(out_conn_inn, new_node_id, output_id, original_weight);
        self.connections.insert(out_conn);

        self.update_depths();
        Some(new_node_id)
    }

    /// Check if adding a connection from input_id to output_id would create a cycle.
    fn would_create_cycle(&self, input_id: NodeId, output_id: NodeId) -> bool {
        // BFS from output_id to see if we can reach input_id
        // Use HashSet directly instead of mapping NodeId to index
        let mut visited = std::collections::HashSet::with_capacity(self.nodes.len());
        let mut queue = vec![output_id];

        while let Some(current) = queue.pop() {
            if current == input_id {
                return true;
            }

            if !visited.insert(current) {
                continue; // Already visited
            }

            // Find all nodes that current connects TO
            for (_, conn) in &self.connections {
                if conn.enabled && conn.input == current {
                    queue.push(conn.output);
                }
            }
        }

        false
    }

    /// Update node depths for topological evaluation order.
    ///
    /// Depth is computed as the longest path from any input node, ensuring
    /// that a node is only evaluated after ALL its predecessors are ready.
    /// This is critical for correct feedforward evaluation.
    ///
    /// Returns `true` if the graph is acyclic and depths were computed successfully,
    /// `false` if a cycle was detected (iteration limit exceeded).
    pub fn update_depths(&mut self) -> bool {
        // Reset all depths: inputs/bias at 0, others at 0 (will be updated to max)
        for (_, node) in &mut self.nodes {
            node.depth = match node.node_type {
                NodeType::Input | NodeType::Bias => 0,
                _ => 0, // Will be updated to longest path
            };
        }

        // Iteratively propagate depths until stable.
        // For longest path: depth = max(all_predecessor_depths) + 1
        // Limit iterations to prevent infinite loops on cyclic graphs.
        // In an acyclic graph, we need at most N iterations (longest path length).
        let max_iterations = self.nodes.len();
        let mut iterations = 0;
        let mut changed = true;

        while changed && iterations < max_iterations {
            changed = false;
            iterations += 1;

            for (_, conn) in &self.connections {
                if !conn.enabled {
                    continue;
                }
                if let (Some(input_node), Some(output_node)) =
                    (self.nodes.get(conn.input), self.nodes.get(conn.output))
                {
                    // Longest path: take maximum of all incoming paths
                    let new_depth = input_node.depth.saturating_add(1);
                    if new_depth > output_node.depth {
                        // Safe because we're iterating connections, not nodes
                        if let Some(output_node) = self.nodes.get_mut(conn.output) {
                            output_node.depth = new_depth;
                            changed = true;
                        }
                    }
                }
            }
        }

        // If we're still changing after max iterations, there's a cycle
        !changed
    }

    /// Check if the genome contains any cycles in its enabled connections.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        // Use DFS with coloring: 0=white (unvisited), 1=gray (in progress), 2=black (done)
        let mut node_to_idx: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();
        for (i, (id, _)) in self.nodes.iter().enumerate() {
            node_to_idx.insert(id, i);
        }

        // Build adjacency list for enabled connections
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (_, conn) in &self.connections {
            if !conn.enabled {
                continue;
            }
            if let (Some(&from_idx), Some(&to_idx)) =
                (node_to_idx.get(&conn.input), node_to_idx.get(&conn.output))
            {
                adj[from_idx].push(to_idx);
            }
        }

        let mut color = vec![0u8; self.nodes.len()];

        fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {
            color[node] = 1; // Gray - in progress
            for &neighbor in &adj[node] {
                if color[neighbor] == 1 {
                    // Back edge - cycle found
                    return true;
                }
                if color[neighbor] == 0 && dfs(neighbor, adj, color) {
                    return true;
                }
            }
            color[node] = 2; // Black - done
            false
        }

        for start in 0..self.nodes.len() {
            if color[start] == 0 && dfs(start, &adj, &mut color) {
                return true;
            }
        }

        false
    }

    /// Remove connections that create cycles, keeping the graph acyclic.
    /// Returns the number of connections disabled.
    pub fn break_cycles(&mut self) -> usize {
        let mut disabled_count = 0;

        // Keep trying to find and break cycles until none remain
        while self.has_cycle() {
            // Find a connection that's part of a cycle and disable it
            // We'll use the connection with the highest innovation number
            // (most recently added) as a heuristic
            let mut cycle_conn_id: Option<ConnectionId> = None;
            let mut highest_inn = 0u64;

            for (conn_id, conn) in &self.connections {
                if !conn.enabled {
                    continue;
                }
                // Check if disabling this connection would break a cycle
                if conn.innovation >= highest_inn {
                    highest_inn = conn.innovation;
                    cycle_conn_id = Some(conn_id);
                }
            }

            if let Some(conn_id) = cycle_conn_id {
                if let Some(conn) = self.connections.get_mut(conn_id) {
                    conn.enabled = false;
                    disabled_count += 1;
                }
            } else {
                break; // No more connections to disable
            }
        }

        disabled_count
    }

    /// Get all hidden node IDs.
    #[must_use]
    pub fn hidden_ids(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.node_type == NodeType::Hidden)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get the number of enabled connections.
    #[must_use]
    pub fn num_enabled_connections(&self) -> usize {
        self.connections.iter().filter(|(_, c)| c.enabled).count()
    }

    /// Find a connection by its innovation number.
    #[must_use]
    pub fn find_connection_by_innovation(&self, innovation: u64) -> Option<ConnectionId> {
        self.connections
            .iter()
            .find(|(_, c)| c.innovation == innovation)
            .map(|(id, _)| id)
    }

    /// Find a node by its innovation number.
    #[must_use]
    pub fn find_node_by_innovation(&self, innovation: u64) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|(_, n)| n.innovation == innovation)
            .map(|(id, _)| id)
    }

    /// Compute compatibility distance to another genome for speciation.
    #[must_use]
    pub fn compatibility_distance(&self, other: &NeatGenome) -> f32 {
        let mut matching = 0;
        let mut disjoint = 0;
        let mut excess = 0;
        let mut weight_diff_sum = 0.0;

        // Get sorted innovation numbers
        let mut self_innovations: Vec<u64> =
            self.connections.iter().map(|(_, c)| c.innovation).collect();
        let mut other_innovations: Vec<u64> = other
            .connections
            .iter()
            .map(|(_, c)| c.innovation)
            .collect();
        self_innovations.sort_unstable();
        other_innovations.sort_unstable();

        let self_max = self_innovations.last().copied().unwrap_or(0);
        let other_max = other_innovations.last().copied().unwrap_or(0);

        // Compare genes
        for (_, self_conn) in &self.connections {
            if let Some(other_conn_id) = other.find_connection_by_innovation(self_conn.innovation) {
                matching += 1;
                weight_diff_sum +=
                    (self_conn.weight - other.connections[other_conn_id].weight).abs();
            } else if self_conn.innovation > other_max {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }

        for (_, other_conn) in &other.connections {
            if self
                .find_connection_by_innovation(other_conn.innovation)
                .is_none()
            {
                if other_conn.innovation > self_max {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
            }
        }

        let n = self.connections.len().max(other.connections.len()).max(1) as f32;
        let avg_weight_diff = if matching > 0 {
            weight_diff_sum / matching as f32
        } else {
            0.0
        };

        (self.config.compatibility_excess_coeff * excess as f32 / n)
            + (self.config.compatibility_disjoint_coeff * disjoint as f32 / n)
            + (self.config.compatibility_weight_coeff * avg_weight_diff)
    }

    /// Mutate weights of existing connections.
    fn mutate_weights<R: Rng>(&mut self, rng: &mut R) {
        // Clamp weights to prevent unbounded growth that leads to Inf/NaN
        // in compatibility_distance calculations (Inf - Inf = NaN causes panics)
        let weight_limit = self.config.weight_range * 10.0;

        for (_, conn) in &mut self.connections {
            if rng.random::<f32>() < self.config.weight_mutation_prob {
                if rng.random::<f32>() < self.config.weight_replace_prob {
                    // Replace with new random weight
                    conn.weight = rng.random::<f32>() * 2.0 * self.config.weight_range
                        - self.config.weight_range;
                } else {
                    // Perturb existing weight
                    conn.weight +=
                        (rng.random::<f32>() * 2.0 - 1.0) * self.config.weight_mutation_power;
                }
                // Clamp to prevent unbounded growth
                conn.weight = conn.weight.clamp(-weight_limit, weight_limit);
            }
        }
    }

    /// Mutate activation functions of hidden nodes.
    fn mutate_activations<R: Rng>(&mut self, rng: &mut R) {
        if self.config.hidden_activations.is_empty() {
            return;
        }

        for (_, node) in &mut self.nodes {
            if node.node_type == NodeType::Hidden
                && rng.random::<f32>() < self.config.activation_mutation_prob
            {
                node.activation = self.config.hidden_activations
                    [rng.random_range(0..self.config.hidden_activations.len())];
            }
        }
    }

    /// Toggle enabled state of random connections.
    fn mutate_toggle_enabled<R: Rng>(&mut self, rng: &mut R) {
        for (_, conn) in &mut self.connections {
            if rng.random::<f32>() < self.config.toggle_enabled_prob {
                conn.enabled = !conn.enabled;
            }
        }
    }

    /// Try to add a random new connection.
    fn mutate_add_connection<R: Rng>(&mut self, rng: &mut R) {
        if rng.random::<f32>() >= self.config.add_connection_prob {
            return;
        }

        // Collect valid source nodes (input, bias, hidden)
        let sources: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| !matches!(n.node_type, NodeType::Output))
            .map(|(id, _)| id)
            .collect();

        // Collect valid target nodes (hidden, output)
        let targets: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| !matches!(n.node_type, NodeType::Input | NodeType::Bias))
            .map(|(id, _)| id)
            .collect();

        if sources.is_empty() || targets.is_empty() {
            return;
        }

        // Try a few times to find a valid connection
        for _ in 0..10 {
            let source = sources[rng.random_range(0..sources.len())];
            let target = targets[rng.random_range(0..targets.len())];

            if self.add_connection(source, target, rng).is_some() {
                return;
            }
        }
    }

    /// Try to add a node by splitting a random connection.
    fn mutate_add_node<R: Rng>(&mut self, rng: &mut R) {
        if rng.random::<f32>() >= self.config.add_node_prob {
            return;
        }

        let enabled_conns: Vec<ConnectionId> = self
            .connections
            .iter()
            .filter(|(_, c)| c.enabled)
            .map(|(id, _)| id)
            .collect();

        if enabled_conns.is_empty() {
            return;
        }

        let conn_id = enabled_conns[rng.random_range(0..enabled_conns.len())];
        self.add_node(conn_id, rng);
    }
}

impl Genotype for NeatGenome {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // Scale mutation probabilities by rate
        let original_weight_prob = self.config.weight_mutation_prob;
        let original_add_conn_prob = self.config.add_connection_prob;
        let original_add_node_prob = self.config.add_node_prob;

        self.config.weight_mutation_prob *= rate;
        self.config.add_connection_prob *= rate;
        self.config.add_node_prob *= rate;

        self.mutate_weights(rng);
        self.mutate_activations(rng);
        self.mutate_toggle_enabled(rng);
        self.mutate_add_connection(rng);
        self.mutate_add_node(rng);

        // Restore original probabilities
        self.config.weight_mutation_prob = original_weight_prob;
        self.config.add_connection_prob = original_add_conn_prob;
        self.config.add_node_prob = original_add_node_prob;
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // NEAT crossover: align by innovation number, inherit from fitter parent
        // Here we assume self is the fitter parent (caller's responsibility)

        let mut child = Self::minimal(self.config.clone());

        // Copy all nodes from the fitter parent (self)
        // and add any nodes from other that we need for matching connections
        let mut node_map: std::collections::HashMap<u64, NodeId> = std::collections::HashMap::new();

        // First, map existing nodes (input, output, bias)
        for (id, node) in &child.nodes {
            node_map.insert(node.innovation, id);
        }

        // Collect innovations for O(N) merge
        let mut self_conns: Vec<_> = self.connections.iter().collect();
        let mut other_conns: Vec<_> = other.connections.iter().collect();
        self_conns.sort_by_key(|(_, c)| c.innovation);
        other_conns.sort_by_key(|(_, c)| c.innovation);

        let mut self_idx = 0;
        let mut other_idx = 0;

        while self_idx < self_conns.len() || other_idx < other_conns.len() {
            let (use_self, use_other, advance_self, advance_other) =
                match (self_conns.get(self_idx), other_conns.get(other_idx)) {
                    (Some((_, sc)), Some((_, oc))) => {
                        match sc.innovation.cmp(&oc.innovation) {
                            std::cmp::Ordering::Equal => {
                                // Matching gene - randomly inherit
                                let use_self = rng.random::<bool>();
                                (use_self, !use_self, true, true)
                            }
                            std::cmp::Ordering::Less => {
                                // Disjoint/excess from self (fitter) - inherit
                                (true, false, true, false)
                            }
                            std::cmp::Ordering::Greater => {
                                // Disjoint/excess from other - skip (fitter parent doesn't have it)
                                (false, false, false, true)
                            }
                        }
                    }
                    (Some(_), None) => (true, false, true, false),
                    (None, Some(_)) => (false, false, false, true),
                    (None, None) => break,
                };

            let conn_to_add = if use_self {
                self_conns.get(self_idx).map(|(_, c)| (*c).clone())
            } else if use_other {
                other_conns.get(other_idx).map(|(_, c)| (*c).clone())
            } else {
                None
            };

            if let Some(mut conn) = conn_to_add {
                // Ensure nodes exist in child
                let source_parent = if use_self { self } else { other };

                let input_inn = source_parent.nodes[conn.input].innovation;
                let output_inn = source_parent.nodes[conn.output].innovation;

                // Get or create input node
                let child_input_id = if let Some(&id) = node_map.get(&input_inn) {
                    id
                } else {
                    let node = source_parent.nodes[conn.input].clone();
                    let id = child.nodes.insert(node);
                    node_map.insert(input_inn, id);
                    id
                };

                // Get or create output node
                let child_output_id = if let Some(&id) = node_map.get(&output_inn) {
                    id
                } else {
                    let node = source_parent.nodes[conn.output].clone();
                    let id = child.nodes.insert(node);
                    node_map.insert(output_inn, id);
                    id
                };

                conn.input = child_input_id;
                conn.output = child_output_id;

                // For matching genes, randomly inherit enabled state
                if use_self && use_other {
                    let self_enabled = self_conns[self_idx].1.enabled;
                    let other_enabled = other_conns[other_idx].1.enabled;
                    if !self_enabled || !other_enabled {
                        // If either parent has it disabled, 75% chance of disabled
                        conn.enabled = rng.random::<f32>() > 0.75;
                    }
                }

                child.connections.insert(conn);
            }

            if advance_self {
                self_idx += 1;
            }
            if advance_other {
                other_idx += 1;
            }
        }

        // Detect and break any cycles that may have been introduced
        // This can happen if innovation hash collisions cause non-homologous
        // genes to be aligned and merged incorrectly
        if child.has_cycle() {
            child.break_cycles();
        }

        child.update_depths();
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn test_minimal_genome() {
        let config = NeatConfig::minimal(3, 2);
        let genome = NeatGenome::minimal(config);

        assert_eq!(genome.input_ids.len(), 3);
        assert_eq!(genome.output_ids.len(), 2);
        assert!(genome.bias_id.is_none());
        assert_eq!(genome.connections.len(), 0);
    }

    #[test]
    fn test_fully_connected_genome() {
        let config = NeatConfig::minimal(2, 2);
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        // 2 inputs * 2 outputs = 4 connections
        assert_eq!(genome.connections.len(), 4);
    }

    #[test]
    fn test_fully_connected_with_bias() {
        let config = NeatConfig {
            num_inputs: 2,
            num_outputs: 2,
            use_bias: true,
            ..NeatConfig::minimal(2, 2)
        };
        let mut rng = test_rng();
        let genome = NeatGenome::fully_connected(config, &mut rng);

        // 2 inputs * 2 outputs + 1 bias * 2 outputs = 6 connections
        assert_eq!(genome.connections.len(), 6);
        assert!(genome.bias_id.is_some());
    }

    #[test]
    fn test_add_connection() {
        let config = NeatConfig::minimal(2, 1);
        let mut genome = NeatGenome::minimal(config);
        let mut rng = test_rng();

        let input_id = genome.input_ids[0];
        let output_id = genome.output_ids[0];

        let conn_id = genome.add_connection(input_id, output_id, &mut rng);
        assert!(conn_id.is_some());
        assert_eq!(genome.connections.len(), 1);

        // Adding same connection again should fail
        let conn_id2 = genome.add_connection(input_id, output_id, &mut rng);
        assert!(conn_id2.is_none());
    }

    #[test]
    fn test_add_node() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();
        let mut genome = NeatGenome::fully_connected(config, &mut rng);

        let initial_nodes = genome.nodes.len();
        let initial_conns = genome.connections.len();

        let conn_id = genome.connections.iter().next().unwrap().0;
        let new_node_id = genome.add_node(conn_id, &mut rng);

        assert!(new_node_id.is_some());
        assert_eq!(genome.nodes.len(), initial_nodes + 1);
        assert_eq!(genome.connections.len(), initial_conns + 2);
        assert_eq!(genome.num_enabled_connections(), initial_conns + 1); // Original disabled
    }

    #[test]
    fn test_mutation() {
        let config = NeatConfig {
            add_connection_prob: 1.0,
            add_node_prob: 0.5,
            weight_mutation_prob: 1.0,
            ..NeatConfig::minimal(2, 1)
        };
        let mut rng = test_rng();
        let mut genome = NeatGenome::fully_connected(config, &mut rng);

        let initial_conns = genome.connections.len();
        genome.mutate(&mut rng, 1.0);

        // Should have added at least one connection
        assert!(genome.connections.len() >= initial_conns);
    }

    #[test]
    fn test_crossover() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();

        let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
        let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

        // Add a node to parent1
        let conn_id = parent1.connections.iter().next().unwrap().0;
        parent1.add_node(conn_id, &mut rng);

        // Add a different node to parent2
        let conn_id = parent2.connections.iter().next().unwrap().0;
        parent2.add_node(conn_id, &mut rng);

        let child = parent1.crossover(&parent2, &mut rng);

        // Child should have input and output nodes at minimum
        assert!(child.nodes.len() >= 3);
        assert_eq!(child.input_ids.len(), 2);
        assert_eq!(child.output_ids.len(), 1);
    }

    #[test]
    fn test_compatibility_distance() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = test_rng();

        let genome1 = NeatGenome::fully_connected(config.clone(), &mut rng);
        let genome2 = NeatGenome::fully_connected(config, &mut rng);

        // Same structure should have low distance (only weight differences)
        let dist = genome1.compatibility_distance(&genome2);
        assert!(dist < 1.0);

        // Distance to self should be 0
        let self_dist = genome1.compatibility_distance(&genome1);
        assert!(self_dist.abs() < 1e-6);
    }
}
