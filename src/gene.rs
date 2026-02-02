//! Gene types for NEAT genomes.
//!
//! This module defines the fundamental building blocks of NEAT networks:
//! - [`NodeGene`]: Represents neurons in the network
//! - [`ConnectionGene`]: Represents weighted connections between nodes

use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

use crate::activation::Activation;

new_key_type! {
    /// Unique identifier for a node within a genome.
    ///
    /// Uses SlotMap's generational indices for safe, cache-friendly storage.
    pub struct NodeId;

    /// Unique identifier for a connection within a genome.
    pub struct ConnectionId;
}

/// The type/role of a node in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Input node - receives external values, no activation applied.
    Input,
    /// Output node - produces final network output.
    Output,
    /// Hidden node - internal processing node added through mutation.
    Hidden,
    /// Bias node - always outputs 1.0, used for learnable biases.
    Bias,
}

/// A node gene representing a neuron in the NEAT network.
///
/// Nodes are stored in a `SlotMap` arena for cache-friendly access and
/// safe indexing without reference counting overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGene {
    /// The innovation number for this node, computed as `Hash(in_node, out_node)`
    /// for nodes created by splitting connections.
    pub innovation: u64,
    /// The type/role of this node in the network.
    pub node_type: NodeType,
    /// The activation function applied to this node's input sum.
    pub activation: Activation,
    /// Bias value added before activation (for hidden/output nodes).
    pub bias: f32,
    /// Cached depth for topological sorting (0 = input layer).
    /// Recomputed when topology changes. Uses u32 to avoid saturation
    /// in adversarially deep networks (u16 would overflow at 65535 layers).
    pub depth: u32,
}

impl NodeGene {
    /// Create a new input node.
    #[must_use]
    pub fn input(innovation: u64) -> Self {
        Self {
            innovation,
            node_type: NodeType::Input,
            activation: Activation::Identity,
            bias: 0.0,
            depth: 0,
        }
    }

    /// Create a new output node.
    #[must_use]
    pub fn output(innovation: u64, activation: Activation) -> Self {
        Self {
            innovation,
            node_type: NodeType::Output,
            activation,
            bias: 0.0,
            depth: u32::MAX,
        }
    }

    /// Create a new hidden node.
    #[must_use]
    pub fn hidden(innovation: u64, activation: Activation) -> Self {
        Self {
            innovation,
            node_type: NodeType::Hidden,
            activation,
            bias: 0.0,
            depth: 0,
        }
    }

    /// Create a bias node that always outputs 1.0.
    #[must_use]
    pub fn bias(innovation: u64) -> Self {
        Self {
            innovation,
            node_type: NodeType::Bias,
            activation: Activation::Identity,
            bias: 1.0,
            depth: 0,
        }
    }
}

/// A connection gene representing a weighted link between two nodes.
///
/// Connections use hash-based innovation numbers computed from their
/// source and target nodes, enabling lock-free parallel mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    /// The innovation number, computed as Hash(input_node_innovation, output_node_innovation).
    /// This enables deterministic, lock-free parallel mutation.
    pub innovation: u64,
    /// The source node of this connection.
    pub input: NodeId,
    /// The target node of this connection.
    pub output: NodeId,
    /// The connection weight.
    pub weight: f32,
    /// Whether this connection is active.
    /// Disabled connections are skipped during evaluation but preserved for crossover.
    pub enabled: bool,
}

impl ConnectionGene {
    /// Create a new enabled connection.
    #[must_use]
    pub fn new(innovation: u64, input: NodeId, output: NodeId, weight: f32) -> Self {
        Self {
            innovation,
            input,
            output,
            weight,
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_gene_creation() {
        let input = NodeGene::input(1);
        assert_eq!(input.node_type, NodeType::Input);
        assert_eq!(input.depth, 0);

        let output = NodeGene::output(2, Activation::Sigmoid);
        assert_eq!(output.node_type, NodeType::Output);
        assert_eq!(output.activation, Activation::Sigmoid);

        let hidden = NodeGene::hidden(3, Activation::Tanh);
        assert_eq!(hidden.node_type, NodeType::Hidden);
        assert_eq!(hidden.activation, Activation::Tanh);

        let bias = NodeGene::bias(0);
        assert_eq!(bias.node_type, NodeType::Bias);
        assert!((bias.bias - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_connection_gene_creation() {
        use slotmap::SlotMap;

        let mut nodes: SlotMap<NodeId, NodeGene> = SlotMap::with_key();
        let n1 = nodes.insert(NodeGene::input(1));
        let n2 = nodes.insert(NodeGene::output(2, Activation::Sigmoid));

        let conn = ConnectionGene::new(100, n1, n2, 0.5);
        assert_eq!(conn.input, n1);
        assert_eq!(conn.output, n2);
        assert!((conn.weight - 0.5).abs() < 1e-6);
        assert!(conn.enabled);
    }
}
