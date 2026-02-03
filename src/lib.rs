//! # Symbios NEAT
//!
//! A high-performance `NeuroEvolution` of Augmenting Topologies (NEAT) engine
//! for morphogenetic engineering applications.
//!
//! ## Features
//!
//! - **Hash-Based Innovation**: Lock-free, deterministic parallel mutation using
//!   `Hash(input_node, output_node)` instead of global counters
//! - **Arena-Graph Model**: Cache-friendly `SlotMap` storage for nodes and connections
//! - **CPPN Support**: Periodic and radial activation functions (Sine, Cosine, Gaussian, Abs)
//!   for Compositional Pattern Producing Networks
//! - **Genotype Trait**: Implements `symbios_genetics::Genotype` for use with evolutionary algorithms
//!
//! ## Quick Start
//!
//! ```rust
//! use symbios_neat::{NeatGenome, NeatConfig, CppnEvaluator};
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! // Create a CPPN for 2D pattern generation
//! let config = NeatConfig::cppn(2, 1);
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let genome = NeatGenome::fully_connected(config, &mut rng);
//!
//! // Compile and evaluate
//! let mut evaluator = CppnEvaluator::new(&genome);
//! let output = evaluator.query_2d(0.5, -0.5);
//! println!("Output: {:?}", output);
//! ```
//!
//! ## Using with Symbios Genetics
//!
//! ```rust,ignore
//! use symbios_genetics::{Evaluator, Evolver, algorithms::simple::SimpleGA};
//! use symbios_neat::{NeatGenome, NeatConfig, CppnEvaluator};
//!
//! // Define fitness function
//! struct XorFitness;
//! impl Evaluator<NeatGenome> for XorFitness {
//!     fn evaluate(&self, genome: &NeatGenome) -> (f32, Vec<f32>, Vec<f32>) {
//!         let mut eval = CppnEvaluator::new(genome);
//!         let mut error = 0.0;
//!
//!         // XOR truth table
//!         for (inputs, expected) in &[
//!             ([0.0, 0.0], 0.0),
//!             ([0.0, 1.0], 1.0),
//!             ([1.0, 0.0], 1.0),
//!             ([1.0, 1.0], 0.0),
//!         ] {
//!             let output = eval.evaluate(inputs)[0];
//!             error += (output - expected).powi(2);
//!         }
//!
//!         let fitness = 4.0 - error; // Max fitness = 4.0
//!         (fitness, vec![fitness], vec![])
//!     }
//! }
//!
//! // Run evolution
//! let config = NeatConfig::minimal(2, 1);
//! let mut rng = rand::rng();
//! let initial: Vec<NeatGenome> = (0..100)
//!     .map(|_| NeatGenome::fully_connected(config.clone(), &mut rng))
//!     .collect();
//!
//! let mut ga = SimpleGA::new(initial, 0.3, 5, 42);
//! for _ in 0..100 {
//!     ga.step(&XorFitness);
//! }
//! ```
//!
//! ## Architecture
//!
//! ### Hash-Based Innovation (Sovereign Innovation)
//!
//! Traditional NEAT uses a global innovation counter requiring synchronization.
//! This crate uses deterministic hashing:
//!
//! - **Connections**: `innovation = Hash(input_node_innovation, output_node_innovation)`
//! - **Nodes (from split)**: `innovation = Hash(connection_innovation, SPLIT_MARKER)`
//!
//! This enables lock-free parallel mutation across threads without coordination.
//!
//! ### Arena-Graph Model
//!
//! Nodes and connections are stored in flat `SlotMap` buffers:
//!
//! - No reference counting overhead
//! - Cache-friendly memory layout
//! - Trivially serializable via Serde
//! - Safe generational indices prevent use-after-free

pub mod activation;
pub mod evaluator;
pub mod gene;
pub mod genome;
pub mod innovation;
pub mod topology;

// Re-exports for convenience
pub use activation::Activation;
pub use evaluator::{generate_pattern, CppnEvaluator, EvaluatorError, PatternError};
pub use gene::{ConnectionGene, ConnectionId, NodeGene, NodeId, NodeType};
pub use genome::{NeatConfig, NeatGenome};
pub use innovation::{
    connection_innovation, node_split_innovation, split_connection_a_innovation,
    split_connection_b_innovation,
};
pub use topology::GraphTopology;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use symbios_genetics::Genotype;

    #[test]
    fn test_genotype_trait_implementation() {
        let config = NeatConfig::minimal(2, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut genome = NeatGenome::fully_connected(config, &mut rng);

        // Test mutation
        genome.mutate(&mut rng, 1.0);

        // Test crossover
        let mut genome2 = genome.clone();
        genome2.mutate(&mut rng, 1.0);

        let child = genome.crossover(&genome2, &mut rng);
        assert!(!child.input_ids.is_empty());
        assert!(!child.output_ids.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = NeatConfig::cppn(3, 2);
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut genome = NeatGenome::fully_connected(config, &mut rng);

        // Add some structure
        let conn_id = genome.connections.iter().next().unwrap().0;
        genome.add_node(conn_id, &mut rng);

        // Serialize
        let json = serde_json::to_string(&genome).expect("Serialization failed");

        // Deserialize
        let restored: NeatGenome = serde_json::from_str(&json).expect("Deserialization failed");

        // Verify structure preserved
        assert_eq!(genome.nodes.len(), restored.nodes.len());
        assert_eq!(genome.connections.len(), restored.connections.len());
        assert_eq!(genome.input_ids.len(), restored.input_ids.len());
        assert_eq!(genome.output_ids.len(), restored.output_ids.len());
    }

    #[test]
    fn test_cppn_pattern_generation() {
        let config = NeatConfig::cppn(2, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);
        let pattern = generate_pattern(&mut evaluator, 16, 16, 0).unwrap();

        assert_eq!(pattern.len(), 256);

        // Verify all values in valid range
        for &val in &pattern {
            assert!((0.0..=1.0).contains(&val), "Value out of range: {}", val);
        }
    }

    #[test]
    fn test_innovation_determinism() {
        // Same structural mutation should produce same innovation
        let inn1 = connection_innovation(1, 2);
        let inn2 = connection_innovation(1, 2);
        assert_eq!(inn1, inn2);

        let node_inn1 = node_split_innovation(100);
        let node_inn2 = node_split_innovation(100);
        assert_eq!(node_inn1, node_inn2);
    }
}
