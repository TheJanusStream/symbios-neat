# Symbios NEAT

A high-performance NeuroEvolution of Augmenting Topologies (NEAT) engine for morphogenetic engineering applications in Rust.

## Features

- **Hash-Based Innovation**: Lock-free, deterministic parallel mutation using `Hash(input_node, output_node)` instead of global counters
- **Arena-Graph Model**: Cache-friendly `SlotMap` storage for nodes and connections with generational indices
- **CPPN Support**: Periodic and radial activation functions (Sine, Cosine, Gaussian, Abs) for Compositional Pattern Producing Networks
- **Genotype Trait**: Implements `symbios_genetics::Genotype` for seamless integration with evolutionary algorithms
- **Serde Support**: Full serialization/deserialization of genomes for checkpointing and analysis

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
symbios-neat = "0.1.0"
```

## Quick Start

```rust
use symbios_neat::{NeatGenome, NeatConfig, CppnEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// Create a CPPN for 2D pattern generation
let config = NeatConfig::cppn(2, 1);
let mut rng = ChaCha8Rng::seed_from_u64(42);
let genome = NeatGenome::fully_connected(config, &mut rng);

// Compile and evaluate. `new` returns `Result<_, EvaluatorError>` —
// it errors on cyclic genomes; call `genome.break_cycles()` first if needed.
let evaluator = CppnEvaluator::new(&genome).expect("acyclic genome");
let output = evaluator.query_2d(0.5, -0.5);
println!("Output: {:?}", output);
```

## Using with Symbios Genetics

```rust
use symbios_genetics::{Evaluator, Evolver, algorithms::simple::SimpleGA};
use symbios_neat::{NeatGenome, NeatConfig, CppnEvaluator};

// Define fitness function
struct XorFitness;

impl Evaluator<NeatGenome> for XorFitness {
    fn evaluate(&self, genome: &NeatGenome) -> (f32, Vec<f32>, Vec<f32>) {
        // CppnEvaluator::new errors on cyclic genomes; assign worst fitness in that case.
        let Ok(eval) = CppnEvaluator::new(genome) else {
            return (0.0, vec![0.0], vec![]);
        };
        let mut error = 0.0;

        // XOR truth table
        for (inputs, expected) in &[
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ] {
            let output = eval.evaluate(inputs)[0];
            error += (output - expected).powi(2);
        }

        let fitness = 4.0 - error;
        (fitness, vec![fitness], vec![])
    }
}

// Run evolution
let config = NeatConfig::minimal(2, 1);
let mut rng = rand::rng();
let initial: Vec<NeatGenome> = (0..100)
    .map(|_| NeatGenome::fully_connected(config.clone(), &mut rng))
    .collect();

let mut ga = SimpleGA::new(initial, 0.3, 5, 42);
for _ in 0..100 {
    ga.step(&XorFitness);
}
```

## Pattern, Voxel, and Image Export

A trained CPPN can be queried over a grid of coordinates to produce 2D
patterns or 3D voxel grids:

```rust,ignore
let evaluator = CppnEvaluator::new(&genome).expect("acyclic genome");

// 2D pattern: width x height pixels in [-1, 1]^2, output normalized to [0, 1].
let pattern: Vec<f32> = evaluator.generate_pattern_2d(64, 64, 0).unwrap();

// 3D voxel grid: (w, h, d) with x-fastest, then y, then z (z slowest).
let voxels: Vec<f32> = evaluator.generate_voxel_grid([32, 32, 32], 0).unwrap();
```

Both methods return `Err(PatternError::InputArityMismatch)` if the CPPN's
input arity does not match the pattern's dimensionality (2 inputs for 2D,
3 for 3D), and `Err(PatternError::OutputIndexOutOfBounds)` if the output
index is too large.

### `image` feature

PNG export via the [`image`](https://crates.io/crates/image) crate is gated
behind the `image` Cargo feature so that consumers who don't need it pay
no compile-time cost:

```toml
[dependencies]
symbios-neat = { version = "...", features = ["image"] }
```

With the feature on:

```rust,ignore
# #[cfg(feature = "image")]
let img: image::RgbaImage = evaluator.generate_image(256, 256, 0).unwrap();
img.save("cppn_pattern.png").unwrap();
```

## HyperNEAT Substrates

For indirect encoding, evolve a CPPN that emits weights for a fixed
substrate of nodes. See `examples/hyperneat_xor.rs` for a complete run.

```rust,ignore
use symbios_neat::{
    substrate::{substrate_to_network, LayeredSubstrate},
    Activation, CppnEvaluator,
};

let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid);
let cppn = CppnEvaluator::new(&cppn_genome).unwrap();
let network = substrate_to_network(&cppn, &substrate, 0.2);
let outputs = network.evaluate(&[0.5, -0.3]);
```

## Architecture

### Hash-Based Innovation (Sovereign Innovation)

Traditional NEAT uses a global innovation counter requiring synchronization across threads. This crate uses deterministic hashing:

- **Connections**: `innovation = Hash(input_node_innovation, output_node_innovation)`
- **Nodes (from split)**: `innovation = Hash(connection_innovation, SPLIT_MARKER)`

This enables lock-free parallel mutation without coordination, while maintaining alignment for crossover.

### Arena-Graph Model

Nodes and connections are stored in flat `SlotMap` buffers:

- No reference counting overhead
- Cache-friendly memory layout
- Trivially serializable via Serde
- Safe generational indices prevent use-after-free

### Activation Functions

| Function  | Formula             | Use Case                       |
|---------- |---------------------|--------------------------------|
| Sigmoid   | `1 / (1 + e^(-x))`  | Classification, bounded output |
| Tanh      | `tanh(x)`           | Centered output [-1, 1]        |
| ReLU      | `max(0, x)`         | Hidden layers                  |
| Sine      | `sin(x)`            | Periodic/wave patterns (CPPN)  |
| Cosine    | `cos(x)`            | Periodic/wave patterns (CPPN)  |
| Gaussian  | `e^(-x^2)`          | Radial patterns (CPPN)         |
| Abs       | `\|x\|`             | Symmetric patterns (CPPN)      |
| Step      | `x >= 0 ? 1 : 0`    | Binary decisions               |
| LeakyReLU | `x > 0 ? x : 0.01x` | Hidden layers                  |
| Identity  | `x`                 | Pass-through                   |

## Configuration

```rust
let config = NeatConfig {
    num_inputs: 2,
    num_outputs: 1,
    use_bias: true,
    output_activation: Activation::Sigmoid,
    hidden_activations: Activation::CPPN.to_vec(),

    // Mutation probabilities
    add_connection_prob: 0.05,
    add_node_prob: 0.03,
    weight_mutation_prob: 0.8,
    weight_mutation_power: 0.5,
    weight_replace_prob: 0.1,
    weight_range: 1.0,
    toggle_enabled_prob: 0.01,
    activation_mutation_prob: 0.1,

    // Speciation coefficients
    compatibility_excess_coeff: 1.0,
    compatibility_disjoint_coeff: 1.0,
    compatibility_weight_coeff: 0.4,
};
```

Convenience constructors:

- `NeatConfig::cppn(inputs, outputs)` - CPPN with Tanh output and CPPN activation set
- `NeatConfig::minimal(inputs, outputs)` - Minimal config for testing

## API Reference

### NeatGenome

```rust
// Creation
NeatGenome::minimal(config)           // Input/output nodes only
NeatGenome::fully_connected(config, rng)  // All inputs connected to outputs

// Mutation
genome.add_connection(input_id, output_id, rng)  // Add a new connection
genome.add_node(connection_id, rng)              // Split a connection with a new node

// Topology
genome.hidden_ids()                   // Get all hidden node IDs
genome.num_enabled_connections()      // Count active connections
genome.has_cycle()                    // Check for cycles
genome.break_cycles()                 // Remove cycle-causing connections
genome.update_depths()                // Recompute topological depths

// Speciation
genome.compatibility_distance(&other) // Compute genetic distance
```

### CppnEvaluator

```rust
// Construction
CppnEvaluator::new(&genome)           // Panics on cyclic genome
CppnEvaluator::try_new(&genome)       // Returns Result

// Evaluation
evaluator.evaluate(&[x, y])           // General evaluation
evaluator.evaluate_into(&inputs, &mut outputs)  // Allocation-free
evaluator.query_2d(x, y)              // 2D coordinates
evaluator.query_3d(x, y, z)           // 3D coordinates
evaluator.query_2d_with_distance(x, y)  // [x, y, sqrt(x^2+y^2)]
evaluator.query_substrate(x1, y1, x2, y2)  // HyperNEAT-style

// Info
evaluator.num_inputs()
evaluator.num_outputs()
```

### Pattern Generation

```rust
use symbios_neat::generate_pattern;

let pattern = generate_pattern(&mut evaluator, width, height, output_index)?;
// Returns Vec<f32> with values in [0, 1]
```

## Examples

Run the XOR example:

```bash
cargo run --example xor
```

## Benchmarks

```bash
cargo bench
```

## Testing

```bash
cargo test
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
- Stanley, K. O. (2007). Compositional Pattern Producing Networks: A Novel Abstraction of Development. *Genetic Programming and Evolvable Machines*, 8(2), 131-162.
