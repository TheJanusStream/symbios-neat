//! HyperNEAT-XOR with painted biases, through a [`SubstratePainter`].
//!
//! The classic example (`hyperneat_xor`) paints only weights, and its
//! substrate's biases are all zero: for the inputs `(0, 0)` every neuron sums
//! to zero, so the sigmoid output is stuck at exactly 0.5 and XOR's first case
//! can never come out as 0. Here a painter asks the CPPN a second question
//! for every hidden and output neuron, its bias, at the neuron's own
//! position (source and target both set to it), and scales both answers so
//! the network can saturate.
//!
//! Run with: `cargo run --example hyperneat_xor_painted --release`

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symbios_genetics::{Evaluator, Evolver, algorithms::simple::SimpleGA};
use symbios_neat::{
    Activation, CppnEvaluator, FeedforwardNetwork, NeatConfig, NeatGenome, Substrate,
    substrate::{LayeredSubstrate, SubstratePainter, substrate_to_network_with, write_endpoints},
};

/// CPPN outputs: the link weight, then the neuron bias.
const WEIGHT: usize = 0;
const BIAS: usize = 1;
/// Weight outputs weaker than this leave the link out.
const EXPRESSION_THRESHOLD: f32 = 0.2;
/// Painted weights and biases are the CPPN's `tanh` outputs times these, so
/// the sigmoid output can reach close to 0 and 1.
const WEIGHT_SCALE: f32 = 5.0;
const BIAS_SCALE: f32 = 5.0;

const CASES: [([f32; 2], f32); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

/// Paints weights from output 0 (thresholded) and biases from output 1.
struct BiasPainter;

impl SubstratePainter<LayeredSubstrate> for BiasPainter {
    fn link_query(&self, substrate: &LayeredSubstrate, link: usize, query: &mut [f32]) {
        write_endpoints(substrate, link, query);
    }

    fn link_weight(&self, _substrate: &LayeredSubstrate, _link: usize, out: &[f32]) -> Option<f32> {
        let weight = out[WEIGHT];
        (weight.abs() >= EXPRESSION_THRESHOLD).then_some(weight * WEIGHT_SCALE)
    }

    fn bias_query(&self, substrate: &LayeredSubstrate, node: usize, query: &mut [f32]) -> bool {
        let at = &substrate.nodes()[node].coords;
        query[..2].copy_from_slice(at);
        query[2..4].copy_from_slice(at);
        true
    }

    fn bias(&self, _substrate: &LayeredSubstrate, _node: usize, out: &[f32]) -> f32 {
        out[BIAS] * BIAS_SCALE
    }
}

fn substrate() -> LayeredSubstrate {
    LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid)
}

fn paint(cppn: &CppnEvaluator, substrate: &LayeredSubstrate) -> FeedforwardNetwork {
    substrate_to_network_with(cppn, substrate, &BiasPainter)
}

/// Fitness: 4 minus the squared error over the four XOR cases.
struct HyperXor {
    substrate: LayeredSubstrate,
}

impl Evaluator<NeatGenome> for HyperXor {
    fn evaluate(&self, genome: &NeatGenome) -> (f32, Vec<f32>, Vec<f32>) {
        let Ok(cppn) = CppnEvaluator::new(genome) else {
            return (0.0, vec![0.0], vec![]);
        };
        let network = paint(&cppn, &self.substrate);
        let error: f32 = CASES
            .iter()
            .map(|(inputs, expected)| (network.evaluate(inputs)[0] - expected).powi(2))
            .sum();
        let fitness = 4.0 - error;
        (fitness, vec![fitness], vec![])
    }
}

fn main() {
    println!("HyperNEAT XOR with painted biases");
    println!("=================================\n");

    // CPPN: 4 inputs (src + tgt 2D coords), 2 outputs (weight, bias).
    let cppn_config = NeatConfig {
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        weight_mutation_prob: 0.8,
        weight_mutation_power: 0.7,
        ..NeatConfig::cppn(4, 2)
    };

    let population_size = 150;
    let generations = 100;
    let mutation_rate = 1.0;
    let elite_count = 5;
    let seed = 42;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let initial: Vec<NeatGenome> = (0..population_size)
        .map(|_| NeatGenome::fully_connected(cppn_config.clone(), &mut rng))
        .collect();
    let mut ga = SimpleGA::new(initial, mutation_rate, elite_count, seed);
    let evaluator = HyperXor {
        substrate: substrate(),
    };

    let mut solution_generation: Option<usize> = None;
    for gen_idx in 0..generations {
        ga.step(&evaluator);
        let best = ga
            .population()
            .iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
            .expect("a population");
        if best.fitness >= 3.9 && solution_generation.is_none() {
            solution_generation = Some(gen_idx);
        }
        if gen_idx % 10 == 0 || gen_idx == generations - 1 {
            println!("Gen {gen_idx:3}: best={:.4}", best.fitness);
        }
    }

    let champion = ga
        .population()
        .iter()
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .expect("a population");
    println!("\nBest CPPN fitness: {:.4}", champion.fitness);
    match solution_generation {
        Some(gen_idx) => println!("Solution (fitness >= 3.9) found at generation {gen_idx}"),
        None => println!("(No solution within {generations} generations.)"),
    }

    let cppn = CppnEvaluator::new(&champion.genotype).expect("champion is acyclic");
    let network = paint(&cppn, &substrate());
    println!(
        "\nChampion XOR outputs ({} links expressed):",
        network.num_connections()
    );
    for (inputs, expected) in CASES {
        let out = network.evaluate(&inputs)[0];
        let status = if (out - expected).abs() < 0.5 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  {} XOR {} = {out:.4} (expected {}) {status}",
            inputs[0] as i32, inputs[1] as i32, expected as i32
        );
    }
}
