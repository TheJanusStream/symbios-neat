//! HyperNEAT-XOR example.
//!
//! Demonstrates indirect encoding: instead of evolving the XOR-solver network
//! directly, we evolve a CPPN that, when queried with the substrate's
//! coordinate pairs, produces the weights of a feedforward network that
//! solves XOR.
//!
//! The substrate is a fixed `[2, 2, 1]` layered topology — 2 inputs, 2 hidden,
//! 1 output. The CPPN takes 4 inputs (src.x, src.y, tgt.x, tgt.y) and emits
//! one weight per query. Evolving the CPPN with [`SimpleGA`] discovers weight
//! patterns that, when expressed onto the substrate, solve XOR.
//!
//! Run with: `cargo run --example hyperneat_xor --release`

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symbios_genetics::{algorithms::simple::SimpleGA, Evaluator, Evolver};
use symbios_neat::{
    substrate::{substrate_to_network, LayeredSubstrate},
    Activation, CppnEvaluator, NeatConfig, NeatGenome, Substrate,
};

const EXPRESSION_THRESHOLD: f32 = 0.2;

/// Fitness: how well the substrate-derived network solves XOR.
struct HyperXor {
    substrate: LayeredSubstrate,
}

impl Evaluator<NeatGenome> for HyperXor {
    fn evaluate(&self, genome: &NeatGenome) -> (f32, Vec<f32>, Vec<f32>) {
        // Cyclic CPPN → worst fitness. Mutation can rarely produce cycles even
        // with feedforward guards, so handle defensively.
        let Ok(cppn) = CppnEvaluator::new(genome) else {
            return (0.0, vec![0.0], vec![]);
        };

        // Materialize the substrate against this CPPN.
        let network = substrate_to_network(&cppn, &self.substrate, EXPRESSION_THRESHOLD);

        let mut total_error = 0.0;
        for (inputs, expected) in &[
            ([0.0_f32, 0.0], 0.0_f32),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ] {
            let out = network.evaluate(inputs)[0];
            total_error += (out - expected).powi(2);
        }

        let fitness = 4.0 - total_error;
        (fitness, vec![fitness], vec![])
    }
}

fn main() {
    println!("HyperNEAT XOR Example");
    println!("=====================\n");

    // CPPN config: 4 inputs (src + tgt 2D coords), 1 output (weight).
    // Use generous structural mutation so the CPPN can develop quickly.
    let cppn_config = NeatConfig {
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        weight_mutation_prob: 0.8,
        weight_mutation_power: 0.7,
        ..NeatConfig::cppn(4, 1)
    };

    let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid);
    println!(
        "Substrate: {} nodes, {} potential links",
        substrate.nodes().len(),
        substrate.links().len(),
    );

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
    let evaluator = HyperXor { substrate };

    let mut best_fitness = f32::NEG_INFINITY;
    let mut solution_generation: Option<usize> = None;

    for gen in 0..generations {
        ga.step(&evaluator);

        let pop = ga.population();
        let best = pop
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();

        if best.fitness > best_fitness {
            best_fitness = best.fitness;
        }
        if best.fitness >= 3.9 && solution_generation.is_none() {
            solution_generation = Some(gen);
        }

        if gen % 10 == 0 || gen == generations - 1 {
            let avg: f32 = pop.iter().map(|p| p.fitness).sum::<f32>() / pop.len() as f32;
            println!(
                "Gen {:3}: best={:.4}, avg={:.4}, cppn_nodes={}, cppn_conns={}",
                gen,
                best.fitness,
                avg,
                best.genotype.nodes.len(),
                best.genotype.num_enabled_connections()
            );
        }
    }

    println!();
    let final_pop = ga.population();
    let champion = final_pop
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();

    println!("Best CPPN fitness: {:.4}", champion.fitness);
    if let Some(gen) = solution_generation {
        println!("Solution found at generation: {}", gen);
    } else {
        println!("(No solution found within {} generations.)", generations);
    }

    // Materialize the champion's substrate and print outputs.
    let cppn = CppnEvaluator::new(&champion.genotype).expect("champion is acyclic");
    let substrate = LayeredSubstrate::new(&[2, 2, 1], Activation::Tanh, Activation::Sigmoid);
    let network = substrate_to_network(&cppn, &substrate, EXPRESSION_THRESHOLD);

    println!("\nChampion XOR outputs (via expressed substrate):");
    for (inputs, expected) in &[
        ([0.0_f32, 0.0], 0.0_f32),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ] {
        let out = network.evaluate(inputs)[0];
        let rounded = if out > 0.5 { 1.0 } else { 0.0 };
        let status = if (rounded - expected).abs() < 0.1 { "✓" } else { "✗" };
        println!(
            "  {} XOR {} = {:.4} (expected {}) {}",
            inputs[0] as i32, inputs[1] as i32, out, *expected as i32, status
        );
    }
}
