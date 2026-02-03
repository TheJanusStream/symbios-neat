//! XOR example using NEAT with symbios-genetics.
//!
//! This example demonstrates evolving a neural network to solve the XOR problem,
//! a classic benchmark for neuroevolution algorithms.
//!
//! Run with: `cargo run --example xor`

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symbios_genetics::{algorithms::simple::SimpleGA, Evaluator, Evolver};
use symbios_neat::{CppnEvaluator, NeatConfig, NeatGenome};

/// XOR fitness evaluator.
///
/// Evaluates how well a network solves the XOR problem.
/// Maximum fitness is 4.0 (perfect solution).
struct XorFitness;

impl Evaluator<NeatGenome> for XorFitness {
    fn evaluate(&self, genome: &NeatGenome) -> (f32, Vec<f32>, Vec<f32>) {
        let evaluator = CppnEvaluator::new(genome);
        let mut total_error = 0.0;

        // XOR truth table
        let test_cases = [
            ([0.0_f32, 0.0], 0.0_f32),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        for (inputs, expected) in &test_cases {
            let output = evaluator.evaluate(inputs)[0];
            let error = (output - expected).powi(2);
            total_error += error;
        }

        // Fitness: higher is better
        // Max possible error is 4.0 (all wrong), so fitness = 4.0 - error
        let fitness = 4.0 - total_error;

        (fitness, vec![fitness], vec![])
    }
}

fn main() {
    println!("NEAT XOR Example");
    println!("================\n");

    // Configuration
    let config = NeatConfig {
        num_inputs: 2,
        num_outputs: 1,
        use_bias: true,
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        weight_mutation_prob: 0.8,
        weight_mutation_power: 0.5,
        ..NeatConfig::default()
    };

    let population_size = 150;
    let generations = 100;
    let mutation_rate = 1.0;
    let elite_count = 5;
    let seed = 42;

    // Create initial population
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let initial_population: Vec<NeatGenome> = (0..population_size)
        .map(|_| NeatGenome::fully_connected(config.clone(), &mut rng))
        .collect();

    // Create evolutionary algorithm
    let mut ga = SimpleGA::new(initial_population, mutation_rate, elite_count, seed);

    println!("Population: {}", population_size);
    println!("Generations: {}", generations);
    println!("Elite count: {}", elite_count);
    println!();

    // Evolution loop
    let evaluator = XorFitness;
    let mut best_fitness = f32::NEG_INFINITY;
    let mut solution_generation = None;

    for gen in 0..generations {
        ga.step(&evaluator);

        // Find best individual
        let population = ga.population();
        let best = population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();

        if best.fitness > best_fitness {
            best_fitness = best.fitness;
        }

        // Check for solution (fitness >= 3.9 is close enough)
        if best.fitness >= 3.9 && solution_generation.is_none() {
            solution_generation = Some(gen);
        }

        // Print progress every 10 generations
        if gen % 10 == 0 || gen == generations - 1 {
            let avg_fitness: f32 =
                population.iter().map(|p| p.fitness).sum::<f32>() / population.len() as f32;
            let best_nodes = best.genotype.nodes.len();
            let best_conns = best.genotype.num_enabled_connections();

            println!(
                "Gen {:3}: best={:.4}, avg={:.4}, nodes={}, connections={}",
                gen, best.fitness, avg_fitness, best_nodes, best_conns
            );
        }
    }

    println!();

    // Final results
    let final_pop = ga.population();
    let champion = final_pop
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();

    println!("Evolution Complete!");
    println!("==================");
    println!("Best fitness: {:.4}", champion.fitness);
    println!("Nodes: {}", champion.genotype.nodes.len());
    println!(
        "Connections: {}",
        champion.genotype.num_enabled_connections()
    );
    println!("Hidden nodes: {}", champion.genotype.hidden_ids().len());

    if let Some(gen) = solution_generation {
        println!("Solution found at generation: {}", gen);
    }

    // Test the champion
    println!("\nChampion XOR outputs:");
    let eval = CppnEvaluator::new(&champion.genotype);

    let test_cases = [
        ([0.0_f32, 0.0], 0.0_f32),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    for (inputs, expected) in &test_cases {
        let output = eval.evaluate(inputs)[0];
        let rounded = if output > 0.5 { 1.0 } else { 0.0 };
        let status = if (rounded - expected).abs() < 0.1 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  {} XOR {} = {:.4} (expected {}) {}",
            inputs[0] as i32, inputs[1] as i32, output, *expected as i32, status
        );
    }
}
