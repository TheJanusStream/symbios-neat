//! Benchmarks for symbios-neat.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symbios_genetics::Genotype;
use symbios_neat::{generate_pattern, CppnEvaluator, NeatConfig, NeatGenome};

fn bench_genome_creation(c: &mut Criterion) {
    let config = NeatConfig::cppn(4, 2);

    c.bench_function("genome_fully_connected", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        b.iter(|| {
            black_box(NeatGenome::fully_connected(config.clone(), &mut rng));
        });
    });
}

fn bench_mutation(c: &mut Criterion) {
    let config = NeatConfig {
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        weight_mutation_prob: 0.8,
        ..NeatConfig::cppn(4, 2)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    c.bench_function("genome_mutation", |b| {
        let mut g = genome.clone();
        b.iter(|| {
            g.mutate(&mut rng, 1.0);
            black_box(&g);
        });
    });
}

fn bench_crossover(c: &mut Criterion) {
    let config = NeatConfig::cppn(4, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

    // Add some structure
    for _ in 0..5 {
        parent1.mutate(&mut rng, 1.0);
        parent2.mutate(&mut rng, 1.0);
    }

    c.bench_function("genome_crossover", |b| {
        b.iter(|| {
            black_box(parent1.crossover(&parent2, &mut rng));
        });
    });
}

fn bench_evaluation(c: &mut Criterion) {
    let config = NeatConfig::cppn(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add some hidden nodes
    for _ in 0..5 {
        if let Some(conn_id) = genome.connections.iter().next().map(|(id, _)| id) {
            genome.add_node(conn_id, &mut rng);
        }
    }

    let mut evaluator = CppnEvaluator::new(&genome);

    c.bench_function("cppn_evaluate_single", |b| {
        b.iter(|| {
            black_box(evaluator.query_2d(0.5, -0.5));
        });
    });

    c.bench_function("cppn_generate_pattern_32x32", |b| {
        b.iter(|| {
            let _ = black_box(generate_pattern(&mut evaluator, 32, 32, 0));
        });
    });
}

fn bench_compatibility_distance(c: &mut Criterion) {
    let config = NeatConfig::cppn(4, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut genome1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut genome2 = NeatGenome::fully_connected(config, &mut rng);

    for _ in 0..10 {
        genome1.mutate(&mut rng, 1.0);
        genome2.mutate(&mut rng, 1.0);
    }

    c.bench_function("compatibility_distance", |b| {
        b.iter(|| {
            black_box(genome1.compatibility_distance(&genome2));
        });
    });
}

/// Benchmark speciation-like workload: O(N²) compatibility comparisons.
/// This simulates what happens during speciation in a real NEAT population.
fn bench_speciation_workload(c: &mut Criterion) {
    let config = NeatConfig {
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        ..NeatConfig::cppn(4, 2)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create a small "population" of diverged genomes
    let pop_size = 50; // 50 genomes = 1225 comparisons
    let mut population: Vec<NeatGenome> = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut genome = NeatGenome::fully_connected(config.clone(), &mut rng);
        for _ in 0..15 {
            genome.mutate(&mut rng, 1.0);
        }
        population.push(genome);
    }

    c.bench_function("speciation_50_genomes", |b| {
        b.iter(|| {
            let mut total_distance = 0.0f32;
            for i in 0..population.len() {
                for j in (i + 1)..population.len() {
                    total_distance += population[i].compatibility_distance(&population[j]);
                }
            }
            black_box(total_distance)
        });
    });
}

/// Benchmark evaluator construction (includes depth computation).
fn bench_evaluator_construction(c: &mut Criterion) {
    let config = NeatConfig::cppn(4, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Build a moderately complex network
    for _ in 0..20 {
        genome.mutate(&mut rng, 1.0);
    }

    c.bench_function("evaluator_construction", |b| {
        b.iter(|| {
            black_box(CppnEvaluator::new(&genome));
        });
    });
}

/// Benchmark update_depths (now uses O(V+E) Kahn's algorithm).
fn bench_update_depths(c: &mut Criterion) {
    let config = NeatConfig::cppn(4, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Build a deep network
    for _ in 0..30 {
        genome.mutate(&mut rng, 1.0);
    }

    c.bench_function("update_depths", |b| {
        b.iter(|| {
            black_box(genome.update_depths());
        });
    });
}

criterion_group!(
    benches,
    bench_genome_creation,
    bench_mutation,
    bench_crossover,
    bench_evaluation,
    bench_compatibility_distance,
    bench_speciation_workload,
    bench_evaluator_construction,
    bench_update_depths,
);
criterion_main!(benches);
