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
            black_box(generate_pattern(&mut evaluator, 32, 32, 0));
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

criterion_group!(
    benches,
    bench_genome_creation,
    bench_mutation,
    bench_crossover,
    bench_evaluation,
    bench_compatibility_distance,
);
criterion_main!(benches);
