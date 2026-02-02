//! Integration tests for symbios-neat.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symbios_genetics::Genotype;
use symbios_neat::{
    connection_innovation, generate_pattern, node_split_innovation, Activation, CppnEvaluator,
    NeatConfig, NeatGenome, NodeType,
};

// =============================================================================
// Bug Fix Regression Tests
// =============================================================================

/// Test that activation functions handle NaN consistently by propagating it.
#[test]
fn test_activation_nan_propagation() {
    let nan = f32::NAN;

    for activation in Activation::ALL {
        let result = activation.apply(nan);
        assert!(
            result.is_nan(),
            "Activation {:?} should propagate NaN, got {}",
            activation,
            result
        );
    }
}

/// Test that activation functions handle extreme values without overflow/panic.
#[test]
fn test_activation_extreme_values() {
    let extreme_values = [
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e38,
        -1e38,
        1e-38,
        -1e-38,
    ];

    for activation in Activation::ALL {
        for &value in &extreme_values {
            let result = activation.apply(value);
            // Result should be finite or a well-defined infinity, not NaN
            assert!(
                !result.is_nan(),
                "Activation {:?} produced NaN for input {}, expected finite or infinite",
                activation,
                value
            );
        }
    }
}

/// Test that update_depths doesn't infinite loop (has iteration limit).
#[test]
fn test_update_depths_terminates() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add many nodes to create a potentially deep network
    for _ in 0..20 {
        if let Some(conn_id) = genome
            .connections
            .iter()
            .filter(|(_, c)| c.enabled)
            .next()
            .map(|(id, _)| id)
        {
            genome.add_node(conn_id, &mut rng);
        }
    }

    // This should terminate quickly, not hang
    let is_acyclic = genome.update_depths();
    assert!(
        is_acyclic,
        "Genome should be acyclic after normal mutations"
    );
}

/// Test that has_cycle correctly detects cycles.
#[test]
fn test_cycle_detection() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    // A freshly created genome should not have cycles
    assert!(!genome.has_cycle(), "Fresh genome should not have cycles");
}

/// Test that crossover produces acyclic offspring.
#[test]
fn test_crossover_acyclic() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create two parents with different structures
    let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

    // Heavily mutate both parents
    for _ in 0..20 {
        parent1.mutate(&mut rng, 1.0);
        parent2.mutate(&mut rng, 1.0);
    }

    // Perform crossover many times
    for _ in 0..100 {
        let child = parent1.crossover(&parent2, &mut rng);
        assert!(
            !child.has_cycle(),
            "Crossover should not produce cyclic offspring"
        );

        // Verify the child can be evaluated without hanging
        let mut evaluator = CppnEvaluator::new(&child);
        let output = evaluator.evaluate(&[0.5, 0.5]);
        assert!(output[0].is_finite(), "Child should produce finite output");
    }
}

/// Test that add_node returns None when hidden_activations is empty.
#[test]
fn test_add_node_empty_activations() {
    let config = NeatConfig {
        hidden_activations: vec![], // Empty!
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    let conn_id = genome.connections.iter().next().unwrap().0;
    let result = genome.add_node(conn_id, &mut rng);

    assert!(
        result.is_none(),
        "add_node should return None when hidden_activations is empty"
    );
}

/// Test that evaluate_into works correctly and matches evaluate.
#[test]
fn test_evaluate_into_matches_evaluate() {
    let config = NeatConfig::cppn(3, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add some structure
    for _ in 0..5 {
        genome.mutate(&mut rng, 1.0);
    }

    let mut evaluator = CppnEvaluator::new(&genome);

    let inputs = [0.5, -0.3, 0.8];
    let outputs_vec = evaluator.evaluate(&inputs);

    let mut outputs_buf = [0.0f32; 2];
    evaluator.evaluate_into(&inputs, &mut outputs_buf);

    assert!(
        (outputs_vec[0] - outputs_buf[0]).abs() < 1e-6,
        "evaluate and evaluate_into should produce same results"
    );
    assert!(
        (outputs_vec[1] - outputs_buf[1]).abs() < 1e-6,
        "evaluate and evaluate_into should produce same results"
    );
}

/// Test that large-scale evolution doesn't crash or hang.
#[test]
fn test_large_scale_evolution_stability() {
    let config = NeatConfig {
        add_connection_prob: 0.3,
        add_node_prob: 0.1,
        ..NeatConfig::minimal(4, 2)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Create population
    let mut population: Vec<NeatGenome> = (0..50)
        .map(|_| NeatGenome::fully_connected(config.clone(), &mut rng))
        .collect();

    // Run many generations
    for generation in 0..20 {
        // Mutate all
        for genome in &mut population {
            genome.mutate(&mut rng, 1.0);
        }

        // Crossover
        let mut offspring = Vec::new();
        for i in (0..population.len()).step_by(2) {
            if i + 1 < population.len() {
                let child = population[i].crossover(&population[i + 1], &mut rng);
                offspring.push(child);
            }
        }
        population.extend(offspring);

        // Verify all genomes are valid
        for (i, genome) in population.iter().enumerate() {
            assert!(
                !genome.has_cycle(),
                "Generation {} genome {} has cycle",
                generation,
                i
            );

            let mut evaluator = CppnEvaluator::new(genome);
            let output = evaluator.evaluate(&[0.1, 0.2, 0.3, 0.4]);
            for (j, &val) in output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Generation {} genome {} output {} is not finite: {}",
                    generation,
                    i,
                    j,
                    val
                );
            }
        }

        // Select
        population.truncate(50);
    }
}

#[test]
fn test_full_evolution_cycle() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create initial population
    let mut population: Vec<NeatGenome> = (0..10)
        .map(|_| NeatGenome::fully_connected(config.clone(), &mut rng))
        .collect();

    // Run a few generations
    for _ in 0..5 {
        // Mutate
        for genome in &mut population {
            genome.mutate(&mut rng, 1.0);
        }

        // Crossover (simple: pair adjacent genomes)
        let mut offspring = Vec::new();
        for i in (0..population.len()).step_by(2) {
            if i + 1 < population.len() {
                let child = population[i].crossover(&population[i + 1], &mut rng);
                offspring.push(child);
            }
        }

        population.extend(offspring);

        // Select (keep best half by connection count as proxy for complexity)
        population.sort_by_key(|g| std::cmp::Reverse(g.num_enabled_connections()));
        population.truncate(10);
    }

    // Verify population is still valid
    for genome in &population {
        let mut evaluator = CppnEvaluator::new(genome);
        let output = evaluator.evaluate(&[0.5, 0.5]);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }
}

#[test]
fn test_structural_innovation_consistency() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng1 = ChaCha8Rng::seed_from_u64(100);
    let mut rng2 = ChaCha8Rng::seed_from_u64(200);

    let genome1 = NeatGenome::fully_connected(config.clone(), &mut rng1);
    let genome2 = NeatGenome::fully_connected(config, &mut rng2);

    // Both genomes should have same connection innovations (same topology)
    let innovations1: std::collections::HashSet<u64> = genome1
        .connections
        .iter()
        .map(|(_, c)| c.innovation)
        .collect();
    let innovations2: std::collections::HashSet<u64> = genome2
        .connections
        .iter()
        .map(|(_, c)| c.innovation)
        .collect();

    assert_eq!(
        innovations1, innovations2,
        "Same topology should have same innovations"
    );

    // Add same connection to both (input 0 -> output 0 if not exists)
    let input_id_1 = genome1.input_ids[0];
    let output_id_1 = genome1.output_ids[0];
    let input_inn_1 = genome1.nodes[input_id_1].innovation;
    let output_inn_1 = genome1.nodes[output_id_1].innovation;

    let input_id_2 = genome2.input_ids[0];
    let output_id_2 = genome2.output_ids[0];
    let input_inn_2 = genome2.nodes[input_id_2].innovation;
    let output_inn_2 = genome2.nodes[output_id_2].innovation;

    // Same structural position should have same innovation
    assert_eq!(input_inn_1, input_inn_2);
    assert_eq!(output_inn_1, output_inn_2);
    assert_eq!(
        connection_innovation(input_inn_1, output_inn_1),
        connection_innovation(input_inn_2, output_inn_2)
    );
}

#[test]
fn test_node_split_creates_correct_topology() {
    let config = NeatConfig::minimal(1, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Initial: 1 input, 1 output, 1 connection
    assert_eq!(genome.nodes.len(), 2);
    assert_eq!(genome.connections.len(), 1);

    let conn_id = genome.connections.iter().next().unwrap().0;
    let original_conn = genome.connections[conn_id].clone();

    // Split the connection
    let new_node_id = genome.add_node(conn_id, &mut rng).unwrap();

    // Now: 1 input, 1 output, 1 hidden = 3 nodes
    // 1 disabled original + 2 new connections = 3 connections
    assert_eq!(genome.nodes.len(), 3);
    assert_eq!(genome.connections.len(), 3);
    assert_eq!(genome.num_enabled_connections(), 2);

    // Verify the new node is hidden
    let new_node = &genome.nodes[new_node_id];
    assert_eq!(new_node.node_type, NodeType::Hidden);

    // Verify innovation is deterministic
    let expected_node_inn = node_split_innovation(original_conn.innovation);
    assert_eq!(new_node.innovation, expected_node_inn);
}

#[test]
fn test_cppn_produces_spatial_patterns() {
    let config = NeatConfig {
        hidden_activations: vec![Activation::Sine, Activation::Gaussian],
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add hidden nodes with periodic activations
    for _ in 0..3 {
        if let Some(conn_id) = genome
            .connections
            .iter()
            .filter(|(_, c)| c.enabled)
            .next()
            .map(|(id, _)| id)
        {
            genome.add_node(conn_id, &mut rng);
        }
    }

    let mut evaluator = CppnEvaluator::new(&genome);
    let pattern = generate_pattern(&mut evaluator, 8, 8, 0);

    // Pattern should have some variation (not all same value)
    let min = pattern.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = pattern.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        max - min > 0.01,
        "Pattern should have variation, got range [{}, {}]",
        min,
        max
    );
}

#[test]
fn test_compatibility_distance_properties() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let genome1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let genome2 = genome1.clone();
    let mut genome3 = NeatGenome::fully_connected(config, &mut rng);

    // Add lots of structure to genome3
    for _ in 0..10 {
        genome3.mutate(&mut rng, 1.0);
    }

    // Distance to self should be ~0
    let self_dist = genome1.compatibility_distance(&genome1);
    assert!(
        self_dist.abs() < 1e-6,
        "Self distance should be 0, got {}",
        self_dist
    );

    // Distance to clone should be ~0
    let clone_dist = genome1.compatibility_distance(&genome2);
    assert!(
        clone_dist.abs() < 1e-6,
        "Clone distance should be 0, got {}",
        clone_dist
    );

    // Distance to different genome should be > 0
    let diff_dist = genome1.compatibility_distance(&genome3);
    assert!(
        diff_dist > 0.0,
        "Different genome distance should be > 0, got {}",
        diff_dist
    );

    // Symmetry: d(a,b) == d(b,a)
    let dist_ab = genome1.compatibility_distance(&genome3);
    let dist_ba = genome3.compatibility_distance(&genome1);
    assert!(
        (dist_ab - dist_ba).abs() < 1e-6,
        "Distance should be symmetric: {} vs {}",
        dist_ab,
        dist_ba
    );
}

#[test]
fn test_serialization_preserves_behavior() {
    let config = NeatConfig::cppn(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add structure
    for _ in 0..3 {
        genome.mutate(&mut rng, 1.0);
    }

    // Evaluate original
    let mut eval1 = CppnEvaluator::new(&genome);
    let output1 = eval1.query_2d(0.5, -0.3);

    // Serialize and deserialize
    let json = serde_json::to_string(&genome).unwrap();
    let restored: NeatGenome = serde_json::from_str(&json).unwrap();

    // Evaluate restored
    let mut eval2 = CppnEvaluator::new(&restored);
    let output2 = eval2.query_2d(0.5, -0.3);

    // Outputs should match
    assert!(
        (output1[0] - output2[0]).abs() < 1e-6,
        "Serialization should preserve behavior: {} vs {}",
        output1[0],
        output2[0]
    );
}

#[test]
fn test_all_activation_functions_work() {
    for activation in Activation::ALL {
        let config = NeatConfig {
            output_activation: activation,
            hidden_activations: vec![activation],
            ..NeatConfig::minimal(1, 1)
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let genome = NeatGenome::fully_connected(config, &mut rng);

        let mut evaluator = CppnEvaluator::new(&genome);
        let output = evaluator.evaluate(&[0.5]);

        assert!(
            output[0].is_finite(),
            "Activation {:?} produced non-finite output",
            activation
        );
    }
}
