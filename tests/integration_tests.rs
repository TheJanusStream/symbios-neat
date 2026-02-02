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

/// Test that Identity activation clamps to prevent overflow propagation.
/// Without clamping, large weights can cause INFINITY to propagate through the network.
#[test]
fn test_identity_activation_bounded() {
    // Identity should clamp to prevent downstream overflow
    let result_pos_inf = Activation::Identity.apply(f32::INFINITY);
    let result_neg_inf = Activation::Identity.apply(f32::NEG_INFINITY);
    let result_large = Activation::Identity.apply(1e30);
    let result_small = Activation::Identity.apply(-1e30);

    // All results should be finite and bounded
    assert!(
        result_pos_inf.is_finite(),
        "Identity(+inf) should be finite, got {}",
        result_pos_inf
    );
    assert!(
        result_neg_inf.is_finite(),
        "Identity(-inf) should be finite, got {}",
        result_neg_inf
    );
    assert!(
        result_large.is_finite() && result_large.abs() <= 1e6,
        "Identity(1e30) should be bounded to 1e6, got {}",
        result_large
    );
    assert!(
        result_small.is_finite() && result_small.abs() <= 1e6,
        "Identity(-1e30) should be bounded to 1e6, got {}",
        result_small
    );
}

/// Test that a network with Identity activation doesn't overflow with large weights.
#[test]
fn test_network_no_overflow_with_large_weights() {
    let config = NeatConfig {
        output_activation: Activation::Identity,
        hidden_activations: vec![Activation::Identity],
        weight_range: 100.0, // Large weights
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set weights to extreme values
    for (_, conn) in &mut genome.connections {
        conn.weight = 1000.0;
    }

    // Add several hidden nodes with Identity
    for _ in 0..5 {
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

    // Set all weights to large values again
    for (_, conn) in &mut genome.connections {
        if conn.enabled {
            conn.weight = 1000.0;
        }
    }

    let mut evaluator = CppnEvaluator::new(&genome);
    let output = evaluator.evaluate(&[100.0, 100.0]);

    // Output should be finite, not overflow to infinity
    assert!(
        output[0].is_finite(),
        "Network output should be finite even with large weights, got {}",
        output[0]
    );
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

/// Test that hash-based innovation numbers don't collide with fixed node IDs.
/// NeatGenome::minimal assigns innovation 0, 1, 2, ... to nodes.
/// Hash-based innovations should never produce these low values.
#[test]
fn test_innovation_no_collision_with_fixed_ids() {
    // The fixed IDs used by NeatGenome::minimal are:
    // - Bias: 0
    // - Inputs: 1..=num_inputs
    // - Outputs: num_inputs+1..=num_inputs+num_outputs
    // For a typical network with 10 inputs and 5 outputs, fixed IDs are 0..=15

    let reserved_range = 1000u64; // Conservative upper bound for fixed IDs

    // Test many connection innovations
    for i in 0..1000u64 {
        for j in 0..100u64 {
            let inn = connection_innovation(i, j);
            assert!(
                inn >= reserved_range,
                "Connection innovation {} (from {}, {}) collides with reserved range",
                inn,
                i,
                j
            );
        }
    }

    // Test node split innovations
    for conn_inn in 0..10000u64 {
        let node_inn = node_split_innovation(conn_inn);
        assert!(
            node_inn >= reserved_range,
            "Node split innovation {} (from conn {}) collides with reserved range",
            node_inn,
            conn_inn
        );
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

/// Test that generate_pattern handles edge case of width=1 or height=1 correctly.
/// A single pixel should be centered at 0.0, not at the boundary.
#[test]
fn test_generate_pattern_single_pixel() {
    let config = NeatConfig {
        output_activation: Activation::Identity,
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set weights so output = x + y (first input + second input)
    // With Identity activation, this lets us verify the input coordinates
    let mut conn_iter = genome.connections.iter_mut();
    if let Some((_, conn)) = conn_iter.next() {
        conn.weight = 1.0; // x weight
    }
    if let Some((_, conn)) = conn_iter.next() {
        conn.weight = 1.0; // y weight
    }
    // Disable any other connections (bias)
    for (_, conn) in conn_iter {
        conn.enabled = false;
    }

    let mut evaluator = CppnEvaluator::new(&genome);

    // For 1x1 pattern, the single pixel should be at (0, 0)
    // Output = x + y = 0 + 0 = 0, normalized to 0.5
    let pattern_1x1 = generate_pattern(&mut evaluator, 1, 1, 0);
    assert_eq!(pattern_1x1.len(), 1);
    assert!(
        (pattern_1x1[0] - 0.5).abs() < 0.01,
        "1x1 pattern should be centered (0.5), got {}",
        pattern_1x1[0]
    );

    // For 2x1 pattern, pixels should be at x=-1 and x=+1, y=0
    // Output[0] = -1 + 0 = -1, normalized to 0.0
    // Output[1] = +1 + 0 = +1, normalized to 1.0
    let pattern_2x1 = generate_pattern(&mut evaluator, 2, 1, 0);
    assert_eq!(pattern_2x1.len(), 2);
    assert!(
        (pattern_2x1[0] - 0.0).abs() < 0.01,
        "2x1 pattern[0] should be 0.0 (x=-1), got {}",
        pattern_2x1[0]
    );
    assert!(
        (pattern_2x1[1] - 1.0).abs() < 0.01,
        "2x1 pattern[1] should be 1.0 (x=+1), got {}",
        pattern_2x1[1]
    );
}

/// Test that generate_pattern panics on invalid output_index instead of silent failure.
#[test]
#[should_panic(expected = "output_index")]
fn test_generate_pattern_invalid_output_index_panics() {
    let config = NeatConfig::cppn(2, 1); // Only 1 output
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    let mut evaluator = CppnEvaluator::new(&genome);

    // Request output index 5 when only 1 output exists - should panic
    let _ = generate_pattern(&mut evaluator, 4, 4, 5);
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

/// Test that node depth is computed as longest path from inputs (not shortest).
/// This is critical for correct feedforward evaluation order.
#[test]
fn test_depth_is_longest_path() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Get initial state: input0 -> output, input1 -> output
    // Both inputs at depth 0, output should be at depth 1
    let output_id = genome.output_ids[0];
    assert_eq!(
        genome.nodes[output_id].depth, 1,
        "Output should be at depth 1 with direct connections"
    );

    // Now split one connection: input0 -> hidden -> output
    // The hidden node will be at depth 1
    // The output now receives from:
    //   - input1 (depth 0) -> path length 1
    //   - hidden (depth 1) -> path length 2
    // Output depth should be MAX(1, 2) = 2, not MIN(1, 2) = 1
    let conn_id = genome.connections.iter().next().unwrap().0;
    let hidden_id = genome.add_node(conn_id, &mut rng).unwrap();

    let hidden_depth = genome.nodes[hidden_id].depth;
    let output_depth = genome.nodes[output_id].depth;

    assert_eq!(hidden_depth, 1, "Hidden node should be at depth 1");
    assert_eq!(
        output_depth, 2,
        "Output should be at depth 2 (longest path), not 1 (shortest path)"
    );
}

/// Test that evaluation order respects node dependencies.
/// A node should not be evaluated until all its inputs are ready.
#[test]
fn test_evaluation_order_respects_dependencies() {
    let config = NeatConfig {
        output_activation: Activation::Identity,
        hidden_activations: vec![Activation::Identity],
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set all weights to 1.0 for predictable math
    for (_, conn) in &mut genome.connections {
        conn.weight = 1.0;
    }

    // Initial network: input0 + input1 -> output
    // With identity activation and weights 1.0: output = input0 + input1
    let mut evaluator = CppnEvaluator::new(&genome);
    let output = evaluator.evaluate(&[1.0, 2.0]);
    assert!(
        (output[0] - 3.0).abs() < 1e-5,
        "Simple sum should work: expected 3.0, got {}",
        output[0]
    );

    // Now split one connection to create a hidden node
    // input0 -> hidden (weight 1.0)
    // hidden -> output (original weight 1.0)
    // input1 -> output (weight 1.0)
    // So: output = hidden + input1 = input0 + input1 = 3.0
    let conn_id = genome.connections.iter().next().unwrap().0;
    genome.add_node(conn_id, &mut rng);

    // Set all weights to 1.0 again
    for (_, conn) in &mut genome.connections {
        if conn.enabled {
            conn.weight = 1.0;
        }
    }

    let mut evaluator = CppnEvaluator::new(&genome);
    let output = evaluator.evaluate(&[1.0, 2.0]);

    // If depth is wrong, hidden might not be evaluated before output
    // causing output to miss the hidden->output signal
    assert!(
        (output[0] - 3.0).abs() < 1e-5,
        "With hidden node, sum should still be ~3.0, got {} (depth bug if < 3)",
        output[0]
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

/// Test that Step activation uses standard Heaviside convention: f(x) = 1 for x >= 0.
#[test]
fn test_step_activation_at_zero() {
    // Standard Heaviside step function: f(x) = 1 for x >= 0, f(x) = 0 for x < 0
    assert_eq!(
        Activation::Step.apply(0.0),
        1.0,
        "Step(0) should be 1.0 (Heaviside convention)"
    );
    assert_eq!(
        Activation::Step.apply(0.001),
        1.0,
        "Step(positive) should be 1.0"
    );
    assert_eq!(
        Activation::Step.apply(-0.001),
        0.0,
        "Step(negative) should be 0.0"
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
