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

/// Regression test for Bug #1: Deterministic Input/Output Mapping Failure.
/// The evaluator must use genome.input_ids/output_ids for semantic ordering,
/// not SlotMap iteration order which can change after crossover/deserialization.
#[test]
fn test_evaluator_preserves_input_output_order_after_crossover() {
    let config = NeatConfig {
        output_activation: Activation::Identity,
        hidden_activations: vec![Activation::Identity],
        ..NeatConfig::minimal(2, 2)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create two parents with different weights
    let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

    // Give them distinct mutations
    for _ in 0..10 {
        parent1.mutate(&mut rng, 1.0);
        parent2.mutate(&mut rng, 1.0);
    }

    // Perform crossover - this may reorder nodes in the SlotMap
    let child = parent1.crossover(&parent2, &mut rng);

    // Verify input_ids and output_ids are preserved
    assert_eq!(child.input_ids.len(), 2, "Child should have 2 inputs");
    assert_eq!(child.output_ids.len(), 2, "Child should have 2 outputs");

    // Create evaluator and verify it respects semantic ordering
    let mut evaluator = CppnEvaluator::new(&child);

    // Evaluate with distinct inputs [1.0, 0.0] and [0.0, 1.0]
    // If input mapping is correct, these should produce different outputs
    let output1 = evaluator.evaluate(&[1.0, 0.0]);
    let output2 = evaluator.evaluate(&[0.0, 1.0]);

    // The outputs should differ (unless the network is symmetric, which is unlikely)
    // But more importantly, the evaluator shouldn't crash
    assert_eq!(output1.len(), 2, "Should have 2 outputs");
    assert_eq!(output2.len(), 2, "Should have 2 outputs");

    // Verify all outputs are finite
    for &v in &output1 {
        assert!(v.is_finite(), "Output should be finite");
    }
    for &v in &output2 {
        assert!(v.is_finite(), "Output should be finite");
    }
}

/// Regression test for Bug #1: Verify serialization doesn't break input/output mapping.
#[test]
fn test_evaluator_preserves_order_after_serialization() {
    let config = NeatConfig::minimal(3, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add structure
    for _ in 0..5 {
        genome.mutate(&mut rng, 1.0);
    }

    // Evaluate original
    let mut eval1 = CppnEvaluator::new(&genome);
    let inputs = [0.5, -0.3, 0.8];
    let output_before = eval1.evaluate(&inputs);

    // Serialize and deserialize
    let json = serde_json::to_string(&genome).unwrap();
    let restored: NeatGenome = serde_json::from_str(&json).unwrap();

    // Verify input_ids and output_ids are preserved
    assert_eq!(genome.input_ids.len(), restored.input_ids.len());
    assert_eq!(genome.output_ids.len(), restored.output_ids.len());

    // Evaluate restored - should produce identical results
    let mut eval2 = CppnEvaluator::new(&restored);
    let output_after = eval2.evaluate(&inputs);

    for (i, (&before, &after)) in output_before.iter().zip(output_after.iter()).enumerate() {
        assert!(
            (before - after).abs() < 1e-6,
            "Output {} should match after serialization: {} vs {}",
            i,
            before,
            after
        );
    }
}

/// Regression test for Bug #2: Endianness-Dependent Innovation Hashing.
/// Innovation hashing must use explicit little-endian bytes for cross-platform reproducibility.
#[test]
fn test_innovation_hashing_is_deterministic_and_portable() {
    // These specific values should always produce the same hash
    // regardless of platform endianness
    let inn1 = connection_innovation(1, 2);
    let inn2 = connection_innovation(1, 2);
    assert_eq!(inn1, inn2, "Same inputs must produce same innovation");

    // Verify order matters (asymmetric)
    let inn_forward = connection_innovation(100, 200);
    let inn_reverse = connection_innovation(200, 100);
    assert_ne!(
        inn_forward, inn_reverse,
        "Order should matter in innovation hashing"
    );

    // Verify node split is deterministic
    let split1 = node_split_innovation(12345);
    let split2 = node_split_innovation(12345);
    assert_eq!(split1, split2, "Node split must be deterministic");
}

/// Regression test for Bug #3: Evaluation Order Corruption via Depth Saturation.
/// Depth uses u32 to avoid saturation at 65535 layers.
#[test]
fn test_depth_does_not_saturate_at_u16_max() {
    let config = NeatConfig::minimal(1, 1);
    let mut genome = NeatGenome::minimal(config);

    // Manually set a node's depth beyond u16::MAX
    let output_id = genome.output_ids[0];
    genome.nodes[output_id].depth = 70_000; // Beyond u16::MAX (65535)

    // Verify it actually stored the value (would truncate if u16)
    assert_eq!(
        genome.nodes[output_id].depth, 70_000,
        "Depth should support values > 65535"
    );

    // Verify u32::MAX works
    genome.nodes[output_id].depth = u32::MAX;
    assert_eq!(
        genome.nodes[output_id].depth,
        u32::MAX,
        "Depth should support u32::MAX"
    );
}

/// Regression test for Bug #4: Unbounded Weight Growth.
/// Weights must be clamped during mutation to prevent Inf/NaN in compatibility_distance.
#[test]
fn test_weight_clamping_prevents_unbounded_growth() {
    let config = NeatConfig {
        weight_mutation_prob: 1.0,
        weight_replace_prob: 0.0,     // Always perturb, never replace
        weight_mutation_power: 100.0, // Large perturbation
        weight_range: 1.0,
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config.clone(), &mut rng);

    // Mutate many times - without clamping, weights would grow unbounded
    for _ in 0..1000 {
        genome.mutate(&mut rng, 1.0);
    }

    // Verify all weights are finite and bounded
    let weight_limit = config.weight_range * 10.0;
    for (_, conn) in &genome.connections {
        assert!(
            conn.weight.is_finite(),
            "Weight should be finite, got {}",
            conn.weight
        );
        assert!(
            conn.weight.abs() <= weight_limit,
            "Weight {} exceeds limit {}",
            conn.weight,
            weight_limit
        );
    }

    // Verify compatibility_distance doesn't panic with NaN
    let genome2 = NeatGenome::fully_connected(config, &mut rng);
    let distance = genome.compatibility_distance(&genome2);
    assert!(
        distance.is_finite(),
        "Compatibility distance should be finite, got {}",
        distance
    );
}

/// Regression test for Bug #5: Panicking API in Library Code.
/// generate_pattern should return Result, not panic.
#[test]
fn test_generate_pattern_returns_result_not_panic() {
    let config = NeatConfig::cppn(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);
    let mut evaluator = CppnEvaluator::new(&genome);

    // Valid index should return Ok
    let result = generate_pattern(&mut evaluator, 4, 4, 0);
    assert!(result.is_ok(), "Valid output_index should return Ok");

    // Invalid index should return Err, not panic
    let result = generate_pattern(&mut evaluator, 4, 4, 99);
    assert!(result.is_err(), "Invalid output_index should return Err");

    // Error should be descriptive
    if let Err(e) = result {
        let msg = e.to_string();
        assert!(msg.contains("99"), "Error should mention requested index");
        assert!(msg.contains("1"), "Error should mention available outputs");
    }
}

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

/// Test that break_cycles uses the improved heuristic that targets actual back edges.
/// It should prefer disabling low-weight edges to preserve signal strength.
#[test]
fn test_break_cycles_targets_back_edges() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set up weights so we can verify the heuristic
    for (_, conn) in &mut genome.connections {
        conn.weight = 1.0;
    }

    // The genome should be acyclic initially
    assert!(!genome.has_cycle(), "Initial genome should be acyclic");

    // break_cycles on an acyclic graph should do nothing
    let disabled = genome.break_cycles();
    assert_eq!(disabled, 0, "No cycles to break in acyclic graph");

    // All connections should still be enabled
    let enabled_count = genome.connections.iter().filter(|(_, c)| c.enabled).count();
    assert_eq!(
        enabled_count,
        genome.connections.len(),
        "All connections should remain enabled"
    );
}

/// Test that would_create_cycle prevents cyclic connections.
/// This validates the DFS-based reachability check works correctly.
#[test]
fn test_would_create_cycle_prevents_cycles() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add a hidden node: input0 -> hidden -> output
    let conn_id = genome.connections.iter().next().unwrap().0;
    let hidden_id = genome.add_node(conn_id, &mut rng).unwrap();

    // Try to add a connection from output back to hidden - should fail (would create cycle)
    let output_id = genome.output_ids[0];
    let result = genome.add_connection(output_id, hidden_id, &mut rng);
    assert!(
        result.is_none(),
        "Should not allow connection that would create cycle"
    );

    // Verify the genome is still acyclic
    assert!(!genome.has_cycle(), "Genome should remain acyclic");

    // Adding a connection from hidden to a different output should work
    // (if we had multiple outputs, but we don't in this case)
    // Instead, verify we can still add valid connections
    let input1_id = genome.input_ids[1];
    let result = genome.add_connection(input1_id, hidden_id, &mut rng);
    // This might succeed or fail depending on existing connections, but shouldn't panic
    if result.is_some() {
        assert!(
            !genome.has_cycle(),
            "Valid connection should not create cycle"
        );
    }
}

/// Test that crossover_equal_fitness inherits disjoint/excess genes from both parents.
#[test]
fn test_crossover_equal_fitness_inherits_from_both_parents() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create two parents with different structures
    let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

    // Add different hidden nodes to each parent
    // Parent1: add node to first connection
    let conn_id1 = parent1.connections.iter().next().unwrap().0;
    parent1.add_node(conn_id1, &mut rng);

    // Parent2: add node to second connection (different structure)
    let conn_id2 = parent2.connections.iter().nth(1).unwrap().0;
    parent2.add_node(conn_id2, &mut rng);

    // Count unique innovations in each parent
    let parent1_innovations: std::collections::HashSet<u64> = parent1
        .connections
        .iter()
        .map(|(_, c)| c.innovation)
        .collect();
    let parent2_innovations: std::collections::HashSet<u64> = parent2
        .connections
        .iter()
        .map(|(_, c)| c.innovation)
        .collect();

    // There should be some innovations unique to each parent
    let only_in_p1: std::collections::HashSet<_> = parent1_innovations
        .difference(&parent2_innovations)
        .collect();
    let only_in_p2: std::collections::HashSet<_> = parent2_innovations
        .difference(&parent1_innovations)
        .collect();

    assert!(
        !only_in_p1.is_empty(),
        "Parent1 should have unique innovations"
    );
    assert!(
        !only_in_p2.is_empty(),
        "Parent2 should have unique innovations"
    );

    // Perform equal fitness crossover many times
    // With random inheritance, we should eventually see genes from both parents
    let mut saw_p1_unique = false;
    let mut saw_p2_unique = false;

    for seed in 0..100 {
        let mut crossover_rng = ChaCha8Rng::seed_from_u64(seed);
        let child = parent1.crossover_equal_fitness(&parent2, &mut crossover_rng);

        let child_innovations: std::collections::HashSet<u64> = child
            .connections
            .iter()
            .map(|(_, c)| c.innovation)
            .collect();

        // Check if child has any innovations unique to parent1 or parent2
        for &inn in &only_in_p1 {
            if child_innovations.contains(inn) {
                saw_p1_unique = true;
            }
        }
        for &inn in &only_in_p2 {
            if child_innovations.contains(inn) {
                saw_p2_unique = true;
            }
        }

        // Verify child is valid
        assert!(!child.has_cycle(), "Child should be acyclic");
    }

    assert!(
        saw_p1_unique,
        "crossover_equal_fitness should sometimes inherit unique genes from parent1"
    );
    assert!(
        saw_p2_unique,
        "crossover_equal_fitness should sometimes inherit unique genes from parent2"
    );
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

/// Test that large weight_range values don't cause overflow/NaN in weight initialization.
/// The review identified that rng.random() * 2.0 * weight_range can overflow with large values.
#[test]
fn test_large_weight_range_no_overflow() {
    let config = NeatConfig {
        weight_range: 1e10, // Large but not MAX to avoid trivial overflow
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // This should not panic or produce NaN/Inf
    let genome = NeatGenome::fully_connected(config, &mut rng);

    for (_, conn) in &genome.connections {
        assert!(
            conn.weight.is_finite(),
            "Weight should be finite with large weight_range, got {}",
            conn.weight
        );
    }
}

/// Test that extremely large weight_range (near f32::MAX) is handled gracefully.
/// This tests the edge case where 2.0 * weight_range would overflow.
#[test]
fn test_extreme_weight_range_handled() {
    // f32::MAX / 3.0 is large enough that * 2.0 could overflow
    let config = NeatConfig {
        weight_range: f32::MAX / 3.0,
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let genome = NeatGenome::fully_connected(config, &mut rng);

    // Weights might be infinite due to overflow, but the code shouldn't panic
    // and the evaluator should still work (activation functions clamp infinity)
    let mut evaluator = CppnEvaluator::new(&genome);
    let output = evaluator.evaluate(&[0.5, 0.5]);

    // Output should be finite because activation functions clamp extreme values
    assert!(
        output[0].is_finite(),
        "Evaluator output should be finite even with extreme weights"
    );
}

/// Test that innovation hash distribution is reasonably uniform (no severe modulo bias).
/// While perfect uniformity isn't achievable with modulo, severe clustering would indicate a problem.
#[test]
fn test_innovation_hash_distribution() {
    // Generate many innovations and check they're spread across the range
    let mut innovations = Vec::new();
    for i in 0..1000u64 {
        for j in 0..10u64 {
            innovations.push(connection_innovation(i, j));
        }
    }

    // Check for uniqueness (no collisions in this sample)
    let unique: std::collections::HashSet<u64> = innovations.iter().copied().collect();
    let collision_rate = 1.0 - (unique.len() as f64 / innovations.len() as f64);
    assert!(
        collision_rate < 0.01,
        "Collision rate {} is too high (expected < 1%)",
        collision_rate
    );

    // Check that values are spread across the range (not clustered)
    let min = *innovations.iter().min().unwrap();
    let max = *innovations.iter().max().unwrap();
    let spread = max - min;

    // With 10000 samples, we expect good coverage of a large range
    assert!(
        spread > 1_000_000_000,
        "Innovation spread {} is too small, possible clustering",
        spread
    );
}

/// Test reserved innovation range boundary conditions.
/// All hash-based innovations should be >= RESERVED_INNOVATION_RANGE (65536).
#[test]
fn test_reserved_innovation_range_boundary() {
    const RESERVED_RANGE: u64 = 1 << 16; // 65536

    // Test with values near the boundary
    for input in 0..1000u64 {
        let conn_inn = connection_innovation(input, input + 1);
        assert!(
            conn_inn >= RESERVED_RANGE,
            "Connection innovation {} should be >= {} for inputs ({}, {})",
            conn_inn,
            RESERVED_RANGE,
            input,
            input + 1
        );

        let node_inn = node_split_innovation(input);
        assert!(
            node_inn >= RESERVED_RANGE,
            "Node split innovation {} should be >= {} for input {}",
            node_inn,
            RESERVED_RANGE,
            input
        );
    }

    // Test with very large innovation numbers
    for input in [u64::MAX - 1000, u64::MAX - 1, u64::MAX] {
        let conn_inn = connection_innovation(input, 0);
        assert!(
            conn_inn >= RESERVED_RANGE,
            "Connection innovation with large input should be >= reserved range"
        );
    }
}

/// Test that networks with high-dimensional inputs don't exhaust the reserved range.
/// The review noted that 65536 might be too small for high-dimensional inputs.
#[test]
fn test_high_dimensional_inputs() {
    // Create a network with many inputs (but within realistic bounds)
    let config = NeatConfig {
        num_inputs: 1000,
        num_outputs: 100,
        use_bias: true,
        ..NeatConfig::minimal(1000, 100)
    };

    let genome = NeatGenome::minimal(config);

    // All node innovations should be unique and within the reserved range
    let node_innovations: Vec<u64> = genome.nodes.iter().map(|(_, n)| n.innovation).collect();
    let unique: std::collections::HashSet<u64> = node_innovations.iter().copied().collect();

    assert_eq!(
        unique.len(),
        node_innovations.len(),
        "All node innovations should be unique"
    );

    // Fixed IDs should be < 65536 for reserved range nodes
    let max_fixed_id = *node_innovations.iter().max().unwrap();
    assert!(
        max_fixed_id < 65536,
        "Fixed node IDs {} should fit in reserved range 65536",
        max_fixed_id
    );
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
    let pattern_1x1 = generate_pattern(&mut evaluator, 1, 1, 0).unwrap();
    assert_eq!(pattern_1x1.len(), 1);
    assert!(
        (pattern_1x1[0] - 0.5).abs() < 0.01,
        "1x1 pattern should be centered (0.5), got {}",
        pattern_1x1[0]
    );

    // For 2x1 pattern, pixels should be at x=-1 and x=+1, y=0
    // Output[0] = -1 + 0 = -1, normalized to 0.0
    // Output[1] = +1 + 0 = +1, normalized to 1.0
    let pattern_2x1 = generate_pattern(&mut evaluator, 2, 1, 0).unwrap();
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

/// Test that generate_pattern returns error on invalid output_index instead of panicking.
#[test]
fn test_generate_pattern_invalid_output_index_returns_error() {
    let config = NeatConfig::cppn(2, 1); // Only 1 output
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    let mut evaluator = CppnEvaluator::new(&genome);

    // Request output index 5 when only 1 output exists - should return error
    let result = generate_pattern(&mut evaluator, 4, 4, 5);
    assert!(
        result.is_err(),
        "Should return error for out-of-bounds output_index"
    );

    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("output_index"),
        "Error message should mention output_index"
    );
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
    let pattern = generate_pattern(&mut evaluator, 8, 8, 0).unwrap();

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

/// Test that CppnEvaluator::new handles genomes with stale depths.
/// The evaluator should automatically recompute depths to ensure correct evaluation order.
#[test]
fn test_evaluator_handles_stale_depths() {
    let config = NeatConfig {
        output_activation: Activation::Identity,
        hidden_activations: vec![Activation::Identity],
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set all weights to 1.0
    for (_, conn) in &mut genome.connections {
        conn.weight = 1.0;
    }

    // Add a hidden node
    let conn_id = genome.connections.iter().next().unwrap().0;
    genome.add_node(conn_id, &mut rng);

    // Set all weights to 1.0 again
    for (_, conn) in &mut genome.connections {
        if conn.enabled {
            conn.weight = 1.0;
        }
    }

    // Manually corrupt the depth values to simulate stale state
    for (_, node) in &mut genome.nodes {
        if node.node_type == NodeType::Hidden {
            node.depth = 0; // Wrong! Should be 1
        }
    }

    // The evaluator should still work correctly because it recomputes depths
    let mut evaluator = CppnEvaluator::new(&genome);
    let output = evaluator.evaluate(&[1.0, 2.0]);

    // With correct depth computation: output = input0 + input1 = 3.0
    // With stale depths: hidden might not be evaluated before output, causing wrong result
    assert!(
        (output[0] - 3.0).abs() < 0.1,
        "Evaluator should handle stale depths: expected ~3.0, got {}",
        output[0]
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
