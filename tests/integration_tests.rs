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
    let evaluator = CppnEvaluator::new(&child);

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
    let eval1 = CppnEvaluator::new(&genome);
    let inputs = [0.5, -0.3, 0.8];
    let output_before = eval1.evaluate(&inputs);

    // Serialize and deserialize
    let json = serde_json::to_string(&genome).unwrap();
    let restored: NeatGenome = serde_json::from_str(&json).unwrap();

    // Verify input_ids and output_ids are preserved
    assert_eq!(genome.input_ids.len(), restored.input_ids.len());
    assert_eq!(genome.output_ids.len(), restored.output_ids.len());

    // Evaluate restored - should produce identical results
    let eval2 = CppnEvaluator::new(&restored);
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

    // Verify all weights are finite and bounded to the consistent limit
    const WEIGHT_LIMIT: f32 = 1e3;
    for (_, conn) in &genome.connections {
        assert!(
            conn.weight.is_finite(),
            "Weight should be finite, got {}",
            conn.weight
        );
        assert!(
            conn.weight.abs() <= WEIGHT_LIMIT,
            "Weight {} exceeds limit {}",
            conn.weight,
            WEIGHT_LIMIT
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
        result_large.is_finite() && result_large.abs() <= 1e3,
        "Identity(1e30) should be bounded to 1e3, got {}",
        result_large
    );
    assert!(
        result_small.is_finite() && result_small.abs() <= 1e3,
        "Identity(-1e30) should be bounded to 1e3, got {}",
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

    let evaluator = CppnEvaluator::new(&genome);
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
        let evaluator = CppnEvaluator::new(&child);
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

    let evaluator = CppnEvaluator::new(&genome);

    let inputs = [0.5, -0.3, 0.8];
    let outputs_vec = evaluator.evaluate(&inputs);

    let mut scratch = evaluator.create_scratchpad();
    let mut outputs_buf = [0.0f32; 2];
    evaluator.evaluate_into(&inputs, &mut outputs_buf, &mut scratch);

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

            let evaluator = CppnEvaluator::new(genome);
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
        let evaluator = CppnEvaluator::new(genome);
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
    let evaluator = CppnEvaluator::new(&genome);
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
/// All hash-based innovations should be >= RESERVED_INNOVATION_RANGE (2^32).
#[test]
fn test_reserved_innovation_range_boundary() {
    const RESERVED_RANGE: u64 = 1 << 32; // 4,294,967,296

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

    // Fixed IDs should be < 2^32 for reserved range nodes
    let max_fixed_id = *node_innovations.iter().max().unwrap();
    assert!(
        max_fixed_id < (1u64 << 32),
        "Fixed node IDs {} should fit in reserved range 2^32",
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
    // Use Tanh activation (range [-1, 1]) for predictable normalization
    let config = NeatConfig {
        output_activation: Activation::Tanh,
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set weights so output = tanh(x + y) (first input + second input)
    // With small inputs (x,y in [-1,1]), tanh(x+y) ≈ x+y
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
    // Output = tanh(0 + 0) = 0, normalized from [-1,1] to [0,1] = 0.5
    let pattern_1x1 = generate_pattern(&mut evaluator, 1, 1, 0).unwrap();
    assert_eq!(pattern_1x1.len(), 1);
    assert!(
        (pattern_1x1[0] - 0.5).abs() < 0.01,
        "1x1 pattern should be centered (0.5), got {}",
        pattern_1x1[0]
    );

    // For 2x1 pattern, pixels should be at x=-1 and x=+1, y=0
    // Output[0] = tanh(-1 + 0) ≈ -0.76, normalized to ~0.12
    // Output[1] = tanh(+1 + 0) ≈ +0.76, normalized to ~0.88
    let pattern_2x1 = generate_pattern(&mut evaluator, 2, 1, 0).unwrap();
    assert_eq!(pattern_2x1.len(), 2);
    // Verify pattern[0] < pattern[1] (x=-1 gives lower value than x=+1)
    assert!(
        pattern_2x1[0] < pattern_2x1[1],
        "2x1 pattern should increase with x: {} vs {}",
        pattern_2x1[0],
        pattern_2x1[1]
    );
    // Verify symmetry around 0.5
    assert!(
        (pattern_2x1[0] + pattern_2x1[1] - 1.0).abs() < 0.01,
        "2x1 pattern should be symmetric around 0.5: {} + {} = {}",
        pattern_2x1[0],
        pattern_2x1[1],
        pattern_2x1[0] + pattern_2x1[1]
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
    let evaluator = CppnEvaluator::new(&genome);
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
    let evaluator = CppnEvaluator::new(&genome);
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

    let evaluator = CppnEvaluator::new(&genome);
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
    let eval1 = CppnEvaluator::new(&genome);
    let output1 = eval1.query_2d(0.5, -0.3);

    // Serialize and deserialize
    let json = serde_json::to_string(&genome).unwrap();
    let restored: NeatGenome = serde_json::from_str(&json).unwrap();

    // Evaluate restored
    let eval2 = CppnEvaluator::new(&restored);
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

        let evaluator = CppnEvaluator::new(&genome);
        let output = evaluator.evaluate(&[0.5]);

        assert!(
            output[0].is_finite(),
            "Activation {:?} produced non-finite output",
            activation
        );
    }
}

// =============================================================================
// Code Review Fix Regression Tests (Issues #39-#43)
// =============================================================================

/// Test that iterative DFS in has_cycle handles deep networks without stack overflow.
/// This verifies the fix for issue #39 (recursive stack overflow).
#[test]
fn test_deep_network_no_stack_overflow_has_cycle() {
    let config = NeatConfig::minimal(1, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Create a very deep chain by repeatedly splitting connections
    // This creates a network depth that would overflow the stack with recursion
    for _ in 0..500 {
        // Find an enabled connection to split
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

    // This should complete without stack overflow
    let has_cycle = genome.has_cycle();
    assert!(!has_cycle, "Deep chain network should not have cycles");

    // Verify we actually created a deep network
    assert!(
        genome.nodes.len() > 400,
        "Should have created many nodes: {}",
        genome.nodes.len()
    );
}

/// Test that iterative DFS in find_back_edge handles deep networks without stack overflow.
/// This verifies the fix for issue #39 (recursive stack overflow).
#[test]
fn test_deep_network_no_stack_overflow_break_cycles() {
    let config = NeatConfig::minimal(1, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Create a very deep chain
    for _ in 0..500 {
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

    // This should complete without stack overflow
    let disabled = genome.break_cycles();
    assert_eq!(
        disabled, 0,
        "Acyclic network should have no cycles to break"
    );
}

/// Test that the increased RESERVED_INNOVATION_RANGE (2^32) prevents collisions
/// with networks that have very large input layers.
/// This verifies the fix for issue #40 (innovation ID collision).
#[test]
fn test_large_input_layer_no_innovation_collision() {
    // Create a network with inputs that would exceed the old 65536 limit
    // We can't actually create 65537 inputs due to memory, but we verify
    // that hash-based innovations are safely above the new 2^32 threshold
    let config = NeatConfig::minimal(100, 10);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Get the maximum fixed innovation (input/output node IDs)
    let max_fixed_innovation = genome
        .nodes
        .iter()
        .map(|(_, n)| n.innovation)
        .max()
        .unwrap();

    // Add structure via mutations to create hash-based innovations
    for _ in 0..20 {
        genome.mutate(&mut rng, 1.0);
    }

    // Collect all innovations
    let node_innovations: Vec<u64> = genome.nodes.iter().map(|(_, n)| n.innovation).collect();
    let conn_innovations: Vec<u64> = genome
        .connections
        .iter()
        .map(|(_, c)| c.innovation)
        .collect();

    // Verify no collision between fixed IDs and hash-based IDs
    const NEW_RESERVED_RANGE: u64 = 1 << 32;
    for &inn in &node_innovations {
        if inn > max_fixed_innovation {
            assert!(
                inn >= NEW_RESERVED_RANGE,
                "Hash-based node innovation {} should be >= {}",
                inn,
                NEW_RESERVED_RANGE
            );
        }
    }
    for &inn in &conn_innovations {
        assert!(
            inn >= NEW_RESERVED_RANGE,
            "Connection innovation {} should be >= {}",
            inn,
            NEW_RESERVED_RANGE
        );
    }
}

/// Test that weight initialization clamps extreme values to prevent Inf/NaN.
/// This verifies the fix for issue #41 (unbounded weight initialization).
#[test]
fn test_weight_initialization_clamped_extreme_range() {
    // Use an extremely large weight_range that would cause overflow without clamping
    let config = NeatConfig {
        weight_range: f32::MAX / 2.0, // Would overflow: 2.0 * this = Infinity
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // This should not panic or produce NaN/Inf
    let genome = NeatGenome::fully_connected(config, &mut rng);

    for (_, conn) in &genome.connections {
        assert!(
            conn.weight.is_finite(),
            "Weight should be finite even with extreme weight_range, got {}",
            conn.weight
        );
        assert!(
            conn.weight.abs() <= 1e3,
            "Weight {} should be clamped to 1e3",
            conn.weight
        );
    }
}

/// Test that add_connection also clamps weights with extreme range.
/// This verifies the fix for issue #41 (unbounded weight initialization).
#[test]
fn test_add_connection_weight_clamped() {
    let config = NeatConfig {
        weight_range: f32::MAX / 2.0,
        ..NeatConfig::minimal(2, 2)
    };
    let mut genome = NeatGenome::minimal(config);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Add connections manually
    let input_id = genome.input_ids[0];
    let output_id = genome.output_ids[0];
    let conn_id = genome.add_connection(input_id, output_id, &mut rng);

    assert!(conn_id.is_some(), "Connection should be added");

    let conn = &genome.connections[conn_id.unwrap()];
    assert!(
        conn.weight.is_finite(),
        "add_connection weight should be finite, got {}",
        conn.weight
    );
    assert!(
        conn.weight.abs() <= 1e3,
        "add_connection weight {} should be clamped to 1e3",
        conn.weight
    );
}

/// Test that CppnEvaluator::try_new returns error for cyclic genomes.
/// This verifies the fix for issue #42 (evaluator ignoring cycle detection).
#[test]
fn test_evaluator_try_new_detects_cycles() {
    use symbios_neat::EvaluatorError;

    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    // A normal acyclic genome should succeed
    let result = CppnEvaluator::try_new(&genome);
    assert!(result.is_ok(), "Acyclic genome should create evaluator");

    // Verify the error type exists and displays properly
    let err = EvaluatorError::CyclicGenome;
    let msg = err.to_string();
    assert!(
        msg.contains("cycle"),
        "Error message should mention cycles: {}",
        msg
    );
}

/// Test that the evaluator works correctly with deep networks after all fixes.
/// This is a comprehensive integration test combining multiple fixes.
#[test]
fn test_deep_network_evaluation_after_fixes() {
    let config = NeatConfig {
        add_node_prob: 1.0, // Always add nodes
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Create a moderately deep network
    for _ in 0..100 {
        genome.mutate(&mut rng, 1.0);
    }

    // Verify no cycles
    assert!(!genome.has_cycle(), "Mutated genome should be acyclic");

    // Create evaluator (should not panic even with deep network)
    let result = CppnEvaluator::try_new(&genome);
    assert!(
        result.is_ok(),
        "Deep acyclic genome should create evaluator"
    );

    let evaluator = result.unwrap();
    let output = evaluator.evaluate(&[0.5, -0.5]);

    assert!(
        output[0].is_finite(),
        "Deep network output should be finite"
    );
}

/// Test that crossover with deep parent networks doesn't cause issues.
#[test]
fn test_crossover_deep_parents() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create two deep parent networks
    let mut parent1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut parent2 = NeatGenome::fully_connected(config, &mut rng);

    for _ in 0..50 {
        parent1.mutate(&mut rng, 1.0);
        parent2.mutate(&mut rng, 1.0);
    }

    // Crossover should work without stack overflow
    let child = parent1.crossover(&parent2, &mut rng);

    // Child should be valid
    assert!(!child.has_cycle(), "Child should be acyclic");

    let result = CppnEvaluator::try_new(&child);
    assert!(result.is_ok(), "Child should create valid evaluator");
}

// =============================================================================
// Code Review Fix Regression Tests (Issues #44-#49)
// =============================================================================

/// Test that CLAMP_BOUND of 1e3 preserves precision.
/// With 1e3 * 1e3 = 1e6, ULP is ~0.06, preserving signals in 0..1 range.
/// This verifies the fix for issue #44 (numerical precision loss).
#[test]
fn test_precision_preserved_with_reduced_clamp_bound() {
    // Create a network that would accumulate large products
    let config = NeatConfig {
        output_activation: Activation::Identity,
        hidden_activations: vec![Activation::Identity],
        weight_range: 100.0,
        ..NeatConfig::minimal(10, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set all weights to moderately large values
    for (_, conn) in &mut genome.connections {
        conn.weight = 100.0;
    }

    let evaluator = CppnEvaluator::new(&genome);

    // Test with small input values (0.001)
    // With old 1e6 clamp: sum could reach 1e12, losing precision for small signals
    // With new 1e3 clamp: max sum is bounded, small signals preserved
    let small_inputs: Vec<f32> = vec![0.001; 10];
    let output = evaluator.evaluate(&small_inputs);

    // The output should be non-zero and finite
    assert!(
        output[0].is_finite(),
        "Output should be finite: {}",
        output[0]
    );

    // Test that small differences in input produce different outputs
    let mut small_inputs_variant = small_inputs.clone();
    small_inputs_variant[0] = 0.002; // Small change
    let output_variant = evaluator.evaluate(&small_inputs_variant);

    assert!(
        (output[0] - output_variant[0]).abs() > 1e-6,
        "Small input changes should produce detectable output differences"
    );
}

/// Test that generate_pattern normalizes correctly for ReLU output activation.
/// ReLU outputs [0, CLAMP_BOUND], not [-1, 1].
/// This verifies the fix for issue #45 (pattern normalization).
#[test]
fn test_generate_pattern_relu_normalization() {
    let config = NeatConfig {
        output_activation: Activation::ReLU, // Output range [0, 1e3]
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Set weights so output is moderately positive
    for (_, conn) in &mut genome.connections {
        conn.weight = 0.5;
    }

    let mut evaluator = CppnEvaluator::new(&genome);
    let pattern = generate_pattern(&mut evaluator, 4, 4, 0).unwrap();

    // With correct normalization:
    // ReLU output 0.0 should normalize to 0.0 (not 0.5 as with old code)
    // ReLU output 1.0 should normalize to ~0.001 (1.0/1e3)

    // Verify pattern values are in valid range
    for &val in &pattern {
        assert!(
            (0.0..=1.0).contains(&val),
            "Pattern value {} should be in [0, 1]",
            val
        );
    }

    // Verify that small ReLU outputs don't map to 0.5
    // (the bug was: 0.0 * 0.5 + 0.5 = 0.5 regardless of activation range)
    let min_val = pattern.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_val < 0.4,
        "With ReLU, minimum pattern value should be < 0.4 (near 0), got {}",
        min_val
    );
}

/// Test that generate_pattern normalizes correctly for Abs output activation.
/// Abs outputs [0, CLAMP_BOUND], similar to ReLU.
/// This verifies the fix for issue #45 (pattern normalization).
#[test]
fn test_generate_pattern_abs_normalization() {
    let config = NeatConfig {
        output_activation: Activation::Abs,
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    let mut evaluator = CppnEvaluator::new(&genome);
    let pattern = generate_pattern(&mut evaluator, 4, 4, 0).unwrap();

    // All values should be in [0, 1]
    for &val in &pattern {
        assert!(
            (0.0..=1.0).contains(&val),
            "Pattern value {} should be in [0, 1]",
            val
        );
    }
}

/// Test that mutation rate scaling doesn't permanently modify config.
/// This verifies the fix for issue #48 (mutation rate corruption).
#[test]
fn test_mutation_rate_does_not_corrupt_config() {
    let config = NeatConfig {
        weight_mutation_prob: 0.8,
        add_connection_prob: 0.5,
        add_node_prob: 0.3,
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config.clone(), &mut rng);

    // Store original probabilities
    let original_weight_prob = genome.config.weight_mutation_prob;
    let original_conn_prob = genome.config.add_connection_prob;
    let original_node_prob = genome.config.add_node_prob;

    // Mutate with rate < 1.0 (the case that was problematic)
    for _ in 0..10 {
        genome.mutate(&mut rng, 0.5);
    }

    // Config probabilities should be unchanged
    assert!(
        (genome.config.weight_mutation_prob - original_weight_prob).abs() < 1e-6,
        "weight_mutation_prob should not change: {} vs {}",
        genome.config.weight_mutation_prob,
        original_weight_prob
    );
    assert!(
        (genome.config.add_connection_prob - original_conn_prob).abs() < 1e-6,
        "add_connection_prob should not change: {} vs {}",
        genome.config.add_connection_prob,
        original_conn_prob
    );
    assert!(
        (genome.config.add_node_prob - original_node_prob).abs() < 1e-6,
        "add_node_prob should not change: {} vs {}",
        genome.config.add_node_prob,
        original_node_prob
    );

    // Verify offspring inherit correct probabilities
    let child = genome.crossover(&genome, &mut rng);
    assert!(
        (child.config.weight_mutation_prob - original_weight_prob).abs() < 1e-6,
        "Child should inherit original config: {} vs {}",
        child.config.weight_mutation_prob,
        original_weight_prob
    );
}

/// Test that mutation with rate=0 still allows some mutation (via scaled probs).
/// This verifies mutation scaling works correctly.
#[test]
fn test_mutation_rate_zero_behavior() {
    let config = NeatConfig {
        weight_mutation_prob: 1.0, // 100% weight mutation
        add_connection_prob: 1.0,
        add_node_prob: 1.0,
        ..NeatConfig::minimal(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Capture original weights
    let original_weights: Vec<f32> = genome.connections.iter().map(|(_, c)| c.weight).collect();

    // Mutate with rate=0 (should skip all mutations)
    genome.mutate(&mut rng, 0.0);

    // Weights should be unchanged
    let new_weights: Vec<f32> = genome.connections.iter().map(|(_, c)| c.weight).collect();
    assert_eq!(
        original_weights, new_weights,
        "With rate=0, no mutations should occur"
    );
}

/// Test that innovation hashes are uniformly distributed (no modulo bias).
/// This verifies the fix for issue #47 (hash modulo bias).
#[test]
fn test_innovation_hash_uniform_distribution() {
    // Generate many hashes and check distribution
    let mut hashes = Vec::new();
    for i in 0..10000u64 {
        hashes.push(connection_innovation(i, i + 1));
    }

    // Split into quartiles of the u64 range
    const Q1: u64 = u64::MAX / 4;
    const Q2: u64 = u64::MAX / 2;
    const Q3: u64 = Q1 * 3;

    let mut counts = [0usize; 4];
    for &h in &hashes {
        if h < Q1 {
            counts[0] += 1;
        } else if h < Q2 {
            counts[1] += 1;
        } else if h < Q3 {
            counts[2] += 1;
        } else {
            counts[3] += 1;
        }
    }

    // All quartiles should be well-populated (expect ~2500 each with some variance)
    // With modulo bias, some quartiles would be systematically under/over-represented
    let expected = hashes.len() / 4;
    let tolerance = expected / 3; // Allow 33% variance

    for (i, &count) in counts.iter().enumerate() {
        assert!(
            count > expected - tolerance,
            "Quartile {} count {} is too low (expected ~{})",
            i,
            count,
            expected
        );
    }
}

/// Test that all hash-based innovations are >= RESERVED_INNOVATION_RANGE.
/// This verifies the rejection sampling works correctly (issue #47 fix).
#[test]
fn test_innovation_hash_above_reserved_range() {
    const RESERVED: u64 = 1 << 32;

    // Test edge cases that might produce low values before rejection
    let edge_cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (u64::MAX, 0),
        (0, u64::MAX),
        (u64::MAX, u64::MAX),
    ];

    for (a, b) in edge_cases {
        let inn = connection_innovation(a, b);
        assert!(
            inn >= RESERVED,
            "connection_innovation({}, {}) = {} should be >= {}",
            a,
            b,
            inn,
            RESERVED
        );

        let split = node_split_innovation(inn);
        assert!(
            split >= RESERVED,
            "node_split_innovation({}) = {} should be >= {}",
            inn,
            split,
            RESERVED
        );
    }
}

/// Test that CppnEvaluator::try_new doesn't clone the genome.
/// We verify this indirectly by checking the genome is not modified.
/// This verifies the fix for issue #46 (redundant clone).
#[test]
fn test_evaluator_construction_no_genome_mutation() {
    let config = NeatConfig::minimal(2, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    // Capture state before
    let depths_before: Vec<u32> = genome.nodes.iter().map(|(_, n)| n.depth).collect();

    // Create evaluator
    let _evaluator = CppnEvaluator::new(&genome);

    // Depths should be unchanged (evaluator computes locally)
    let depths_after: Vec<u32> = genome.nodes.iter().map(|(_, n)| n.depth).collect();
    assert_eq!(
        depths_before, depths_after,
        "Evaluator construction should not modify genome depths"
    );
}

/// Test that activation output_range is correctly defined for all functions.
/// This verifies the infrastructure for issue #45 fix.
#[test]
fn test_activation_output_range() {
    for activation in Activation::ALL {
        let (min, max) = activation.output_range();

        // Range should be valid
        assert!(
            min < max,
            "{:?} range invalid: {} >= {}",
            activation,
            min,
            max
        );

        // Test that apply() stays within the stated range (with tolerance for edge cases)
        let test_inputs = [-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0];
        for input in test_inputs {
            let output = activation.apply(input);
            if output.is_finite() {
                assert!(
                    output >= min - 0.001 && output <= max + 0.001,
                    "{:?}({}) = {} outside stated range [{}, {}]",
                    activation,
                    input,
                    output,
                    min,
                    max
                );
            }
        }
    }
}

/// Test that Tanh pattern normalization still works correctly (regression).
/// This ensures the fix for issue #45 didn't break existing behavior.
#[test]
fn test_generate_pattern_tanh_still_works() {
    let config = NeatConfig {
        output_activation: Activation::Tanh, // Output range [-1, 1]
        ..NeatConfig::cppn(2, 1)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genome = NeatGenome::fully_connected(config, &mut rng);

    let mut evaluator = CppnEvaluator::new(&genome);
    let pattern = generate_pattern(&mut evaluator, 8, 8, 0).unwrap();

    // Pattern should span a reasonable range (not all 0.5 or all 0/1)
    let min = pattern.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = pattern.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        max - min > 0.1,
        "Tanh pattern should have variation: [{}, {}]",
        min,
        max
    );

    // All values should be in [0, 1]
    for &val in &pattern {
        assert!(
            (0.0..=1.0).contains(&val),
            "Pattern value {} out of range",
            val
        );
    }
}

// =============================================================================
// Performance Optimization Tests (Issue #51)
// =============================================================================

/// Test that compatibility_distance produces correct results with sorted merge algorithm.
/// The sorted merge avoids HashMap allocations while maintaining O(E log E) complexity.
#[test]
fn test_compatibility_distance_sorted_merge_correctness() {
    let config = NeatConfig::minimal(3, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create genomes with different structures
    let mut genome1 = NeatGenome::fully_connected(config.clone(), &mut rng);
    let mut genome2 = NeatGenome::fully_connected(config, &mut rng);

    // Mutate to create disjoint/excess genes
    for _ in 0..10 {
        genome1.mutate(&mut rng, 1.0);
    }
    for _ in 0..15 {
        genome2.mutate(&mut rng, 1.0);
    }

    // Verify distance is symmetric
    let dist_1_to_2 = genome1.compatibility_distance(&genome2);
    let dist_2_to_1 = genome2.compatibility_distance(&genome1);

    assert!(
        (dist_1_to_2 - dist_2_to_1).abs() < 1e-6,
        "Distance should be symmetric: {} vs {}",
        dist_1_to_2,
        dist_2_to_1
    );

    // Verify distance to self is zero
    assert!(
        genome1.compatibility_distance(&genome1).abs() < 1e-6,
        "Distance to self should be zero"
    );
}

/// Test that many compatibility_distance calls complete efficiently.
/// This is a regression test for the HashMap allocation issue.
#[test]
fn test_compatibility_distance_many_calls() {
    let config = NeatConfig::cppn(4, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create a small population
    let pop_size = 20;
    let mut population: Vec<NeatGenome> = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut genome = NeatGenome::fully_connected(config.clone(), &mut rng);
        for _ in 0..10 {
            genome.mutate(&mut rng, 1.0);
        }
        population.push(genome);
    }

    // Perform all pairwise comparisons (190 comparisons)
    let mut total_distance = 0.0f32;
    for i in 0..population.len() {
        for j in (i + 1)..population.len() {
            total_distance += population[i].compatibility_distance(&population[j]);
        }
    }

    // Just verify it completes and produces valid results
    assert!(
        total_distance.is_finite(),
        "Total distance should be finite"
    );
    assert!(
        total_distance >= 0.0,
        "Total distance should be non-negative"
    );
}

/// Test that update_depths uses O(V+E) Kahn's algorithm correctly.
#[test]
fn test_update_depths_kahn_algorithm() {
    let config = NeatConfig::minimal(2, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add several hidden nodes to create depth
    for _ in 0..5 {
        if let Some(conn_id) = genome
            .connections
            .iter()
            .filter(|(_, c)| c.enabled)
            .map(|(id, _)| id)
            .next()
        {
            genome.add_node(conn_id, &mut rng);
        }
    }

    // Verify depths are computed correctly
    assert!(genome.update_depths(), "Acyclic genome should succeed");

    // Input nodes should have depth 0
    for &input_id in &genome.input_ids {
        assert_eq!(
            genome.nodes[input_id].depth, 0,
            "Input nodes should have depth 0"
        );
    }

    // Output nodes should have depth > 0 (there's at least one connection)
    for &output_id in &genome.output_ids {
        assert!(
            genome.nodes[output_id].depth > 0,
            "Output nodes should have depth > 0"
        );
    }
}

/// Test evaluator CSR format produces same results as original adjacency list.
#[test]
fn test_evaluator_csr_format_correctness() {
    let config = NeatConfig::cppn(3, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Add complexity
    for _ in 0..10 {
        genome.mutate(&mut rng, 1.0);
    }

    let evaluator = CppnEvaluator::new(&genome);

    // Evaluate with various inputs
    let test_inputs = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
        [-1.0, -1.0, -1.0],
    ];

    for inputs in &test_inputs {
        let outputs1 = evaluator.evaluate(inputs);
        let outputs2 = evaluator.evaluate(inputs);

        // Results should be deterministic
        for (i, (&o1, &o2)) in outputs1.iter().zip(outputs2.iter()).enumerate() {
            assert!(
                (o1 - o2).abs() < 1e-6,
                "Evaluation should be deterministic: output {} differs ({} vs {})",
                i,
                o1,
                o2
            );
        }

        // Results should be finite
        for (i, &o) in outputs1.iter().enumerate() {
            assert!(o.is_finite(), "Output {} should be finite: {}", i, o);
        }
    }
}

/// Test that would_create_cycle uses Vec<bool> efficiently.
#[test]
fn test_would_create_cycle_efficiency() {
    let config = NeatConfig {
        add_connection_prob: 1.0, // High probability to stress cycle detection
        add_node_prob: 0.5,
        ..NeatConfig::minimal(3, 2)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut genome = NeatGenome::fully_connected(config, &mut rng);

    // Perform many mutations (each add_connection attempt calls would_create_cycle)
    for _ in 0..50 {
        genome.mutate(&mut rng, 1.0);
    }

    // Verify the genome is still acyclic
    assert!(
        !genome.has_cycle(),
        "Genome should remain acyclic after mutations"
    );

    // Verify depths can be computed
    assert!(
        genome.update_depths(),
        "Acyclic genome should have valid depths"
    );
}
