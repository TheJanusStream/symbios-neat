//! NEAT adapter for the generic [`symbios_genetics::speciation`] primitives.
//!
//! Provides [`NeatDistance`], a zero-sized [`CompatibilityDistance`] impl that
//! delegates to [`NeatGenome::compatibility_distance`]. Use it to plug a NEAT
//! population into [`Speciation`](symbios_genetics::speciation::Speciation).
//!
//! # Example
//!
//! ```rust
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//! use symbios_genetics::{
//!     speciation::Speciation,
//!     Genotype, Phenotype,
//! };
//! use symbios_neat::{species::NeatDistance, NeatConfig, NeatGenome};
//!
//! let config = NeatConfig::cppn(2, 1);
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//!
//! let mut population: Vec<Phenotype<NeatGenome>> = (0..30)
//!     .map(|_| {
//!         let mut g = NeatGenome::fully_connected(config.clone(), &mut rng);
//!         for _ in 0..5 { g.mutate(&mut rng, 1.0); }
//!         Phenotype { genotype: g, fitness: 1.0, objectives: vec![], descriptor: vec![] }
//!     })
//!     .collect();
//!
//! let mut speciation = Speciation::new(NeatDistance, 1.0, 5);
//! speciation.assign(&population);
//! speciation.share_fitness(&mut population);
//! speciation.adjust_threshold();
//! assert!(!speciation.species().is_empty());
//! ```

use symbios_genetics::speciation::CompatibilityDistance;

use crate::NeatGenome;

/// Zero-sized [`CompatibilityDistance`] adapter that calls
/// [`NeatGenome::compatibility_distance`].
///
/// The compatibility coefficients (`compatibility_excess_coeff`,
/// `compatibility_disjoint_coeff`, `compatibility_weight_coeff`) come from each
/// genome's `NeatConfig`, so they can vary across genomes. In typical usage all
/// genomes in a population share a config and the distance is symmetric.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeatDistance;

impl CompatibilityDistance<NeatGenome> for NeatDistance {
    fn distance(&self, a: &NeatGenome, b: &NeatGenome) -> f32 {
        a.compatibility_distance(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use symbios_genetics::{Genotype, Phenotype, speciation::Speciation};

    use crate::NeatConfig;

    fn make_population(seed: u64, size: usize, mut_steps: usize) -> Vec<Phenotype<NeatGenome>> {
        let config = NeatConfig {
            add_connection_prob: 0.3,
            add_node_prob: 0.1,
            ..NeatConfig::cppn(4, 2)
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..size)
            .map(|_| {
                let mut g = NeatGenome::fully_connected(config.clone(), &mut rng);
                for _ in 0..mut_steps {
                    g.mutate(&mut rng, 1.0);
                }
                Phenotype {
                    genotype: g,
                    fitness: 1.0,
                    objectives: vec![],
                    descriptor: vec![],
                }
            })
            .collect()
    }

    /// Build a population with varied structural complexity so compatibility
    /// distances span a wide range. Each genome receives a different number of
    /// mutation steps, producing a range of topologies and innovation patterns.
    fn make_diverse_population(seed: u64, size: usize) -> Vec<Phenotype<NeatGenome>> {
        let config = NeatConfig {
            add_connection_prob: 0.4,
            add_node_prob: 0.3,
            ..NeatConfig::cppn(4, 2)
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..size)
            .map(|i| {
                let mut g = NeatGenome::fully_connected(config.clone(), &mut rng);
                // Vary mutation depth across the population (1..=2*size mutations).
                let depth = 1 + (i * 2);
                for _ in 0..depth {
                    g.mutate(&mut rng, 1.0);
                }
                Phenotype {
                    genotype: g,
                    fitness: 1.0,
                    objectives: vec![],
                    descriptor: vec![],
                }
            })
            .collect()
    }

    #[test]
    fn neat_distance_matches_genome_method() {
        let pop = make_population(7, 2, 3);
        let direct = pop[0].genotype.compatibility_distance(&pop[1].genotype);
        let via_trait = NeatDistance.distance(&pop[0].genotype, &pop[1].genotype);
        assert!((direct - via_trait).abs() < f32::EPSILON);
    }

    #[test]
    fn speciation_assigns_neat_population() {
        let pop = make_population(9, 30, 5);
        let mut spec = Speciation::new(NeatDistance, 1.0, 5);
        spec.assign(&pop);
        let total: usize = spec.species().iter().map(|s| s.member_indices.len()).sum();
        assert_eq!(
            total,
            pop.len(),
            "every phenotype should be in some species"
        );
    }

    #[test]
    fn species_count_converges_toward_target_for_neat() {
        // Use varied mutation depth so compatibility distances span a wide
        // range — otherwise threshold tuning has no axis to act on.
        let pop = make_diverse_population(17, 40);

        // Track best (closest-to-target) count seen during the run. Since the
        // population is static, the threshold/count map is a step function and
        // a fixed-point step-size search will oscillate around discrete count
        // values; validate that the algorithm visits the target band at all.
        let target = 6;
        let mut spec = Speciation::new(NeatDistance, 2.0, target)
            .with_threshold_step(0.05)
            .with_min_threshold(0.05);

        let mut min_diff = usize::MAX;
        for _ in 0..200 {
            spec.assign(&pop);
            let diff = spec.species().len().abs_diff(target);
            min_diff = min_diff.min(diff);
            spec.adjust_threshold();
        }

        assert!(
            min_diff <= 2,
            "best species-count distance from target {target} was {min_diff}; \
             threshold tuning failed to drive count near target"
        );
    }

    #[test]
    fn fitness_sharing_reduces_dominant_species_pressure() {
        let mut pop = make_population(3, 20, 1);
        for p in &mut pop {
            p.fitness = 10.0;
        }
        let mut spec = Speciation::new(NeatDistance, 5.0, 1); // big threshold -> one species
        spec.assign(&pop);
        spec.share_fitness(&mut pop);

        assert_eq!(spec.species().len(), 1);
        for p in &pop {
            assert!(
                (p.fitness - 10.0 / 20.0).abs() < 1e-5,
                "fitness should be raw / |species| = 0.5, got {}",
                p.fitness
            );
        }
    }
}
