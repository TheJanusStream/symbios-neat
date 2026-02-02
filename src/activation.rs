//! Activation functions for NEAT networks.
//!
//! This module provides activation functions suitable for both traditional neural networks
//! and Compositional Pattern Producing Networks (CPPNs). CPPNs use periodic and symmetric
//! functions to generate natural patterns like ripples, segments, and bilateral symmetry.

use serde::{Deserialize, Serialize};

/// Activation function types supported by NEAT nodes.
///
/// For CPPN applications, periodic functions (Sine, Cosine) and symmetric functions
/// (Gaussian, Abs) are particularly useful for generating geometric patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Activation {
    /// Identity function: f(x) = x
    #[default]
    Identity,
    /// Sigmoid: f(x) = 1 / (1 + e^(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU,
    /// Sine function: f(x) = sin(x) - useful for periodic/wave patterns in CPPNs
    Sine,
    /// Cosine function: f(x) = cos(x) - useful for periodic/wave patterns in CPPNs
    Cosine,
    /// Gaussian: f(x) = e^(-x^2) - useful for radial patterns in CPPNs
    Gaussian,
    /// Absolute value: f(x) = |x| - useful for symmetric patterns in CPPNs
    Abs,
    /// Step function: f(x) = 1 if x > 0 else 0
    Step,
    /// Leaky ReLU: `f(x) = x` if `x > 0` else `0.01x`
    LeakyReLU,
}

impl Activation {
    /// All available activation functions.
    pub const ALL: [Self; 10] = [
        Self::Identity,
        Self::Sigmoid,
        Self::Tanh,
        Self::ReLU,
        Self::Sine,
        Self::Cosine,
        Self::Gaussian,
        Self::Abs,
        Self::Step,
        Self::LeakyReLU,
    ];

    /// CPPN-optimized activation functions (periodic and symmetric).
    pub const CPPN: [Self; 6] = [
        Self::Sigmoid,
        Self::Tanh,
        Self::Sine,
        Self::Cosine,
        Self::Gaussian,
        Self::Abs,
    ];

    /// Apply this activation function to an input value.
    ///
    /// All activation functions propagate NaN consistently.
    /// Extreme values (including infinity) are handled to produce finite outputs
    /// where mathematically sensible, ensuring numerical stability.
    #[inline]
    #[must_use]
    pub fn apply(self, x: f32) -> f32 {
        // Propagate NaN consistently across all activation functions
        if x.is_nan() {
            return f32::NAN;
        }

        match self {
            Self::Identity => x,
            Self::Sigmoid => {
                // Handle infinity: sigmoid(+inf) = 1, sigmoid(-inf) = 0
                if x == f32::INFINITY {
                    return 1.0;
                }
                if x == f32::NEG_INFINITY {
                    return 0.0;
                }
                // Clamp to avoid overflow: sigmoid(-88) ≈ 0, sigmoid(88) ≈ 1
                let clamped = x.clamp(-88.0, 88.0);
                1.0 / (1.0 + (-clamped).exp())
            }
            Self::Tanh => {
                // Handle infinity: tanh(+inf) = 1, tanh(-inf) = -1
                if x == f32::INFINITY {
                    return 1.0;
                }
                if x == f32::NEG_INFINITY {
                    return -1.0;
                }
                x.tanh()
            }
            Self::ReLU => {
                if x == f32::NEG_INFINITY {
                    return 0.0;
                }
                x.max(0.0)
            }
            Self::Sine => {
                // sin(infinity) is undefined; clamp to bounded range
                if x.is_infinite() {
                    return 0.0;
                }
                x.sin()
            }
            Self::Cosine => {
                // cos(infinity) is undefined; clamp to bounded range
                if x.is_infinite() {
                    return 0.0;
                }
                x.cos()
            }
            Self::Gaussian => {
                // gaussian(±inf) = 0
                if x.is_infinite() {
                    return 0.0;
                }
                // For |x| > 26, result is effectively 0 (exp(-676) ≈ 0)
                if x.abs() > 26.0 {
                    0.0
                } else {
                    (-x * x).exp()
                }
            }
            Self::Abs => {
                // abs(-inf) = +inf, abs(+inf) = +inf
                x.abs()
            }
            Self::Step => {
                // step(+inf) = 1, step(-inf) = 0
                if x > 0.0 || x == f32::INFINITY {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if x == f32::NEG_INFINITY {
                    return f32::NEG_INFINITY; // 0.01 * -inf = -inf
                }
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        assert!((Activation::Identity.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((Activation::Identity.apply(-2.0) - -2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-6);
        assert!(Activation::Sigmoid.apply(10.0) > 0.99);
        assert!(Activation::Sigmoid.apply(-10.0) < 0.01);
    }

    #[test]
    fn test_tanh() {
        assert!((Activation::Tanh.apply(0.0)).abs() < 1e-6);
        assert!(Activation::Tanh.apply(10.0) > 0.99);
        assert!(Activation::Tanh.apply(-10.0) < -0.99);
    }

    #[test]
    fn test_relu() {
        assert!((Activation::ReLU.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((Activation::ReLU.apply(-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_sine_cosine() {
        use std::f32::consts::PI;
        assert!(Activation::Sine.apply(0.0).abs() < 1e-6);
        assert!((Activation::Sine.apply(PI / 2.0) - 1.0).abs() < 1e-6);
        assert!((Activation::Cosine.apply(0.0) - 1.0).abs() < 1e-6);
        assert!(Activation::Cosine.apply(PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian() {
        assert!((Activation::Gaussian.apply(0.0) - 1.0).abs() < 1e-6);
        assert!(Activation::Gaussian.apply(3.0) < 0.001);
    }

    #[test]
    fn test_abs() {
        assert!((Activation::Abs.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((Activation::Abs.apply(-0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_step() {
        assert!((Activation::Step.apply(0.1) - 1.0).abs() < 1e-6);
        assert!(Activation::Step.apply(-0.1).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        assert!((Activation::LeakyReLU.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((Activation::LeakyReLU.apply(-1.0) - -0.01).abs() < 1e-6);
    }
}
