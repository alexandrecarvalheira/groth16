# Elliptic Curves in Cryptography

## Overview

Elliptic curves are mathematical structures that form the foundation of modern cryptographic systems, including Groth16. This document explains elliptic curves and their implementation in the Arkworks ecosystem.

## Elliptic Curve Fundamentals

### 1. Basic Definition

An elliptic curve over a field K is defined by the Weierstrass equation:
```
y² = x³ + ax + b
```
where a, b ∈ K, and 4a³ + 27b² ≠ 0

In cryptography, we typically use curves over finite fields:

```rust
use ark_ff::PrimeField;
use ark_ec::{AffineRepr, CurveGroup};

// Example using BN254 curve
use ark_bn254::{Fr, G1Affine, G1Projective};

// Point on the curve
let g = G1Projective::generator();
let scalar = Fr::from(42);
let point = g.mul(scalar);
```

### 2. Point Representation

Points can be represented in different coordinate systems:

#### Affine Coordinates
```rust
// Point in affine coordinates (x, y)
pub struct G1Affine<P: Parameters> {
    pub x: P::BaseField,
    pub y: P::BaseField,
    pub infinity: bool,
}
```

#### Projective Coordinates
```rust
// Point in projective coordinates (X: Y: Z)
pub struct G1Projective<P: Parameters> {
    pub x: P::BaseField,
    pub y: P::BaseField,
    pub z: P::BaseField,
}
```

### 3. Group Operations

Elliptic curve points form a group with the following operations:

#### Point Addition
```rust
// Adding two points
let p1 = G1Projective::generator();
let p2 = G1Projective::generator().mul(Fr::from(2));
let sum = p1 + p2;
```

#### Scalar Multiplication
```rust
// Scalar multiplication (repeated addition)
fn scalar_mul<G: CurveGroup>(point: G, scalar: G::ScalarField) -> G {
    point.mul(scalar)
}
```

## Curves Used in Groth16

### 1. BN254 (Barreto-Naehrig)

Popular for zk-SNARKs due to its efficiency:

```rust
use ark_bn254::{Bn254, Fr, Fq};

// Field elements
let fr = Fr::from(1); // Scalar field
let fq = Fq::from(1); // Base field

// Curve points
let g1 = G1Projective::generator();
let g2 = G2Projective::generator();
```

Properties:
- Prime order: ~254 bits
- Embedding degree: 12
- Optimal ate pairing

### 2. Other Supported Curves

```rust
// BLS12-381
use ark_bls12_381::{Bls12_381, Fr as BlsFr};

// MNT4/6
use ark_mnt4_298::{MNT4_298, Fr as Mnt4Fr};
use ark_mnt6_298::{MNT6_298, Fr as Mnt6Fr};
```

## Implementation Details

### 1. Field Arithmetic

Efficient field operations are crucial:

```rust
// Field arithmetic in Rust
fn field_operations<F: PrimeField>(a: F, b: F) -> F {
    let sum = a + b;
    let product = a * b;
    let inverse = a.inverse().unwrap();
    let square = a.square();
    product * inverse + square
}
```

### 2. Point Arithmetic

Optimized point operations:

```rust
// Mixed addition (faster when one point is in affine form)
fn mixed_addition<G: CurveGroup>(
    p: &G::Projective,
    q: &G::Affine,
) -> G::Projective {
    p.add_mixed(q)
}

// Double-and-add scalar multiplication
fn double_and_add<G: CurveGroup>(
    p: &G,
    scalar: &G::ScalarField,
) -> G {
    let mut result = G::zero();
    let mut temp = *p;
    
    for bit in scalar.to_bits().iter().rev() {
        result = result.double();
        if *bit {
            result += temp;
        }
    }
    result
}
```

### 3. Serialization

Efficient point compression and serialization:

```rust
use ark_serialize::{CanonicalSerialize, Compress};

fn serialize_point<G: CurveGroup>(point: &G) -> Vec<u8> {
    let mut bytes = Vec::new();
    point.serialize_with_mode(&mut bytes, Compress::Yes)
        .expect("Serialization failed");
    bytes
}
```

## Security Considerations

### 1. Curve Selection Criteria

Important properties for cryptographic curves:

```rust
// Example security parameters check
fn check_curve_security<G: CurveGroup>() -> bool {
    // Prime order
    let order = G::ScalarField::MODULUS;
    
    // Large embedding degree
    let embedding_degree = 12; // for BN254
    
    // Check minimum bit size
    order.bits() >= 254
}
```

### 2. Constant-Time Operations

Preventing timing attacks:

```rust
// Constant-time scalar multiplication
fn constant_time_mul<G: CurveGroup>(
    point: &G,
    scalar: &G::ScalarField,
) -> G {
    // Use Montgomery ladder or similar constant-time algorithm
    point.mul_bits(scalar.to_bits().iter())
}
```

## Performance Optimization

### 1. Precomputation

```rust
// Window-based precomputation
struct PrecomputedPoint<G: CurveGroup> {
    multiples: Vec<G>,
}

impl<G: CurveGroup> PrecomputedPoint<G> {
    fn new(point: &G, window_size: usize) -> Self {
        let mut multiples = Vec::with_capacity(1 << window_size);
        let mut current = G::zero();
        for _ in 0..(1 << window_size) {
            multiples.push(current);
            current += point;
        }
        Self { multiples }
    }
}
```

### 2. Batch Operations

```rust
// Batch scalar multiplication
fn batch_mul<G: CurveGroup>(
    points: &[G],
    scalars: &[G::ScalarField],
) -> G {
    G::msm(points, scalars)
}
```

## Testing and Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_curve_operations() {
        // Group law tests
        let g = G1Projective::generator();
        let a = Fr::from(2);
        let b = Fr::from(3);
        
        // Scalar multiplication distributive property
        let left = g.mul(a + b);
        let right = g.mul(a) + g.mul(b);
        assert_eq!(left, right);
        
        // Point addition commutativity
        let p1 = g.mul(a);
        let p2 = g.mul(b);
        assert_eq!(p1 + p2, p2 + p1);
    }
}
```

## Next Steps

After understanding elliptic curves, we'll explore:
1. Bilinear pairings
2. Polynomial arithmetic
3. The specific curve operations in Groth16

## References

1. [Guide to Elliptic Curve Cryptography](https://link.springer.com/book/10.1007/b97644)
2. [Pairings for Beginners](https://www.craigcostello.com.au/pairings)
3. [Arkworks Curves Documentation](https://docs.rs/ark-ec) 