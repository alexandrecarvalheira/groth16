 # Quadratic Arithmetic Programs (QAP)

## Overview

Quadratic Arithmetic Programs (QAP) are an intermediate representation between R1CS and the final zero-knowledge proof. This document explains QAP structure, generation, and its role in Groth16.

## QAP Structure

### 1. Basic Definition

A QAP consists of:
- Sets of polynomials {u_i(X)}, {v_i(X)}, {w_i(X)}
- A target polynomial t(X)
- A solution is a set of values {a_i} such that:
```
(Σ a_i * u_i(X)) * (Σ a_i * v_i(X)) - (Σ a_i * w_i(X)) = h(X) * t(X)
```

```rust
use ark_poly::{
    univariate::DensePolynomial,
    UVPolynomial,
};

// QAP structure
struct QAP<F: Field> {
    u_polynomials: Vec<DensePolynomial<F>>,
    v_polynomials: Vec<DensePolynomial<F>>,
    w_polynomials: Vec<DensePolynomial<F>>,
    t_polynomial: DensePolynomial<F>,
}
```

### 2. Polynomial Sets

```rust
// Polynomial set generation
fn generate_polynomial_set<F: Field>(
    evaluations: &[Vec<F>],
    domain: &GeneralEvaluationDomain<F>,
) -> Vec<DensePolynomial<F>> {
    evaluations
        .iter()
        .map(|evals| {
            let mut padded = evals.clone();
            padded.resize(domain.size(), F::zero());
            DensePolynomial::from_coefficients_vec(domain.ifft(&padded))
        })
        .collect()
}
```

## QAP Generation from R1CS

### 1. Matrix to Polynomial Conversion

```rust
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

// Convert R1CS matrix to QAP polynomials
fn matrix_to_polynomials<F: Field>(
    matrix: &[Vec<F>],
    num_constraints: usize,
) -> Vec<DensePolynomial<F>> {
    let domain = GeneralEvaluationDomain::<F>::new(num_constraints)
        .expect("Failed to create domain");
    
    let num_variables = matrix[0].len();
    let mut polynomials = Vec::with_capacity(num_variables);
    
    for var_idx in 0..num_variables {
        let mut evaluations = Vec::with_capacity(num_constraints);
        for constraint_idx in 0..num_constraints {
            evaluations.push(matrix[constraint_idx][var_idx]);
        }
        
        polynomials.push(interpolate_polynomial(&evaluations, &domain));
    }
    
    polynomials
}
```

### 2. Vanishing Polynomial

```rust
// Generate the vanishing polynomial t(X)
fn generate_vanishing_polynomial<F: Field>(
    domain: &GeneralEvaluationDomain<F>,
) -> DensePolynomial<F> {
    let mut coeffs = vec![F::zero(); domain.size() + 1];
    coeffs[0] = -F::one();
    coeffs[domain.size()] = F::one();
    DensePolynomial::from_coefficients_vec(coeffs)
}
```

## QAP Operations

### 1. Polynomial Evaluation

```rust
// Evaluate QAP at a point
fn evaluate_qap<F: Field>(
    qap: &QAP<F>,
    assignment: &[F],
    point: F,
) -> (F, F, F) {
    let evaluate_set = |polys: &[DensePolynomial<F>]| {
        polys.iter()
            .zip(assignment)
            .map(|(poly, &coeff)| poly.evaluate(&point) * coeff)
            .sum::<F>()
    };
    
    let u_eval = evaluate_set(&qap.u_polynomials);
    let v_eval = evaluate_set(&qap.v_polynomials);
    let w_eval = evaluate_set(&qap.w_polynomials);
    
    (u_eval, v_eval, w_eval)
}
```

### 2. QAP Satisfaction Check

```rust
// Check if an assignment satisfies the QAP
fn check_qap_satisfaction<F: Field>(
    qap: &QAP<F>,
    assignment: &[F],
    h_polynomial: &DensePolynomial<F>,
) -> bool {
    let domain = GeneralEvaluationDomain::<F>::new(qap.u_polynomials[0].degree() + 1)
        .expect("Failed to create domain");
    
    for point in domain.elements() {
        let (u, v, w) = evaluate_qap(qap, assignment, point);
        let h = h_polynomial.evaluate(&point);
        let t = qap.t_polynomial.evaluate(&point);
        
        if (u * v - w) != h * t {
            return false;
        }
    }
    
    true
}
```

## Optimization Techniques

### 1. FFT-Based Operations

```rust
// Efficient polynomial multiplication using FFT
fn multiply_polynomials<F: Field>(
    poly1: &DensePolynomial<F>,
    poly2: &DensePolynomial<F>,
) -> DensePolynomial<F> {
    let degree = poly1.degree() + poly2.degree() + 1;
    let domain = GeneralEvaluationDomain::<F>::new(degree)
        .expect("Failed to create domain");
    
    let evals1 = domain.fft(&poly1.coeffs);
    let evals2 = domain.fft(&poly2.coeffs);
    
    let mut product_evals = vec![F::zero(); domain.size()];
    for i in 0..domain.size() {
        product_evals[i] = evals1[i] * evals2[i];
    }
    
    DensePolynomial::from_coefficients_vec(domain.ifft(&product_evals))
}
```

### 2. Parallel Processing

```rust
use ark_std::{cfg_iter, cfg_iter_mut};

// Parallel polynomial operations
fn parallel_evaluate_polynomials<F: Field>(
    polynomials: &[DensePolynomial<F>],
    point: F,
) -> Vec<F> {
    cfg_iter!(polynomials)
        .map(|poly| poly.evaluate(&point))
        .collect()
}
```

## QAP in Groth16

### 1. Setup Phase

```rust
// Generate QAP-based structured reference string
fn generate_structured_reference_string<E: Pairing>(
    qap: &QAP<E::ScalarField>,
    tau: E::ScalarField, // Toxic waste
) -> (ProvingKey<E>, VerifyingKey<E>) {
    let powers_of_tau: Vec<_> = (0..qap.u_polynomials[0].degree())
        .map(|i| tau.pow(&[i as u64]))
        .collect();
    
    // Generate proving key elements
    let pk_u: Vec<_> = qap.u_polynomials.iter()
        .map(|poly| evaluate_in_exponent(poly, &powers_of_tau))
        .collect();
    
    let pk_v: Vec<_> = qap.v_polynomials.iter()
        .map(|poly| evaluate_in_exponent(poly, &powers_of_tau))
        .collect();
    
    // Similar for other elements...
    
    // Generate verification key elements
    // ...
    
    (proving_key, verification_key)
}
```

### 2. Proof Generation

```rust
// Generate proof using QAP
fn generate_proof<E: Pairing>(
    pk: &ProvingKey<E>,
    qap: &QAP<E::ScalarField>,
    assignment: &[E::ScalarField],
) -> Proof<E> {
    // Compute A = Σ a_i * u_i(τ)
    let a = compute_linear_combination(&pk.u_elements, assignment);
    
    // Compute B = Σ a_i * v_i(τ)
    let b = compute_linear_combination(&pk.v_elements, assignment);
    
    // Compute C = Σ a_i * w_i(τ)
    let c = compute_linear_combination(&pk.w_elements, assignment);
    
    Proof { a, b, c }
}
```

## Testing and Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr};
    
    #[test]
    fn test_qap_satisfaction() {
        // Create a simple QAP for x * y = z
        let domain = GeneralEvaluationDomain::<Fr>::new(4)
            .expect("Failed to create domain");
        
        let qap = create_multiplication_qap(&domain);
        
        // Test valid assignment
        let assignment = vec![
            Fr::from(1), // One
            Fr::from(3), // x
            Fr::from(2), // y
            Fr::from(6), // z = x * y
        ];
        
        let h_poly = compute_h_polynomial(&qap, &assignment);
        assert!(check_qap_satisfaction(&qap, &assignment, &h_poly));
        
        // Test invalid assignment
        let invalid_assignment = vec![
            Fr::from(1),
            Fr::from(3),
            Fr::from(2),
            Fr::from(7), // Wrong result
        ];
        
        let h_poly = compute_h_polynomial(&qap, &invalid_assignment);
        assert!(!check_qap_satisfaction(&qap, &invalid_assignment, &h_poly));
    }
}
```

## Next Steps

After understanding QAP, we'll explore:
1. The R1CS to QAP reduction in detail
2. The complete Groth16 protocol
3. Performance optimizations and implementation details

## References

1. [QAP Original Paper](https://eprint.iacr.org/2012/215.pdf)
2. [Why and How zk-SNARK Works](https://arxiv.org/abs/1906.07221)
3. [Arkworks QAP Implementation](https://github.com/arkworks-rs/groth16/blob/master/src/r1cs_to_qap.rs)