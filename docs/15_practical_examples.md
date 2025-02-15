# Practical Examples of Groth16 Zero-Knowledge Proofs

## Overview

This document provides practical examples of using Groth16 to create zero-knowledge proofs for real-world scenarios. We'll start with a simple example and progress to more complex use cases.

## Example 1: Proving Knowledge of a Secret Number

### Scenario
You want to prove that you know a number whose square is 9 without revealing the number itself (which could be either 3 or -3).

### Circuit Implementation
```rust
use ark_ff::Field;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_bn254::Fr;

struct SquareRootCircuit {
    // Private input (the secret number)
    secret_number: Option<Fr>,
    // Public input (the square)
    square: Option<Fr>,
}

impl ConstraintSynthesizer<Fr> for SquareRootCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate the secret number as a private witness
        let a = cs.new_witness_variable(|| {
            self.secret_number.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Allocate the square as a public input
        let square = cs.new_input_variable(|| {
            self.square.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Enforce that a * a = square
        cs.enforce_constraint(
            lc!() + a,
            lc!() + a,
            lc!() + square,
        )?;
        
        Ok(())
    }
}

// Usage example
fn prove_square_root() -> Result<(), SynthesisError> {
    let circuit = SquareRootCircuit {
        secret_number: Some(Fr::from(3)),
        square: Some(Fr::from(9)),
    };
    
    // Generate parameters
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng)?;
    
    // Create proof
    let proof = Groth16::prove(&pk, circuit, &mut rng)?;
    
    // Verify proof
    let public_input = vec![Fr::from(9)];
    assert!(Groth16::verify(&vk, &public_input, &proof)?);
    
    Ok(())
}
```

## Example 2: Proving Age Range for KYC

### Scenario
You want to prove that your age is between 18 and 120 without revealing your exact age. This is useful for age verification in KYC (Know Your Customer) processes.

### Circuit Implementation
```rust
struct AgeRangeCircuit {
    // Private input (actual age)
    age: Option<Fr>,
    // No public inputs needed besides the implicit range check
}

impl ConstraintSynthesizer<Fr> for AgeRangeCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate age as private witness
        let age = cs.new_witness_variable(|| {
            self.age.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Create binary decomposition of age (0 to 127)
        let age_bits = age.to_bits_le(cs)?;
        
        // Ensure age is at least 18
        let eighteen = Fr::from(18);
        let mut age_accumulated = Fr::zero();
        for (i, bit) in age_bits.iter().enumerate() {
            age_accumulated += Fr::from(2).pow(&[i as u64]) * bit;
        }
        cs.enforce_constraint(
            lc!() + age_accumulated,
            lc!() + Fr::one(),
            lc!() + age,
        )?;
        cs.enforce_constraint(
            lc!() + age,
            lc!() + Fr::one(),
            lc!() + (age_accumulated - eighteen),
        )?;
        
        // Ensure age is at most 120
        let hundred_twenty = Fr::from(120);
        cs.enforce_constraint(
            lc!() + (hundred_twenty - age),
            lc!() + Fr::one(),
            lc!() + (hundred_twenty - age_accumulated),
        )?;
        
        Ok(())
    }
}

// Usage example
fn prove_age_range() -> Result<(), SynthesisError> {
    let circuit = AgeRangeCircuit {
        age: Some(Fr::from(25)),
    };
    
    // Generate parameters and proof as before
    // ...
}
```

## Example 3: Proving Ownership of Assets Without Revealing Balance

### Scenario
You want to prove that you own enough cryptocurrency (> 1000 USD) to participate in a private sale without revealing your exact balance. This involves proving ownership of multiple UTXOs and their sum.

### Circuit Implementation
```rust
// Represents a UTXO (Unspent Transaction Output)
struct UTXO {
    amount: Fr,
    nullifier: Fr,
    merkle_path: Vec<(Fr, bool)>, // Path in Merkle tree
}

struct AssetOwnershipCircuit {
    // Private inputs
    utxos: Vec<UTXO>,
    private_key: Fr,
    
    // Public inputs
    merkle_root: Fr,
    minimum_required: Fr,
}

impl ConstraintSynthesizer<Fr> for AssetOwnershipCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // 1. Verify ownership of each UTXO
        let mut total_amount = Fr::zero();
        
        for utxo in &self.utxos {
            // Allocate UTXO amount as witness
            let amount = cs.new_witness_variable(|| Ok(utxo.amount))?;
            
            // Verify Merkle path (proves UTXO exists in set)
            let mut current = utxo.nullifier;
            for (sibling, is_left) in &utxo.merkle_path {
                let sibling_var = cs.new_witness_variable(|| Ok(*sibling))?;
                current = if *is_left {
                    hash_points(sibling_var, current)
                } else {
                    hash_points(current, sibling_var)
                };
            }
            
            // Enforce that computed root matches public root
            let root = cs.new_input_variable(|| Ok(self.merkle_root))?;
            cs.enforce_constraint(
                lc!() + current,
                lc!() + Fr::one(),
                lc!() + root,
            )?;
            
            // Add amount to total
            total_amount += amount;
        }
        
        // 2. Verify total amount exceeds minimum
        let min_required = cs.new_input_variable(|| Ok(self.minimum_required))?;
        cs.enforce_constraint(
            lc!() + total_amount,
            lc!() + Fr::one(),
            lc!() + min_required,
        )?;
        
        Ok(())
    }
}

// Helper function to hash two field elements
fn hash_points(left: Fr, right: Fr) -> Fr {
    // Implement secure hashing (e.g., Poseidon or MiMC)
    // ...
}

// Usage example
fn prove_asset_ownership() -> Result<(), SynthesisError> {
    // Create UTXOs
    let utxos = vec![
        UTXO {
            amount: Fr::from(500),
            nullifier: Fr::random(&mut rng),
            merkle_path: compute_merkle_path(/* ... */),
        },
        UTXO {
            amount: Fr::from(700),
            nullifier: Fr::random(&mut rng),
            merkle_path: compute_merkle_path(/* ... */),
        },
    ];
    
    let circuit = AssetOwnershipCircuit {
        utxos,
        private_key: Fr::random(&mut rng),
        merkle_root: compute_merkle_root(/* ... */),
        minimum_required: Fr::from(1000),
    };
    
    // Generate parameters and proof as before
    // ...
}
```

## Real-World Applications

1. **Simple Example (Square Root)**
   - Use case: Password verification without revealing the password
   - Benefits: Server never sees actual password
   - Real usage: Zero-knowledge password proof protocols

2. **Age Range Proof**
   - Use case: KYC/AML compliance
   - Benefits: Privacy-preserving age verification
   - Real usage: Online age-restricted services, voting systems

3. **Asset Ownership**
   - Use case: Private transactions, DeFi protocols
   - Benefits: Privacy-preserving wealth verification
   - Real usage: Private DEXs, lending protocols, DAO governance

## Converting Real Problems to Circuits

To convert a real-world problem into a zero-knowledge proof:

1. **Identify the Statement**
   - What are you trying to prove?
   - What should remain private vs. public?

2. **Convert to Mathematical Constraints**
   - Express the problem as equations
   - Break complex operations into simple arithmetic

3. **Implement Circuit**
   - Convert equations to R1CS constraints
   - Ensure efficient implementation

4. **Test and Optimize**
   - Verify correctness
   - Optimize constraint count

## Common Patterns

1. **Range Proofs**
   ```rust
   // x is in range [a, b]
   let x_bits = x.to_bits_le();
   enforce(x >= a);
   enforce(x <= b);
   ```

2. **Merkle Proofs**
   ```rust
   // Verify membership in Merkle tree
   let mut current = leaf;
   for (sibling, is_left) in merkle_path {
       current = hash(if is_left { sibling, current } else { current, sibling });
   }
   enforce(current == root);
   ```

3. **Hash Preimage**
   ```rust
   // Prove knowledge of preimage
   let preimage = witness;
   let hash_result = hash(preimage);
   enforce(hash_result == public_hash);
   ```

## References

1. [Groth16 Paper](https://eprint.iacr.org/2016/260.pdf)
2. [ZK-SNARKs for Engineers](https://blog.decentriq.com/zk-snarks-primer-part-one/)
3. [Practical Examples Repository](https://github.com/arkworks-rs/r1cs-tutorial) 