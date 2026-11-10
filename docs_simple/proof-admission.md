# Proof admission, explained simply

[Full reference](../docs/proof-admission.md)

Each benchmark has a score such as "372 of 373 proved". A score is only a number, and a number can go up for a bad reason: a new rule might "prove" a pair of queries that are not really the same.

So each covered benchmark also has a list of exactly which proofs its score counts. A new proof gets onto the list only after it has been attacked: thousands of small test databases are tried to find one where the two queries give different answers, and small deliberate mistakes are made in the queries to see whether the prover wrongly accepts those too. If anything turns up, the proof is rejected and reported as a bug.

A quick test runs on every pull request and fails if a score and its list disagree, or if a list has no score to be compared with. That means nobody can raise a score without putting the new proofs through the attack first.

## Example

A rule change makes one more QED pair prove, so its score would go from 372 to 373. The quick test fails until someone runs the admission tool, which tests the new proof and, if it survives, adds one line to the list.

## Limits

- Surviving the attack is strong evidence, not certainty. Some wrong proofs found by hand had passed every automatic check.
- The quick test compares two recorded numbers. If a change makes the prover lose a proof, the test only notices once someone measures the benchmark again.
- The proofs that were already counted when the lists were started were not attacked by this tool.
- Eleven benchmarks have a list; the others are not covered.

Which benchmarks are covered, the commands and the exact checks are in the [full reference](../docs/proof-admission.md).
