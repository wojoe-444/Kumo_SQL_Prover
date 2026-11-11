# Understanding the SQL-IQ tasks

[Simple eval index](README.md) · [Full reference](../../docs/evals/sql-iq.md)

SQL-IQ includes different tasks. They should not be read as one kind of SQL equivalence score.

## Equivalence Judge

This task compares two queries. KumoSQL first tries an algebraic proof using the supplied SQLite schema and keys. It then searches constraint-respecting random and targeted databases for a difference. A bounded solver can also propose data, but SQLite itself must confirm a counterexample.

Unresolved pairs do not become unbounded proofs just because tested databases agree. Check how the benchmark's final answer is formed and which evidence supports it in the full guide.

## SQL Judge and Error Classification

These tasks choose a candidate or classify likely errors using the question, evidence, schema, and candidate SQL. They use deterministic checks, such as unknown columns or unsupported literals.

That is a heuristic judgment of the task's visible information. It is different from proving that a query always returns the intended business answer.

## Why agreement needs an audit

Published labels can differ from semantic results because of row ordering, tied sort keys, or assumptions about real data. The full guide records disputed answers and the checks used to investigate them.

Read label agreement, proof coverage, replayed counterexamples, and heuristic-task accuracy separately. The reference has the task-specific commands and results files.

A second audit (November 2026) sorted the 211 answers that disagree with the labels. For 135 of them KumoSQL holds a small database on which the two queries really return different results although the label says "equivalent". For 22 it holds a proof although the label says "different", and thousands of random databases showed no difference for any of them. That leaves 54 real guesses: pairs where no difference was found and the label says there is one. So most of the remaining gap is disagreement with the labels, not something a better prover would close.
