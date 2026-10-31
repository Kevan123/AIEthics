# Certified decisions from dependent AI judges: simulation code

This repository contains the code and results for the simulation study and the numerical checks in the manuscript *When Does AI Consensus Warrant Action? Evidential Proportionality and Certified Decisions under Shared Error*.

The study asks when the agreement of several AI judges is strong enough evidence to accept or reject a claim, given that the judges may share errors. A decision is certified only if its error probability, given the observed pattern of votes, is within a declared tolerance with high confidence. All metrics are computed exactly from the known true distribution, so the results contain no test-set sampling error.

## Files

| File | Purpose |
| --- | --- |
| `council_sim.py` | Runs the full simulation study (Section 6 of the paper) and writes `results.json`. |
| `check_prop.py` | Checks Propositions 1 and 4 numerically against linear programmes. |
| `summarise.py` | Prints the tables reported in the paper from `results.json`. |
| `results.json` | Results of the reported run (400 replications per cell, 8,400 in total). |

## Requirements

- Python 3.10 or later (the reported run used Python 3.13)
- NumPy (reported run: 2.5.3)
- SciPy 1.9 or later, for the HiGHS linear-programming solver (reported run: 1.18.1)

```bash
pip install numpy scipy
```

## Usage

Run all commands from the repository folder.

**1. Check the propositions** (a few seconds):

```bash
python check_prop.py
```

For five settings of the per-judge error rate `e` and the pairwise co-error rate `r`, this prints the sharp bounds on the probability that all three judges err. The bounds are computed two ways: by a linear programme over all joint distributions, and from the closed form in Proposition 4. The two must match. The script then prints the two distributions of Proposition 1, which have identical individual accuracy (0.6) and pairwise error correlation (1/6) but majority accuracy of 0.8 and 0.4.

**2. Run the simulation:**

```bash
python council_sim.py 400
```

The argument is the number of replications per regime and audit size (default 200). With 400, the run solves several hundred thousand small linear programmes and took about 10 minutes on two CPU cores. It writes `results.json`, replacing the copy in the repository.

**3. Print the tables:**

```bash
python summarise.py
```

## What the simulation does

**Council and population.** Three judges vote on binary claims. 40 per cent of claims are true. There are three dependence regimes with similar individual accuracy (about 0.85) but different joint structure:

| Regime | Construction | Pairwise error correlation | P(all three err) |
| --- | --- | --- | --- |
| `independent` | Each judge correct with probability 0.85, independently | 0 | 0.0034 |
| `shared_signal` | Each judge copies a common signal (accuracy 0.8) with probability 0.5, otherwise uses a private signal (accuracy 0.9) | 0.31 | 0.033 |
| `higher_order` | Exchangeable; P(3, 2, 1, 0 correct) = 0.75, 0.12, 0.03, 0.10 | 0.63 | 0.10 |

**Samples.** Audit sizes are 250, 500, 1,000, 2,000, 5,000, 10,000 and 20,000 labelled claims. Each replication also draws 20,000 unlabelled claims.

**Decision parameters.** These are set at the top of `council_sim.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `TAU_A`, `TAU_R` | 0.05 | Tolerances on false acceptance and false rejection |
| `ALPHA` | 0.05 | Confidence level of the simultaneous confidence set |
| `C_A`, `C_R`, `C_H` | 10, 5, 1 | Costs of false acceptance, false rejection and referral |
| `PREV` | 0.4 | Prevalence of true claims |
| `N_UNLAB` | 20,000 | Size of the unlabelled sample |

**Methods compared:**

| Code | Method |
| --- | --- |
| `MV` | Majority vote, no referral |
| `PLUG` | Empirical probability per vote pattern from the audit, used as a point estimate |
| `CI` | Conditional-independence (naive Bayes) model fitted to the audit |
| `JS1` | Proposed rule: Clopper and Pearson intervals on the 16 cell probabilities, Bonferroni allocation, sharp bounds by linear programming |
| `JS2` | JS1 plus pooled audit events (prevalence, per-judge and pairwise rates) and unlabelled vote-pattern frequencies, 37 events in total |

For each vote pattern, the proposed rule first checks that the pattern has positive probability over the whole confidence set. It then computes lower and upper bounds on the probability that the claim is true, using the Charnes and Cooper transformation, and certifies acceptance or rejection only when the worst case meets the tolerance. Otherwise it refers the claim.

## Output fields

`results.json` has two parts. The `truth` part gives each regime's true probabilities by vote pattern. The `summary` part has one row per regime, audit size and method, with each metric averaged over replications.

| Field | Meaning in the paper |
| --- | --- |
| `viol`, `viol_se` | Breach rate and its Monte Carlo standard error: share of replications in which any certified decision exceeds its tolerance (Table 1) |
| `pA`, `pR`, `pH` | Share of claims accepted, rejected and referred |
| `fa_rate` | Probability that an accepted claim is false |
| `fr_rate` | Probability that a rejected claim is true |
| `loss` | Expected loss per claim, in units of the referral cost (Table 2) |
| `accept111` | Share of replications that accept on unanimous support |
| `width111`, `mean_width` | Width of the bound on P(true) for unanimous support, and averaged over patterns (JS1 and JS2 only) |
| `empty` | Share of replications in which the confidence set was empty (JS1 and JS2 only) |

When a method never accepts or never rejects, `fa_rate` or `fr_rate` is undefined and shown as `NaN`. NumPy may print a "Mean of empty slice" warning in that case; it is expected.

## Reproducibility

Each replication uses its index in the job list as its random seed, so a run with the same number of replications reproduces `results.json` exactly. Changing the number of replications, or the regimes or audit sizes, changes the seed assignment.

## Limitations

The simulation uses a three-judge council, stylised dependence regimes, equal accuracy on true and false claims, a single evidence condition, and no shift between audit and deployment. It shows how the methods behave under known dependence structures. It is not evidence about the error structure of any real AI council.

## Citation

Citation details will be added after peer review.

## Licence
Not applicable
