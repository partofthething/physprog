# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-02

First stable release. It fixes several bugs in how class functions were
built and evaluated, so **optimization results differ from 0.1.0** for the
same inputs. It also completes the set of preference classes from Messac's
paper: all four soft and all four hard classes.

### Added

- `ValueIsBetter` (3-S) soft preference: smooth minimum at an optional
  `target`, with a `pull` setting (0–1, default 0.5) for how strongly the
  target attracts the design.
- `RangeIsBetter` (4-S) soft preference: flat across a range, rising
  smoothly outside it.
- `MustEqual` (3-H) hard preference, applied as an equality constraint.
- `MustBeInRange` (4-H) hard preference, applied as two inequality limits.
- YAML input for two-sided preferences (`lower_bounds`, `upper_bounds`,
  `target`, `pull`) and hard ranges (`bounds: [lower, upper]`).
- `objective.ModelEvaluator`, which runs `model.evaluate(x)` once per design
  and caches the results for the objective and all constraints.
  `build_objective` and `get_constraints` accept a shared `evaluator=`.
- Plots: `plots.plot_class_function` (one class function with shaded,
  labeled preference regions) and `plots.plot_class_functions` (a grid of
  all soft preferences with initial and final designs marked). Both take
  `yscale="log"` or `"linear"`, and labels use the `name` and `units` from
  the input file.
- Input validation: unknown class names and misordered bounds now raise
  `ValueError`.
- `optimize()` returns the `scipy.optimize.OptimizeResult`, warns if the
  optimizer did not converge, and reports how many times it ran the model.
- README figures, `scripts/make_readme_plots.py` to regenerate them,
  documentation for all preference classes and for using the objective
  with other optimizers, and this changelog.
- GitHub Actions CI running ruff and pytest on Python 3.12–3.14.
- Tests for continuity, monotonicity, penalties, constraints, two-sided
  preferences, model evaluation and plotting.

### Changed

- **Breaking:** requires Python 3.12 or newer.
- **Breaking:** `optimize.get_constraints` returns at most one vector-valued
  `"ineq"` constraint and one `"eq"` constraint, instead of one constraint
  per limit.
- **Breaking:** options of the plot functions are keyword-only.
  `plot_optimization_results` takes dicts keyed by dependent-variable name
  (sequences still work), plus `fname` and `show`, and has a new look.
- The beam sample problem now finds a design with frequency in the
  desirable region and no metric worse than tolerable.
- matplotlib is only imported when plotting.
- Linting uses a broader ruff rule set (bugbear, simplify, pylint rules,
  docstrings) instead of pylint.
- Dependencies relocked (scipy 1.18, numpy 2.5, matplotlib 3.11) so the
  package installs on Python 3.14.
- Hard-preference docstrings follow Messac's class numbering.

### Fixed

- The highly desirable region evaluated to about zero because of a misplaced
  parenthesis, so the class function jumped at the first bound. It is now an
  exponential matching the value and slope there, with the first slope set
  as in Messac's algorithm.
- The penalty beyond the last bound jumped at the bound, and for
  `LargerBetter` it got *smaller* as the value got worse. It now continues
  the last spline smoothly and grows quadratically.
- All hard constraints used the last constraint's cutoff because of a
  late-binding closure in a loop.
- Hard constraints read model outputs without calling `model.evaluate(x)`,
  so models that compute outputs in `evaluate` gave constraint values for
  the wrong design and the optimizer failed. Each constraint also re-ran the
  model separately; the beam sample now runs the model 217 times instead of
  965.
- `optimize()` left the model at the last design the optimizer tried
  instead of the optimum.
- `acceptability()` crashed with `IndexError` for unacceptable values, which
  broke `plot=True`.
- Unimplemented class function methods returned `NotImplementedError`
  instead of raising it.
- Sample input listed cost in kg instead of dollars.

### Removed

- Travis CI configuration and `.pylintrc`.

## [0.1.0] - 2018-02-27

First version of physprog, an implementation of Messac's Physical
Programming method (1996) for building aggregate objective functions.

### Added

- Preferences loaded from a YAML file, with soft `SmallerBetter` (1-S) and
  `LargerBetter` (2-S) class functions and hard `MustBeBelow` and
  `MustBeAbove` constraints.
- Class function splines built automatically on load, iterating β until
  every soft preference satisfies the one-vs-others rule.
- `objective.build_objective` for using the aggregate objective with any
  optimizer.
- `optimize.optimize`, which runs `scipy.optimize.minimize` with hard
  preferences as constraints and can plot how each metric changed.
- Beam design sample problem from Messac (1996) as an end-to-end test.
- Apache-2.0 license (the very first commit was MIT).
- Support for Python 2.7, 3.5 and 3.6, tested with tox on Travis CI.

### Changed (2026-03-07, without a version bump)

- Packaging moved from `setup.py` and requirements files to
  `pyproject.toml`, with uv for installation and development. Requires
  Python 3.10 or newer, which drops Python 2.
- Code formatted with ruff.
- References in the README updated.

### Security (2026-03-07, without a version bump)

- Input files are read with `yaml.safe_load` instead of `yaml.load`.

[1.0.0]: https://github.com/partofthething/physprog/compare/0409ed1...v1.0.0
[0.1.0]: https://github.com/partofthething/physprog/tree/0409ed1
