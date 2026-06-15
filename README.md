# Integrated Autonomous Software Prototype

This repository contains the fixed autonomous software prototype and the locked environmental-gradient validation runner used in the study.

## Files

* `darca_true_3d_integrated_task_battery_v11.py`
  Main autonomous model implementation.

* `run_v11_locked_gradient_staged.py`
  Locked one-factor environmental-gradient and representative-environment validation runner.

## Purpose

The code evaluates whether a fixed autonomous software prototype shows graded environment-conditioned behavioral organization in TRUE 3D closed-loop environments.

The model includes:

* autonomous regulatory core
* qualitative valuation module
* physical-risk/action-consequence learning module
* anonymous social-signalling module
* 3D motor policy

No language-model controller or external prompting is used during simulation.

## Run

```bash
python3 -u run_v11_locked_gradient_staged.py \
  --outdir ./results \
  2>&1 | tee run.log
```

## Outputs

The runner writes CSV summaries and reports to the output directory, including:

* episode-level summaries
* gradient-response summaries
* module-to-action coupling summaries
* representative-environment robustness summaries
* post hoc validation reports

## Reproducibility

All model variants are evaluated under the same environments, task contexts, seeds, steps, and metrics.
The autonomous model is fixed across conditions; only environmental factors are varied.

## Citation

If using this code, cite the accompanying manuscript.
