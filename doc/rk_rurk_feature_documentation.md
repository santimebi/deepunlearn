# Functional and Technical Documentation: Residual Knowledge, RURK, and CFKG+RURK

## 0. Document purpose

This document defines the functional and technical requirements for implementing three new features in the `deepunlearn` benchmark codebase:

1. **Residual Knowledge (RK)** as a final evaluation metric.
2. **RURK** as a new machine unlearning method.
3. **CFKG+RURK** as a sequential combined unlearning method.

The target reader is a senior Python / machine learning engineer who will implement the features in the existing `deepunlearn` repository.

The immediate experimental target is:

```text
Dataset: CIFAR-10
Model: ResNet18
Evaluation setting: post-hoc machine unlearning
Primary benchmark objective: objective10, unchanged
New metric: Residual Knowledge, computed after unlearning
```

The implementation must preserve the current `objective10` behavior and should not overwrite existing experiment artifacts.

---

# Part I — Functional Documentation

## 1. Background and motivation

The current benchmark evaluates machine unlearning methods mainly through utility, privacy, and efficiency metrics such as retain accuracy, forget accuracy, validation/test accuracy, membership inference attack behavior, runtime efficiency, and retention deviation against the retrained reference model.

However, these metrics evaluate the model mostly on original samples. A model may appear to have forgotten the forget set on the original samples while still recognizing slightly perturbed versions of those same forget samples. This phenomenon is called **Residual Knowledge**.

Residual Knowledge is relevant because a successful unlearning method should not only forget the exact forget samples, but should also avoid retaining useful predictive information in the local neighborhood around those samples.

In this project, Residual Knowledge will be added as a **final evaluation metric**. It will not be used as an Optuna objective in the first implementation phase.

## 2. Locked design decisions

The following decisions are fixed for this implementation phase.

```text
Residual Knowledge:
    Role: final evaluation metric only
    Optuna objective: not in this phase
    Reference model: retrained/naive model, always available in this benchmark
    Perturbation type: Gaussian only
    Default tau: 0.03
    Default c: 100 perturbations per forget sample
    Default K: None, meaning evaluate the full forget set
    Fast/debug K: 100 optional
    Per-sample values: must be saved
    Final tables: must include RK@0.03
    Curves over tau: optional script, not required in the main pipeline

RURK:
    Unlearner name: rurk
    Starts from: original trained model
    Model update: full model fine-tuning
    Frozen layers: no
    Perturbation type: Gaussian only
    Loss: retain_loss - lambda_f * forget_loss - lambda_a * adv_forget_loss
    Hyperparameters: all relevant training and RURK parameters except freeze_layers

CFKG+RURK:
    Unlearner name: cfkg_rurk
    Combination type: sequential
    Phase 1: run catastrophic_forgetting_gamma_k
    Phase 2: run RURK initialized from the CFKG output

Experiment scope:
    First target: ResNet18 + CIFAR-10 only
    Reduced experimental list:
        finetune
        cfk
        euk
        catastrophic_forgetting_gamma_k
        rurk
        cfkg_rurk
        neggradplus

Reporting:
    Generate a new ranking where Residual Knowledge is treated as its own metric.
    Do not merge RK into objective10.
    Do not overwrite existing objective10 artifacts.
```

## 3. Feature 1 — Residual Knowledge evaluation

### 3.1 Functional goal

Implement a new evaluation metric that estimates how much more likely an unlearned model is to correctly classify Gaussian-perturbed versions of forget samples compared with the retrained reference model.

For each forget sample `(x, y)`, generate `c` Gaussian perturbations around `x`, evaluate both the unlearned model and the retrained model, and compute the ratio:

```text
RK_tau(x, y) = P[unlearned_model(x') = y] / P[retrained_model(x') = y]
```

where `x'` is sampled from a Gaussian perturbation around `x`, clipped/renormalized so that it remains a valid model input.

The aggregate metric is the average across forget samples:

```text
RK_tau(Sf) = mean_{(x,y) in Sf} RK_tau(x,y)
```

Interpretation:

```text
RK_tau ≈ 1:
    The unlearned model behaves similarly to the retrained model around forget samples.

RK_tau > 1:
    The unlearned model recognizes perturbed forget samples more often than the retrained model.
    This indicates residual knowledge.

RK_tau < 1:
    The unlearned model recognizes perturbed forget samples less often than the retrained model.
    This may indicate stronger forgetting or over-forgetting.
```

The main value to report is:

```text
RK@0.03
```

A secondary useful value is:

```text
ExcessRK@0.03 = max(0, RK@0.03 - 1)
```

`ExcessRK` isolates the problematic part of RK, because only values above 1 indicate that the unlearned model is more capable than the retrained model of recognizing perturbed forget samples.

### 3.2 Functional requirements

The implementation must:

1. Compute RK for any saved unlearned checkpoint and its corresponding retrained/naive checkpoint.
2. Use the forget set only.
3. Use Gaussian perturbations only.
4. Support configurable `tau`, `c`, `K`, `seed`, `batch_size`, and `device`.
5. Use `tau=0.03`, `c=100`, `K=None` by default for final evaluation.
6. Use deterministic sampling when `seed` is fixed.
7. Save both aggregate and per-sample RK values.
8. Include RK in the final result tables.
9. Support optional fast/debug evaluation with `K=100`.
10. Avoid modifying `objective10`.

### 3.3 Non-goals

The first implementation must not include:

```text
FGSM perturbations
PGD perturbations
RK as an Optuna objective
Adversarial disagreement as a separate metric
RK for generative models
RK for non-classification tasks
```

These can be added later.

### 3.4 Expected outputs

For each evaluated method, model, dataset, and seed, the RK evaluator should write an aggregate output file such as:

```text
artifacts/cifar10/residual_knowledge/<method>/10_resnet18_<seed>_rk.json
```

Recommended aggregate JSON schema:

```json
{
  "dataset": "cifar10",
  "model": "resnet18",
  "seed": 0,
  "unlearner": "cfk",
  "reference": "naive",
  "tau": 0.03,
  "c": 100,
  "K": null,
  "perturbation": "gaussian",
  "rk_mean": 1.1234,
  "rk_excess": 0.1234,
  "num_forget_samples": 4250,
  "num_evaluated_samples": 4250,
  "num_perturbations_per_sample": 100,
  "denominator_epsilon": 1e-12
}
```

For per-sample values, write a CSV or parquet file such as:

```text
artifacts/cifar10/residual_knowledge/<method>/10_resnet18_<seed>_rk_per_sample.csv
```

Recommended per-sample columns:

```text
sample_index
label
tau
c
unlearned_correct_count
reference_correct_count
unlearned_correct_rate
reference_correct_rate
rk_sample
rk_sample_excess
```

### 3.5 Reporting requirements

The final tables should include:

```text
RK@0.03
ExcessRK@0.03
```

The new ranking should treat RK as an independent metric. It should not be merged into `RetDev` or `Indisc`.

Recommended ranking direction:

```text
RK distance from 1: minimize |RK@0.03 - 1|
Excess RK: minimize max(0, RK@0.03 - 1)
```

For the primary privacy-risk ranking, use `ExcessRK@0.03`. For a model-similarity ranking, use `|RK@0.03 - 1|`.

---

## 4. Feature 2 — RURK unlearner

### 4.1 Functional goal

Implement `rurk` as a new unlearning method that fine-tunes the original model using a loss that preserves retain-set performance while discouraging the model from recognizing both original forget samples and Gaussian-perturbed forget samples.

The RURK training loss is:

```text
L = retain_loss - lambda_f * forget_loss - lambda_a * adv_forget_loss
```

where:

```text
retain_loss:
    Cross-entropy on retain batch.

forget_loss:
    Cross-entropy on forget batch.
    It is subtracted, so optimization increases the loss on forget samples.

adv_forget_loss:
    Cross-entropy on Gaussian-perturbed forget batch.
    It is subtracted, so optimization increases the loss on perturbed forget samples.
```

The method must update the full model. It must not freeze layers in this implementation.

### 4.2 Functional requirements

The implementation must:

1. Register a new unlearner named `rurk`.
2. Start from the original trained model.
3. Use both retain and forget dataloaders.
4. Generate Gaussian perturbations of forget batches during training.
5. Clip and renormalize perturbations consistently with the dataset preprocessing pipeline.
6. Optimize the full model.
7. Save the final checkpoint and metadata using the same conventions as existing unlearners.
8. Expose hyperparameters through the current Hydra/config/Optuna system.
9. Run for ResNet18 + CIFAR-10.
10. Produce logs for retain loss, forget loss, adversarial forget loss, and total loss.

### 4.3 RURK hyperparameters

Required hyperparameters:

```text
epochs
learning_rate
weight_decay
momentum
lambda_f
lambda_a
tau
num_adv_samples
batch_size if supported by existing unlearner configs
```

Default values for first implementation:

```text
epochs: 2
learning_rate: 0.01
momentum: 0.9
weight_decay: 0.0005
lambda_f: 0.03
lambda_a: 0.00045
tau: 0.03
num_adv_samples: 1
```

These defaults are intentionally conservative and aligned with the intended first target: CIFAR-10 + ResNet18.

### 4.4 Non-goals

The first RURK implementation must not include:

```text
FGSM or PGD attack generation
torchattacks dependency
layer freezing
joint CFKG/RURK loss
RK as an Optuna objective
```

---

## 5. Feature 3 — CFKG+RURK sequential unlearner

### 5.1 Functional goal

Implement `cfkg_rurk` as a combined unlearning method that first applies the existing `catastrophic_forgetting_gamma_k` method and then applies RURK to the resulting model.

This method is intended to test whether CFKG can be improved by a robust local forgetting phase that reduces Residual Knowledge around the forget samples.

### 5.2 Functional behavior

The method must execute:

```text
original_model
    -> catastrophic_forgetting_gamma_k
        -> intermediate_cfkg_model
            -> rurk
                -> final_cfkg_rurk_model
```

The final checkpoint must be saved under the `cfkg_rurk` method name, not under `catastrophic_forgetting_gamma_k` or `rurk`.

### 5.3 Functional requirements

The implementation must:

1. Register a new unlearner named `cfkg_rurk`.
2. Reuse the existing CFKG implementation for phase 1.
3. Reuse the new RURK implementation for phase 2.
4. Clearly separate logs for both phases.
5. Save metadata containing both CFKG and RURK hyperparameters.
6. Save only the final model as the official `cfkg_rurk` checkpoint.
7. Optionally save the intermediate CFKG model in a temporary or debug path, but this must not be required for final evaluation.

### 5.4 Non-goals

The first implementation must not implement a joint loss such as:

```text
L = CE + CFKG_penalty - lambda_f * forget_loss - lambda_a * adv_forget_loss
```

That joint formulation is a future research extension.

---

## 6. Reduced experimental protocol

The first experimental run should focus only on:

```text
Dataset: CIFAR-10
Architecture: ResNet18
```

The reduced list of methods is:

```text
finetune
cfk
euk
catastrophic_forgetting_gamma_k
rurk
cfkg_rurk
neggradplus
```

For all methods with existing valid checkpoints, RK should be computed without rerunning Optuna.

For new methods (`rurk`, `cfkg_rurk`), run the same unlearning/evaluation pipeline as existing methods and then compute RK.

---

# Part II — Technical Documentation

## 7. Current repository context

The current repository state is approximately:

```text
Repository: ~/deepunlearn
Current branch: main
Python environment: munl
Python version: 3.10.19
Primary artifact root: /data/santiago.medina/deepunlearn/artifacts
```

Important current files and modules:

```text
pipeline/__init__.py
pipeline/optuna_search_hp.py
pipeline/step_7_generate_optuna.py
pipeline/step_8_generate_all_best_hp.py
munl/configurations.py
munl/settings.py
munl/evaluation/
munl/hpsearch/objectives.py
munl/unlearning/common.py
munl/unlearning/catastrophic_forgetting_k.py
munl/unlearning/catastrophic_forgetting_gamma_k.py
munl/unlearning/exact_unlearning_k.py
munl/unlearning/finetune.py
munl/unlearning/neggradplus.py
munl/unlearning/scrub_utils.py
```

Current known unlearner registration is handled through `unlearner_store` in:

```text
munl/configurations.py
```

The existing custom method `catastrophic_forgetting_gamma_k` is already registered and imported through:

```text
munl/configurations.py
munl/unlearning/__init__.py
munl/unlearning/catastrophic_forgetting_gamma_k.py
```

Current objective configuration:

```text
pipeline/__init__.py: OBJECTIVES = ["objective10"]
pipeline/optuna_search_hp.py: objective10 uses unlearner_optuna
munl/hpsearch/objectives.py: current objective terms are retain, forget, val, and discernibility
```

## 8. Important implementation risk: artifact path duplication

The current artifact tree contains paths such as:

```text
/data/santiago.medina/deepunlearn/artifacts/cifar10/artifacts/cifar10/unlearn/...
```

This indicates a possible path resolution issue where `artifacts/cifar10` may be duplicated inside itself.

The RK evaluator must not assume paths by string concatenation only. It should reuse the same path utilities already used by the repository wherever possible. If path utilities are inconsistent, the evaluator should expose explicit CLI arguments for:

```text
--unlearned-checkpoint
--reference-checkpoint
--output-dir
```

This makes RK evaluation robust even if existing artifact paths are irregular.

## 9. Recommended module structure

Add the following files:

```text
munl/evaluation/residual_knowledge.py
munl/unlearning/rurk.py
munl/unlearning/cfkg_rurk.py
scripts/evaluate_residual_knowledge.py
scripts/generate_residual_knowledge_tables.py
tests/test_residual_knowledge.py
tests/test_rurk.py
tests/test_cfkg_rurk_config.py
```

Optional later:

```text
scripts/plot_residual_knowledge_curves.py
```

## 10. Residual Knowledge implementation details

### 10.1 Public API

Recommended function signature:

```python
def compute_residual_knowledge(
    unlearned_model,
    reference_model,
    forget_loader,
    *,
    tau: float = 0.03,
    c: int = 100,
    max_samples: int | None = None,
    seed: int = 123,
    device: str = "cuda",
    denominator_epsilon: float = 1e-12,
    return_per_sample: bool = True,
) -> dict:
    ...
```

Expected returned dictionary:

```python
{
    "rk_mean": float,
    "rk_excess": float,
    "tau": float,
    "c": int,
    "max_samples": int | None,
    "num_evaluated_samples": int,
    "per_sample": pandas.DataFrame | None,
}
```

### 10.2 Perturbation generation

Only Gaussian perturbations are required.

Pseudo-implementation:

```python
def gaussian_perturb(x, tau, generator):
    noise = torch.randn_like(x, generator=generator) * tau
    x_perturbed = x + noise
    x_perturbed = clip_or_project_to_valid_input_domain(x_perturbed)
    x_perturbed = renormalize_if_needed(x_perturbed)
    return x_perturbed
```

The exact clipping/renormalization must respect the repository’s dataset transform convention.

If tensors are already normalized when they arrive from the loader, implementation has two possible choices:

```text
Option A:
    Add noise in normalized space and clamp using normalized bounds.

Option B:
    Denormalize to pixel space, add noise, clamp to [0,1], then normalize again.
```

Recommended choice:

```text
Use Option B if dataset mean/std are easily available.
Use Option A only if existing repository utilities already operate in normalized tensor space.
```

For CIFAR-10, the implementation must be explicit about which convention is used and must keep it consistent between RK evaluation and RURK training.

### 10.3 Denominator stability

If the retrained/reference model never correctly classifies any perturbation for a sample, then the denominator is zero.

Use:

```text
rk_sample = unlearned_correct_rate / max(reference_correct_rate, denominator_epsilon)
```

However, this can produce very large RK values. Therefore, the output should also include raw counts and rates, so extreme values can be inspected.

Alternative optional reporting:

```text
rk_sample_is_unstable = reference_correct_count == 0
```

### 10.4 Efficient batching

For each forget batch of size `B`, instead of looping sample by sample, expand the batch along the perturbation dimension:

```text
Input batch: B x C x H x W
Repeated batch: (B*c) x C x H x W
Predictions: reshape back to B x c
```

This avoids `B*c` individual model calls.

Pseudo-flow:

```python
for x, y in forget_loader:
    x, y = x.to(device), y.to(device)
    x_rep = repeat each x c times
    y_rep = repeat each y c times
    x_pert = gaussian_perturb(x_rep, tau, generator)

    with torch.no_grad():
        pred_unlearned = unlearned_model(x_pert).argmax(dim=1)
        pred_reference = reference_model(x_pert).argmax(dim=1)

    correct_unlearned = (pred_unlearned == y_rep).reshape(B, c).sum(dim=1)
    correct_reference = (pred_reference == y_rep).reshape(B, c).sum(dim=1)
```

### 10.5 CLI script

Add:

```text
scripts/evaluate_residual_knowledge.py
```

Recommended CLI arguments:

```text
--dataset cifar10
--model resnet18
--seed 0
--unlearner cfk
--tau 0.03
--c 100
--max-samples none
--perturbation gaussian
--output-dir /data/santiago.medina/deepunlearn/artifacts/cifar10/residual_knowledge
--device cuda
```

For robustness, also support explicit checkpoint arguments:

```text
--unlearned-checkpoint /path/to/model.pth
--reference-checkpoint /path/to/naive_or_retrained.pth
```

## 11. RURK implementation details

### 11.1 New file

Add:

```text
munl/unlearning/rurk.py
```

Recommended class names:

```python
class RURKUnlearner(BaseUnlearner):
    ...

@dataclass
class DefaultRURKConfig:
    ...
```

The class should follow the same conventions as existing unlearners such as `FinetuneUnlearner`, `CatastrophicForgettingK`, and `ForgettingGammaK`.

### 11.2 Registration

Update:

```text
munl/configurations.py
munl/unlearning/__init__.py
munl/settings.py
pipeline/__init__.py or reduced experiment list generator
```

Add to `munl/configurations.py`:

```python
@unlearner_store(name="rurk")
def default_rurk_config():
    from munl.unlearning.rurk import DefaultRURKConfig
    return unlearner_config_factory(cls=RURKUnlearner, cfg=DefaultRURKConfig())
```

The exact syntax must follow the current `unlearner_config_factory` pattern in the repository.

Add readable name, color, and marker to `munl/settings.py`:

```text
"rurk": "RURK"
```

### 11.3 Training loop

Pseudo-flow:

```python
for epoch in range(cfg.epochs):
    for retain_batch in retain_loader:
        x_r, y_r = retain_batch
        x_f, y_f = next(forget_iterator_cycle)

        logits_r = model(x_r)
        retain_loss = CE(logits_r, y_r)

        logits_f = model(x_f)
        forget_loss = CE(logits_f, y_f)

        x_adv = gaussian_perturb(x_f, tau=cfg.tau)
        logits_adv = model(x_adv)
        adv_forget_loss = CE(logits_adv, y_f)

        loss = retain_loss - cfg.lambda_f * forget_loss - cfg.lambda_a * adv_forget_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_if_configured()
        optimizer.step()
        scheduler.step_if_configured()
```

The forget loader must be cycled safely when it is shorter than the retain loader.

### 11.4 Logging

Log at least:

```text
retain_loss
forget_loss
adv_forget_loss
total_loss
learning_rate
epoch
batch_idx
```

Metadata file should include:

```text
unlearner: rurk
epochs
learning_rate
momentum
weight_decay
lambda_f
lambda_a
tau
num_adv_samples
perturbation: gaussian
```

## 12. CFKG+RURK implementation details

### 12.1 New file

Add:

```text
munl/unlearning/cfkg_rurk.py
```

Recommended class names:

```python
class CFGKRURKUnlearner(BaseUnlearner):
    ...

@dataclass
class DefaultCFGKRURKConfig:
    ...
```

### 12.2 Configuration

The config should be nested or clearly prefixed:

```python
@dataclass
class DefaultCFGKRURKConfig:
    cfkg_epochs: int = ...
    cfkg_learning_rate: float = ...
    cfkg_k: int = ...
    cfkg_c: float = ...
    cfkg_gamma: float = ...

    rurk_epochs: int = 2
    rurk_learning_rate: float = 0.01
    rurk_lambda_f: float = 0.03
    rurk_lambda_a: float = 0.00045
    rurk_tau: float = 0.03
    rurk_num_adv_samples: int = 1
```

Alternatively, reuse existing config objects internally if the repository pattern makes this cleaner.

### 12.3 Execution

Pseudo-flow:

```python
model = load_original_model()

cfkg_model = run_cfkg(
    model=model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    cfg=cfkg_cfg,
)

final_model = run_rurk(
    model=cfkg_model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    cfg=rurk_cfg,
)

save(final_model, method="cfkg_rurk")
```

### 12.4 Logging

Logs must clearly separate phases:

```text
[cfkg_rurk][phase=cfkg] ...
[cfkg_rurk][phase=rurk] ...
```

Metadata must include:

```text
method: cfkg_rurk
phase_1: catastrophic_forgetting_gamma_k
phase_2: rurk
all_cfkg_hyperparameters
all_rurk_hyperparameters
```

## 13. Reporting and ranking implementation

Add or update scripts to combine existing metrics with RK outputs.

Recommended output:

```text
reports/resnet18_cifar10_rk/final_metrics_with_rk.csv
reports/resnet18_cifar10_rk/ranking_rk.csv
reports/resnet18_cifar10_rk/ranking_retdev_indisc_rk.csv
```

Final table should include at least:

```text
method
seed
retain_acc
forget_acc
val_acc
test_acc
val_mia
test_mia
RetDev
Indisc
RTE
RK@0.03
ExcessRK@0.03
```

Ranking options:

```text
Ranking A: minimize ExcessRK@0.03
Ranking B: minimize |RK@0.03 - 1|
Ranking C: existing RetDev/Indisc ranking plus RK as an additional separate column
```

Recommended primary RK ranking:

```text
minimize ExcessRK@0.03
```

because this focuses on the privacy risk of residual knowledge above the retrained model.

## 14. Testing requirements

### 14.1 Unit tests for Residual Knowledge

Add:

```text
tests/test_residual_knowledge.py
```

Minimum tests:

1. **Identical models produce RK approximately 1**

```text
Given two identical mock models,
when RK is computed,
then RK should be approximately 1.
```

2. **Unlearned model better than reference produces RK > 1**

```text
Given a mock unlearned model that always predicts the true label
and a mock reference model that predicts the true label half of the time,
then RK should be greater than 1.
```

3. **Denominator zero does not crash**

```text
Given reference_correct_count = 0,
then computation should not divide by zero
and should mark the sample as unstable or use denominator_epsilon.
```

4. **Fixed seed gives deterministic RK**

```text
Given the same model, data, tau, c and seed,
then RK output must be identical across repeated calls.
```

### 14.2 Unit tests for RURK

Add:

```text
tests/test_rurk.py
```

Minimum tests:

1. Config instantiates correctly.
2. Loss can be computed on a tiny mock model and synthetic batch.
3. One optimization step changes at least one trainable parameter.
4. No layer freezing is applied.

### 14.3 Unit tests for CFKG+RURK

Add:

```text
tests/test_cfkg_rurk_config.py
```

Minimum tests:

1. Config instantiates correctly.
2. Method name is registered as `cfkg_rurk`.
3. Metadata contains both CFKG and RURK hyperparameter groups.

### 14.4 Smoke tests

Add a small smoke test script or documented command:

```text
Run RURK on CIFAR-10 + ResNet18 for one seed with tiny debug settings:
    epochs=1
    max_batches=2 if supported
    RK max_samples=100
    c=5
```

The smoke test is considered successful if:

```text
The unlearner runs without crashing.
A checkpoint is saved.
RK aggregate JSON is saved.
RK per-sample CSV is saved.
```

## 15. Definition of Done

### 15.1 Residual Knowledge is done when

```text
- `munl/evaluation/residual_knowledge.py` exists.
- Gaussian perturbation RK is implemented.
- RK supports tau, c, K/max_samples, seed, device and denominator_epsilon.
- RK compares unlearned model against retrained/naive model.
- RK returns aggregate and per-sample results.
- RK writes JSON and CSV outputs.
- RK appears in the final result tables.
- RK ranking can be generated.
- Unit tests pass.
- A smoke test runs on CIFAR-10 + ResNet18.
```

### 15.2 RURK is done when

```text
- `munl/unlearning/rurk.py` exists.
- `rurk` is registered in the Hydra unlearner store.
- `rurk` is importable from `munl/unlearning/__init__.py`.
- `rurk` appears in names/colors/markers where required.
- RURK uses full model fine-tuning.
- RURK implements retain_loss - lambda_f * forget_loss - lambda_a * adv_forget_loss.
- RURK uses Gaussian perturbations only.
- RURK saves checkpoint and metadata consistently.
- RURK can run on CIFAR-10 + ResNet18 for at least one seed.
- Unit tests pass.
```

### 15.3 CFKG+RURK is done when

```text
- `munl/unlearning/cfkg_rurk.py` exists.
- `cfkg_rurk` is registered in the Hydra unlearner store.
- The method runs CFKG first and RURK second.
- Logs clearly identify both phases.
- Metadata includes both CFKG and RURK hyperparameters.
- The final checkpoint is saved under `cfkg_rurk`.
- The method can be included in the reduced experimental list.
- Unit tests pass.
```

### 15.4 Reporting is done when

```text
- Existing objective10 outputs are not overwritten.
- Final CSV includes RK@0.03 and ExcessRK@0.03.
- A new RK ranking is generated.
- Reduced-method results can be generated for ResNet18 + CIFAR-10.
- The process is documented with runnable commands.
```

## 16. Suggested implementation order

Recommended order for the senior developer:

```text
1. Create a feature branch.
2. Inspect and preserve current local modifications.
3. Implement residual_knowledge.py with mock-model unit tests.
4. Add evaluate_residual_knowledge.py script.
5. Validate RK on existing checkpoints for one method and one seed.
6. Implement RURK unlearner.
7. Register RURK in configuration/settings/pipeline lists.
8. Smoke test RURK on CIFAR-10 + ResNet18.
9. Implement CFKG+RURK sequential unlearner.
10. Smoke test CFKG+RURK.
11. Add reporting script for final tables with RK.
12. Generate reduced experimental results.
13. Generate final ranking.
14. Write final implementation notes and commands.
```

## 17. Suggested branch and artifact policy

Before coding:

```bash
cd ~/deepunlearn
git status --short
git checkout -b feature/residual-knowledge-rurk
```

Because the current `main` branch has local modifications, the developer should either commit, stash, or explicitly document them before implementing the new features.

Suggested artifact directory:

```text
/data/santiago.medina/deepunlearn/artifacts/cifar10/residual_knowledge/
```

Suggested report directory:

```text
reports/resnet18_cifar10_rk/
```

No existing `objective10` directory should be overwritten.

## 18. Future extensions

The following items are intentionally excluded from the first implementation, but should be easy to add later:

```text
Residual Knowledge as objective10_rk for Optuna.
FGSM perturbations.
PGD perturbations.
Tau curves as first-class reporting output.
Joint CFKG+RURK loss.
Support for all datasets and architectures.
Residual Knowledge evaluated under multiple perturbation distributions.
```

