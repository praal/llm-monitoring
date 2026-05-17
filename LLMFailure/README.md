# LLM Failures in Temporal Reasoning

Code for the LLM-as-a-Judge temporal-reasoning failure-mode experiments in our
paper. We use synthetic event traces with four random, uncorrelated attributes
(animal, shape, color, number) rendered as natural language, and ask LLMs to
judge whether each trace satisfies a temporal-logic constraint.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."   # required by every *_evaluate.py
```

All evaluations are routed through OpenRouter (`utils.prompt_model`), so the
same code works for any provider hosted there (OpenAI, Anthropic, Google,
Meta, Mistral, Qwen, etc.).

## Reproducing the paper figures

Each experiment is a pipeline of three scripts: a **trace generator**, an
**evaluator** that queries an LLM, and a **plot** script. Default output
locations (`data/`, `results*/`, `plots/`) are gitignored.

| Paper figure | Experiment | Run |
|---|---|---|
| `Figure 3a` | Temporal elasticity, simple formula `F(A ∧ XF B)`, varying gap | `simple_trace_generator.py` → `simple_evaluate.py` → `simple_plot.py` |
| `Figure 3d` | Temporal elasticity, complex (tree-LTL, `b=2, d=4`), varying gap | `tree_trace_generator.py --b 2 --d 4` → `tree_evaluate.py` → `tree_plot.py --b 2 --d 4` |
| `Figure 3b` | Constraint scalability, simple formula, varying N constraints | `multi_simple_trace_generator.py` → `multi_simple_evaluate.py` → `multi_simple_plot.py` |
| `Figure 3e` | Constraint scalability, complex (forest of tree-LTL), varying N | `forest_trace_generator.py` → `forest_evaluate.py` → `forest_plot.py` |
| `Figure 3c` | Proposition scalability, simple formula, varying entities per step | `prop_trace_generate.py` → `prop_evaluate.py` → `prop_plot.py` |
| `Figure 3f` | Proposition scalability, complex formula, varying entities per step | `prop_tree_trace_generator.py` → `prop_tree_evaluate.py` → `prop_tree_plot.py` |
| `Figure 6` (appendix) | Specification format — 7 patterns × 3 levels (informal NL, precise NL, precise NL + LTL) | `spec_trace_generator.py` → `spec_build_dataset.py` → `spec_evaluate.py` → `spec_plot.py` |

Each `*_evaluate.py` takes `--model <openrouter-model-id>` (e.g.
`google/gemini-2.5-pro`, `openai/gpt-4.1`, `anthropic/claude-sonnet-4`,
`meta-llama/llama-3.3-70b-instruct`) and a `--workers` flag for parallel
requests. Run with `--help` for the full list of options.

## Sanity checking ground-truth labels

The `*_verify.py` scripts re-derive ground-truth labels from the generated
dataset by simulating LTL progression over the traces. They are independent
of the LLM-evaluation step and can be run to spot-check that the dataset
labels match what the formulas actually require.

## Repository layout

```
simple_*             Temporal elasticity (simple formula)
tree_*               Temporal elasticity (complex tree-LTL formula)
multi_simple_*       Constraint scalability (simple formula)
forest_*             Constraint scalability (complex formula)
prop_*               Proposition scalability (simple formula)
prop_tree_*          Proposition scalability (complex formula)
spec_*               Specification-format experiment (appendix)
utils.py             Shared OpenRouter API wrapper
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{alamdari2026auditing,
  title={Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced {AI} Systems},
  author={Alamdari, Parand A. and Klassen, Toryn Q. and McIlraith, Sheila A.},
  booktitle={Proceedings of the Conference on Fairness, Accountability, and Transparency (FAccT)},
  year={2026},
}
```
