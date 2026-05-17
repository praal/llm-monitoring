# Monitoring, Auditing, and Intervention for LLM Agents

Code for the auditing, monitoring, and intervention experiments in our paper. The
scripts in this folder evaluate whether an LLM agent's action trace satisfies a
set of temporal (LTL) rules in three domains:

| Domain        | Auditor              | Labeler key   | Agent / Planner       |
|---------------|----------------------|---------------|-----------------------|
| ScienceWorld  | `auditScience.py`    | `scienceworld`| `science-agent.py`    |
| TextWorld     | `auditTextWorld.py`  | `textworld`   | `textworld-agent.py`  |
| Trucks (IPC)  | `auditTruck.py`      | `qual20`      | `truck-plan.py`       |

The auditors can run in four modes:

1. **LTL monitor** (`--ltl`) — the symbolic monitor in `monitor.py` progresses
   the LTL formulas over the trace and counts violations exactly.
2. **Zero-shot LLM auditor** (`--llm`) — the LLM is given the rules in natural
   language and the action trace, and asked to count violations.
3. **In-context LLM auditor** (`--icl`) — same as `--llm` but with a few-shot
   prompt.
4. **LLM auditor with state-oracle labels** (`--iclwithlabels` / `--llmlabel`)
   — the LLM additionally sees per-step propositional labels produced by
   `labler.py`.

Agent / planner scripts (`science-agent.py`, `textworld-agent.py`,
`truck-plan.py`) run the actual interactive episode and optionally apply one of
three monitor-driven interventions when the monitor predicts an upcoming
violation: `--inject` (safety-warning prompt injection), `--resample`
(best-of-N resampling scored by the monitor), and `--switch` (switch to a
safer prompt).

## Setup

```bash
pip install numpy requests aiohttp scienceworld textworld
export OPENROUTER_API_KEY="sk-or-v1-..."
```

All LLM calls go through OpenRouter (`utils.prompt_model` and
`sampler.BestOfNSampler`), so any model hosted there can be used — pass it via
`--model <openrouter-model-id>`, e.g. `meta-llama/llama-3.3-70b-instruct`
(default), `google/gemini-2.5-flash`, `openai/gpt-4.1`,
`anthropic/claude-sonnet-4`.

For the TextWorld agent you additionally need a TextWorld `.z8` game placed at
`./textworld/envs/<env_name>.z8`. The `.ni` / `.json` files for the cooking
game shipped in `textworld/tw_games/` can be compiled to `.z8` with the
`tw-make` / `inform7` tools from the `textworld` package.

All scripts must be run from this folder (`Monitoring/`) since they use
relative paths like `./tests/...`, `./prompts/...`, `./commands/...`, and
`./outputs/...`. Create the output directories before running:

```bash
mkdir -p outputs/{audit-science,audit-science-raw,audit-textworld,audit-textworld-raw,\
audit-truck20,audit-truck20-raw,science-label,textworld-label,qual20-label,\
plan-science,plan-textworld,plan-truck}
```

## Directory layout

```
auditScience.py        Auditor for ScienceWorld traces
auditTextWorld.py      Auditor for TextWorld traces
auditTruck.py          Auditor for IPC Trucks (qual-20) traces
labler.py              Builds per-step propositional state labels
science-agent.py       LLM agent for ScienceWorld with monitor-in-the-loop
textworld-agent.py     LLM agent for TextWorld with monitor-in-the-loop
truck-plan.py          LLM planner for the Trucks domain with monitor-in-the-loop
monitor.py             LTL monitor used by every auditor and agent
ltl.py                 LTL formula AST and progression
sampler.py             Best-of-N sampler scored by the monitor
utils.py               OpenRouter API wrapper and shared helpers

tests/                 Action traces and rule files used by the auditors
  scienceworld/        ScienceWorld traces (out1.txt … out10.txt)
  textworld/           TextWorld traces (out1.txt … out9.txt)
  trucks/              IPC Trucks traces (out1.txt … out18.txt)
  IPC/                 Trucks domain + qual-20 rules (NL and LTL)
  science-rules.txt    ScienceWorld rules in natural language
  textworld.txt        TextWorld rules in natural language
  textworld-icl.txt    TextWorld ICL prompt rules
prompts/               System and audit prompts used by the LLM auditors/labelers
commands/              Rules, task descriptions, and allowed actions for the agents
textworld/             TextWorld game files and constraint templates
outputs/               Auditor / labeler / agent outputs (created on demand)
```

## 1. Auditing existing traces

The auditors loop over every `.txt` file in the relevant `tests/<domain>/`
directory and write one output per trace. Common flags:

| Flag              | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `--model M`       | OpenRouter model id (only used by LLM modes)                  |
| `--filter STR`    | Only audit files whose name contains `STR`                    |
| `--ex STR`        | Extra suffix appended to output filenames                     |
| `--func STR`      | Sanitized model id of the labeler that produced the labels    |
| `--labelfile STR` | (LTL mode) sanitized labeler id whose labels feed the monitor |

### `auditScience.py`

```bash
# Symbolic LTL monitor over rule-based propositional labels
python auditScience.py --ltl

# LTL monitor consuming LLM-generated labels (label file suffix == sanitized model id)
python labler.py --test scienceworld --model meta-llama/llama-3.3-70b-instruct
python auditScience.py --ltl --labelfile meta-llama_llama-3.3-70b-instruct

# Zero-shot LLM auditor
python auditScience.py --llm --model meta-llama/llama-3.3-70b-instruct

# In-context (few-shot) LLM auditor
python auditScience.py --icl --model meta-llama/llama-3.3-70b-instruct

# In-context auditor that also reads per-step labels generated by --func
python auditScience.py --iclwithlabels --model meta-llama/llama-3.3-70b-instruct \
    --func meta-llama_llama-3.3-70b-instruct
```

Traces are read from `./tests/scienceworld/`, rules from
`./tests/science-rules.txt`, and outputs go to `./outputs/audit-science/`
(monitor logs) or `./outputs/audit-science-raw/` (raw LLM answers).

### `auditTextWorld.py`

```bash
python auditTextWorld.py --ltl
python auditTextWorld.py --ltl --labelfile <sanitized-model-id>
python auditTextWorld.py --llm  --model meta-llama/llama-3.3-70b-instruct
python auditTextWorld.py --icl  --model meta-llama/llama-3.3-70b-instruct
python auditTextWorld.py --llmlabel --model meta-llama/llama-3.3-70b-instruct \
    --func <sanitized-model-id>
```

Traces from `./tests/textworld/`, rules from `./tests/textworld.txt` (or
`./tests/textworld-icl.txt` for `--icl`), outputs in
`./outputs/audit-textworld[-raw]/`.

### `auditTruck.py`

```bash
python auditTruck.py --ltl
python auditTruck.py --ltl --labelfile <sanitized-model-id>
python auditTruck.py --llm  --model meta-llama/llama-3.3-70b-instruct
python auditTruck.py --icl  --model meta-llama/llama-3.3-70b-instruct
python auditTruck.py --llmlabel    --model meta-llama/llama-3.3-70b-instruct --func <sanitized-model-id>
python auditTruck.py --iclwithlabels --model meta-llama/llama-3.3-70b-instruct --func <sanitized-model-id>
```

Traces from `./tests/trucks/`, rules from `./tests/IPC/truck-qual20-rules.txt`
(natural language) and `./tests/IPC/truck-qual20-LTL.txt` (LTL form), outputs
in `./outputs/audit-truck20[-raw]/`.

## 2. Generating per-step labels — `labler.py`

`labler.py` produces, for every action in every test trace, a dictionary of
propositional values (e.g. `{"go": True, "kitchen": False, ...}`). These
labels are consumed by the auditors in `--labelfile`, `--llmlabel`, and
`--iclwithlabels` modes.

```bash
python labler.py --test scienceworld --model meta-llama/llama-3.3-70b-instruct
python labler.py --test textworld    --model meta-llama/llama-3.3-70b-instruct
python labler.py --test qual20       --model meta-llama/llama-3.3-70b-instruct
```

Flags:

| Flag         | Meaning                                                              |
|--------------|----------------------------------------------------------------------|
| `--test`     | One of `scienceworld`, `textworld`, `qual20` (selects the LTL/dirs)  |
| `--model`    | OpenRouter model id used to label and to name the output file        |
| `--ex`       | Extra suffix appended to output filenames                            |

Output goes to `./outputs/<domain>-label/<trace-name>---<sanitized-model>--labels.txt`.
The sanitized model id is what you pass to the auditors as `--labelfile` /
`--func`. In the current `main`, the labeler runs in rule-based mode
(`labeler.run(True)`); flipping it to `False` enables the LLM-based labeling
prompts in `prompts/labeling-*`.

## 3. Running agents with monitor-in-the-loop 

The agent scripts execute an interactive episode and call `Monitor.step(...)`
on every action. They share the same intervention flags:

| Flag           | Meaning                                                                                  |
|----------------|------------------------------------------------------------------------------------------|
| `--model M`    | OpenRouter model id used both for the agent and the best-of-N judge                       |
| `--max-steps N`| Cap on episode length                                                                    |
| `--inject`     | When predicted violation rate ≥ threshold, inject a safety warning into the next prompt   |
| `--resample`   | Best-of-N sampling (scored by `trace_scorer` against the monitor) when risk is high       |
| `--switch`     | Switch to a "safer" system prompt when risk is high                                       |
| `--verbose`    | Print extra per-step info (TextWorld / Trucks only)                                       |

Each script logs the trace to `./outputs/plan-<domain>/` and prints a final
summary line with the total number of violations.

### `science-agent.py`

Loads the ScienceWorld task `chemistry-mix-paint-secondary-color` (set in
`main`). Uses the rules in `commands/science-rules.txt` and the LTL in
`auditScience.scienceworldLTL2()`.

```bash
python science-agent.py --model meta-llama/llama-3.3-70b-instruct --max-steps 30
python science-agent.py --model meta-llama/llama-3.3-70b-instruct --inject
python science-agent.py --model meta-llama/llama-3.3-70b-instruct --resample
python science-agent.py --model meta-llama/llama-3.3-70b-instruct --switch
```

`--inject`, `--resample`, `--switch` are boolean flags (no value).

### `textworld-agent.py`

Loads `./textworld/envs/<env>.z8`. Uses the rules in
`commands/textworld-rules.txt` and the LTL in this script's
`textworldLTL()`.

```bash
python textworld-agent.py --env <env_name> --model meta-llama/llama-3.3-70b-instruct \
    --max-steps 60
python textworld-agent.py --env <env_name> --inject  true
python textworld-agent.py --env <env_name> --resample true
python textworld-agent.py --env <env_name> --switch   true
```

For `textworld-agent.py` the intervention flags take a boolean value
(`--inject true`), unlike the other two scripts where they are bare flags.

### `truck-plan.py`

Open-loop planner for the Trucks domain — no environment, the LLM produces
the entire action sequence and the monitor scores it. Uses the rules in
`commands/truck-rules.txt` and the LTL in `qual20_LTL()` defined inside the
script.

```bash
python truck-plan.py --model meta-llama/llama-3.3-70b-instruct --max-steps 30
python truck-plan.py --inject  true
python truck-plan.py --resample true
python truck-plan.py --switch   true
```

Like `textworld-agent.py`, the intervention flags take a boolean value.

## Output conventions

* `audit-*-raw/` — verbatim LLM auditor responses (one file per trace × mode).
* `audit-*/` — symbolic LTL monitor logs (one Python-literal dict per step,
  ending with the cumulative `total_violations` map).
* `<domain>-label/` — labeler output: `<trace>---<model>--labels.txt`
  (one dict per step) and `<trace>---<model>--prompts.txt` (raw LLM labeling
  responses when the LLM labeler is enabled).
* `plan-<domain>/` — per-episode logs from the agents (`--actions.txt`,
  `--states.txt`, `--monitor.txt`).

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
