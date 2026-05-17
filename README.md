# Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced AI Systems (FAccT 2026)

Code for the paper *Formal Methods Meet LLMs: Auditing, Monitoring, and
Intervention for Compliance of Advanced AI Systems* (FAccT 2026).


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The repository is split into two self-contained subprojects, each with its own
README and setup instructions:

- [`LLMFailure/`](LLMFailure/README.md) — LLM-as-a-Judge temporal-reasoning
  failure-mode experiments (Section 5.2 of the paper). Synthetic event traces are rendered as natural
  language and LLMs are asked to judge whether each trace satisfies a given
  LTL constraint. Reproduces the figures on temporal elasticity, constraint
  scalability, proposition scalability, and specification format. 

- [`Monitoring/`](Monitoring/README.md) — auditing, monitoring, and
  intervention pipeline for LLM agents in three domains (ScienceWorld,
  TextWorld, and IPC Trucks). Includes the symbolic LTL monitor, LLM-based
  auditors, per-step state labelers, and agents/planners that use the monitor
  for prompt injection, best-of-N resampling, and prompt switching.

See each subfolder's `README.md` for dependencies and run instructions.

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
