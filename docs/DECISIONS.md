# Decision log

Format: one entry per decision, newest at the bottom. Keep it short; it feeds the report.

```
## D-XXX: <title>
Date: YYYY-MM-DD | Author: <name>
Context: what problem, what constraints
Options: A ..., B ...
Decision: ...
Why: ...
```

## D-001: Scope is one patient, overfitted on purpose
Date: (fill in) | Author: (fill in)
Context: course project with limited time and compute; no ready-made project exists for SIREN on 4D cardiac MRI.
Options: (A) generalising model across patients; (B) per-patient implicit representation.
Decision: B, with levels L1 (static 3D), L2 (4D), L3 (stretch), and a fallback rule at the end of W3.
Why: B is the standard setting for implicit representations, fits the timeline, and still demonstrates the full method.
