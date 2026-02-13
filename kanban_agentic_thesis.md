# Thesis Sprint Kanban (Feb 13 – Mar 3, 2026)

> Goal by **Mar 3, 2026**: ✅ Working agentic chatbot + ✅ Complete written thesis (defense-ready)

---

## Backlog (Nice-to-have / only if time)
- [ ] Add lightweight UI demo (Streamlit) for defense (optional)
- [ ] Add simple “reflection” / error-recovery step (optional)
- [ ] Add cost/latency analysis table (optional)
- [ ] Add extra baselines (only if experiments are already done)

---

## To Do (This Sprint)
### Agent (Implementation)
- [ ] Stabilize router logic (intent + confidence + fallbacks)
- [ ] Fix state overwrite issues (messages, current_flow, intent fields)
- [ ] Confirm multi-turn billing flow works end-to-end
- [ ] Confirm escalation flow works end-to-end
- [ ] Clean separation: `main.py` vs `src/nodes/*.py` (no circular imports)
- [ ] Add structured logging (minimal, consistent, removable)
- [ ] Freeze architecture (no new nodes after freeze date)

### Retrieval (RAG)
- [ ] Validate FAISS index loading path + error handling
- [ ] Confirm retrieval injection format (system vs tool context vs appended)
- [ ] Tune retrieval params (k, chunk format) — choose final settings
- [ ] Add “no retrieval” toggle for ablation experiments

### Agentic “Minimum Upgrade” (pick ONE and finish it)
- [ ] Option A: Wrap retrieval as a “tool” abstraction
- [ ] Option B: Add LLM decision step before routing (route selection)
- [ ] Integrate KPI/Vibe signal into state + routing decision (lightweight but real)

### Experiments & Evaluation
- [ ] Create evaluation prompt set (50 prompts total)
  - [ ] 20 technical support
  - [ ] 10 billing (multi-turn)
  - [ ] 10 escalation
  - [ ] 10 info_lookup/chitchat/unknown
- [ ] Define metrics (simple, defensible)
  - [ ] Intent accuracy (%)
  - [ ] Multi-turn success rate (%)
  - [ ] Retrieval helpfulness (binary or 1–5 rubric)
  - [ ] Hallucination/unsupported claims rate (binary)
- [ ] Run Ablation #1: Retrieval ON vs OFF
- [ ] Run Ablation #2: Memory ON vs OFF
- [ ] Compile results into tables (CSV → thesis tables)
- [ ] Failure analysis: top 5 failure modes + examples

### Thesis Writing
- [ ] Update thesis outline + section headers (final structure)
- [ ] Chapter 1: Introduction (problem, motivation, contributions)
- [ ] Chapter 2: Background / Related Work (agentic systems, RAG, dialogue)
- [ ] Chapter 3: Methodology (architecture, state graph, nodes, data, retrieval)
- [ ] Chapter 4: Experimental Setup (prompts, metrics, configs, baselines)
- [ ] Chapter 5: Results (tables + qualitative examples)
- [ ] Chapter 6: Discussion (why results look like this, limitations)
- [ ] Chapter 7: Conclusion + Future Work
- [ ] Abstract (write last)
- [ ] References + citations cleanup

### Diagrams & Presentation Assets
- [ ] Export LangGraph diagram (final graph)
- [ ] Architecture diagram (1-page system overview)
- [ ] Data flow diagram (retrieval + memory + routing)
- [ ] Tables/figures captions + numbering

### Final Polish / Submission
- [ ] Code cleanup + README “How to run”
- [ ] Add `requirements.txt` / environment notes
- [ ] Add reproducibility checklist (seed, configs, model names)
- [ ] Final proofreading pass
- [ ] Final PDF export
- [ ] GitHub tag/release for submission snapshot

---

## In Progress
- [ ] (Move cards here while actively working)

---

## Review / QA (Must pass before “Done”)
### Agent QA Checklist
- [ ] 10/10 runs without crashing
- [ ] Multi-turn billing works (at least 3 scenarios)
- [ ] Escalation works (at least 2 scenarios)
- [ ] Retrieval helps on retrieval-dependent questions
- [ ] Fallback works when intent is unclear
- [ ] Logs are readable and not noisy

### Thesis QA Checklist
- [ ] All chapters present
- [ ] Figures/tables numbered + referenced in text
- [ ] Citations consistent
- [ ] Clear contributions listed
- [ ] Limitations included (hallucinations, evaluation scope, domain shift)

---

## Done
- [ ] (Move completed items here)

---

## Milestones
- **Feb 19:** Architecture frozen + stable agent end-to-end
- **Feb 24:** Experiments complete + results tables drafted
- **Feb 28:** Full thesis first complete draft
- **Mar 2:** Final edits + formatting + references
- **Mar 3:** Submission-ready PDF + final GitHub snapshot

