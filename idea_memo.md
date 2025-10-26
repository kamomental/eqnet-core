# Idea Memo

## IDEA-001: DSPy Integration (Reflection Mode Only)
- **Context**: Discussed integrating DSPy to improve the quality of reflective diary outputs while keeping core thermodynamic controls unchanged.
- **Proposal**: Keep the default (real-time) mode as-is. In reflection mode, optionally route diary/analysis text generation through DSPy to auto-tune prompts.
- **Benefits**:
  - Automated prompt optimisation for diary tone and weekly reports.
  - Reduced manual maintenance of language templates.
  - Potential to incorporate user feedback metrics for continuous improvement.
- **Risks / Considerations**:
  - Adds dependency/latency; must guard against overfitting to textual metrics.
  - Need versioning of DSPy programs for transparency and audit.
  - Requires fallbacks when DSPy or LLM is unavailable.
- **Status**: Deferred. No implementation now; revisit when reflection quality tuning becomes a priority.
