# Quick Patterns

## Optimization Approach
1. Define constraints clearly
2. Build solution incrementally
3. Track objective function
4. Validate constraints at each step

## Plotting Patterns
- Use matplotlib/seaborn in the notebook; call `plt.show()` once per figure
- Keep datasets small for plotting: `expr.limit()` or summary tables
- Label axes and add titles that explain the insight
- Prefer clear chart types (line, bar, scatter) unless the task demands more
- Mention any important parameters (bins, colors) in markdown to aid interpretation

## Result Summary Patterns
- Recap the objective in one sentence
- Highlight the key quantitative findings or metrics
- Call out limitations, assumptions, or next steps
- Reference the code cells/plots that support each claim
- Close with a clear recommendation or follow-up action
