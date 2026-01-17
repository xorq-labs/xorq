# Phase Completion Checks

## Task Understanding Checklist
- Restate the user request and confirm success criteria
- Identify which tables, files, or models are relevant
- Note required outputs (plots, tables, summaries)
- Decide whether the task is ML, data prep, optimization, or reporting
- Plan the first code block before writing anything

## Phase Check: Initialization
- Have you clarified the desired outputs?
- Do you know which connections / tables will be used?
- Did you outline the first execution step?

**If not, pause and plan before writing code.**

## Phase Check: Data Preparation
- Schema verified and column names adjusted?
- Required filters, joins, or feature columns created?
- Data cached or backend switched where necessary for UDFs/ML?

**Only move on once the dataset is truly modeling-ready.**

## Phase Check: Data Transform
Have you completed all data transformations?
- Filtering applied?
- Aggregations done?
- Joins completed?

**If yes, move to next phase.**

## Phase Check: Modeling
- Model(s) trained with the prepared features?
- Metrics calculated on validation/test data?
- Comparison of alternatives captured in markdown?

**If any of these are missing, continue iterating before summarizing.**

## Phase Check: Communication
- Visuals rendered and explained?
- Key findings summarized with supporting numbers?
- Recommendations or next actions stated clearly?

**Wrap up only after the story is understandable without reading raw code.**
