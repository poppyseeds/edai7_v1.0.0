---
name: repository-analysis
description: 'Analyze a repository and explain its architecture, logical execution flow, module connections, dependencies, data flow, entry points, and outputs. Use for codebase exploration, repository walkthroughs, architecture summaries, pipeline tracing, onboarding documentation, and beginner-friendly project analysis.'
argument-hint: 'Describe the repository or scope to analyze'
user-invocable: true
disable-model-invocation: false
---

# Repository Analysis

## Purpose

Produce a concise, beginner-friendly explanation of how a repository is organized and how its main workflow executes. Focus on the repository's internal logic and cite key files by workspace-relative path.

## Procedure

1. **Establish the repository boundary**
   - Identify the workspace root and inspect its top-level files and folders.
   - Exclude generated output, caches, virtual environments, vendored code, and unrelated external content unless they affect runtime behavior.
   - Check repository guidance files such as `README.md`, `AGENTS.md`, `copilot-instructions.md`, and build or deployment manifests.

2. **Inventory the structure**
   - Group files into logical areas such as entry points, API or UI, domain logic, data access, models, utilities, configuration, tests, and deployment.
   - Identify the primary language, framework, package manager, and test tooling from manifests and configuration.
   - Note important external libraries only when they influence the execution pipeline.

3. **Find entry points**
   - Locate executable entry points such as CLI commands, web server startup files, application factories, scheduled jobs, workers, and frontend bootstrap files.
   - Use documentation, package scripts, Docker files, and deployment configuration to confirm which entry points are actually used.
   - Distinguish primary paths from examples, tests, optional integrations, and dead or legacy paths when evidence supports that distinction.

4. **Trace the main execution path**
   - Start at the primary entry point and follow calls through the nearest controlling functions or classes.
   - Record the order in which modules validate inputs, load or transform data, invoke services or models, persist state, and produce responses or files.
   - Follow imports and call sites rather than inferring behavior from filenames alone.
   - Stop at external libraries or services and explain their role without analyzing unrelated internals.

5. **Map component interactions**
   - Explain how major folders or modules connect and what each boundary owns.
   - Identify the architectural pattern when supported by the code, such as layered architecture, pipeline/orchestrator, MVC, service-oriented, plugin-based, or event-driven.
   - Call out important shared schemas, configuration objects, interfaces, base classes, and adapters.

6. **Describe data flow**
   - Track the main input format from its origin through validation, parsing, transformation, computation or generation, evaluation, storage, and final output.
   - Name important intermediate representations and output artifacts.
   - Include alternate branches such as labeled versus unlabeled data, optional providers, retries, fallbacks, or test-only paths when they materially change behavior.

7. **Check the explanation against evidence**
   - Verify every reported entry point and major transition against a file, import, function call, package script, or test.
   - Compare the described workflow with at least one relevant test or documented run command.
   - Mark uncertain conclusions as assumptions instead of presenting them as facts.
   - Do not claim runtime behavior that was not observed or supported by the repository.

8. **Write the result**
   - Start with a short architecture overview.
   - Present the execution pipeline as numbered steps.
   - Use flat bullet points for key files and their roles.
   - Include a text pipeline diagram, for example:
     `entry point -> input loader -> validation -> core service -> persistence/output`
   - Include a compact dependency summary and a data-flow summary.
   - Keep the report modular so new components can be added without rewriting unrelated sections.
   - Avoid raw code except for a small excerpt needed to clarify a non-obvious transition.

## Output Format

Use this structure unless the user requests a different format:

1. **Architecture overview**: primary pattern, major layers, and runtime boundary.
2. **Key files**: flat bullets with workspace-relative file links and one-line roles.
3. **Step-by-step execution**: numbered path from entry point to outputs.
4. **Pipeline diagram**: one or more text-form flow lines.
5. **Data flow**: inputs, transformations, intermediate state, and outputs.
6. **Dependencies**: only libraries and services that influence the workflow.
7. **Tests and verification**: relevant tests or commands that support the explanation.
8. **Assumptions or gaps**: unresolved paths, optional behavior, or unavailable runtime evidence.

## Quality Checklist

Before finishing, confirm that:

- The repository boundary and exclusions are explicit.
- At least one real entry point and its output are identified.
- Each major pipeline transition names the responsible module or file.
- The architecture description is grounded in imports, calls, configuration, or tests.
- Data flow explains both where data starts and where it ends.
- Dependencies are tied to a concrete role instead of merely listed.
- The text diagram agrees with the numbered execution steps.
- Optional branches and uncertainty are labeled clearly.
- The explanation is concise, modular, beginner-friendly, and free of unnecessary raw code.
