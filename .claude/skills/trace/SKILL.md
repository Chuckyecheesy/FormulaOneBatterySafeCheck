---
name: trace
description: Chronological code execution and line tracing. Gives explicit, non-summarized, line-by-line explanations of code, and translates computations into mathematical notation. Use when the user types /trace, /explain or /translate, asks to translate code or a computation into a math formula, asks "how does this work" or "walk me through this code", or highlights a block of code for review.
---

# SKILL: Chronological Code Execution & Line Tracing

## Skill Objective
To completely break down code files into explicit, non-summarized, line-by-line explanations. This skill eliminates "lazy summarizing" and forces absolute visibility into the execution stack, state changes, and logic flow.

## Activation Triggers
This skill activates automatically whenever the user:
- Uses the keyword `/explain` or `/trace`
- Uses the keyword `/translate` (runs **Translate mode**, below)
- Asks "how does this work" or "walk me through this code"
- Highlights a specific block of code for review

## Strict Execution Steps
When this skill is triggered, you MUST execute these exact steps in order:

### Step 1: Context Isolation
Isolate the entry point of the code. State what triggers this code block and what parameters/data are injected into it.

### Step 2: Continuous Line-by-Line Tracing
Evaluate the code sequentially. You are forbidden from skipping lines or grouping unrelated operations. Use the following structured format for the walkthrough:

- **Line [X]:** `[Exact Code Snippet]`
  - **Mechanics:** What this line tells the engine/runtime to do in plain English.
  - **State Impact:** What variables change, what memory is allocated, or what side effects occur.
  - **Why:** The architectural rationale behind writing it this way instead of an alternative.

**Computations:** When the code performs a computation, translate it into readable mathematical notation (for example, $\sum$ for a sum, $\mathbb{I}(\cdot)$ for an indicator, $\lvert x \rvert$ for an absolute value), and define every symbol used. Follow the **Accent notation** rules in Translate mode (for example, `y_pred` → ŷ).

### Step 3: Edge-Case Evaluation
Identify any specific inputs, empty arrays, null values, or async racing conditions that would cause this specific sequence of lines to fail or throw an exception.

## Translate mode (`/translate`)
When the user uses `/translate`, translate the computation in the given code into a readable math formula instead of running the full line-by-line trace:

1. **Formula:** Write the whole computation as one formula in mathematical notation, using LaTeX (`$$ ... $$`). Use standard symbols: $\sum$ for sums, $\prod$ for products, $\mathbb{I}(\cdot)$ for indicators/conditions, $\lvert x \rvert$ for absolute value, $\bar{y}$ for a mean, $\max$/$\min$, $\operatorname{round}_n$ for rounding, and piecewise braces for `if`/`else` branches.
2. **Symbols:** Define every symbol in the formula and map it to the variable name in the code.
3. **Code ↔ math table:** For each line that contributes to the computation, show the code snippet next to its math equivalent.
4. **Worked example:** Plug small concrete numbers into the formula and show the result step by step.
5. **Differences:** Point out anywhere the code differs from the textbook formula (for example rounding, clipping, or floating-point handling).

Loops map to $\sum$/$\prod$ over their index range; nested loops map to nested sums. Do not refactor the code.

**Accent notation:** Write decorated symbols as the symbol itself, never as words or code names:

| Code name / wording | Write in LaTeX | Rendered |
|---|---|---|
| `y_pred`, `y_hat`, "hat(y)", "predicted y" | `\hat{y}` | ŷ |
| `y_mean`, `y_bar`, "bar(y)", "mean of y" | `\bar{y}` | ȳ |
| `x_tilde`, "tilde(x)" | `\tilde{x}` | x̃ |
| `v_dot`, "dot(v)", "rate of v" | `\dot{v}` | v̇ |
| `y[i]`, `y_i` | `y_i` | yᵢ |

Outside `$$ ... $$` blocks (for example in tables and inline text), use the Unicode form (ŷ, ȳ, x̃, v̇, yᵢ) so it still reads correctly where LaTeX isn't rendered.

## Prohibited Behaviors
- **NEVER** use hand-waving phrases like *"Lines 15-30 handle the user authentication setup..."*
- **NEVER** skip boilerplate or setup lines if they initialize critical state.
- **NEVER** rewrite or refactor the code during the explanation step unless explicitly requested by the user. Keep focus entirely on explaining the existing lines.
