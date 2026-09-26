---
name: mathformat
description: MathFormat. Translates LaTeX math, or a computation in code, into a typeset mathematical expression shown as a rendered equation (Unicode), not raw LaTeX source. Use when the user types /mathformat, asks to render, typeset or pretty-print a LaTeX formula, or asks to show a computation as a rendered equation.
---

# SKILL: MathFormat — LaTeX → Rendered Equation

## Skill Objective
Turn LaTeX source (for example `\frac{1}{N}\sum_{i=1}^{N}`) or a computation in code into a **typeset mathematical expression** that reads as a rendered equation anywhere, including terminals and editors that do not render LaTeX.

```
LaTeX / code computation  →  typeset mathematical expression  →  rendered equation
```

## Activation Triggers
- The user types `/mathformat`
- The user asks to "render", "typeset", or "pretty-print" a LaTeX formula
- The user asks to show a computation "as an equation" or "in math format"

## Execution Steps

### Step 1: Get the source
- If the input is **LaTeX**, use it as-is.
- If the input is **code**, first translate the computation into LaTeX (loops → $\sum$ / $\prod$, `if`/`else` → piecewise braces, `abs` → $\lvert x \rvert$, `mean` → $\frac{1}{N}\sum$ or $\bar{x}$), following the Translate mode rules in the `trace` skill.

### Step 2: Typeset it as a rendered equation
Write the equation using Unicode math characters, not LaTeX commands. Put it on its own lines and align multi-line parts in a fenced block with no language tag, so the spacing is preserved.

Conversion rules:

| LaTeX | Rendered |
|---|---|
| `\frac{a}{b}` (simple) | a / b |
| `\frac{a}{b}` (display) | a stacked over b with a `───` bar, centred |
| `\sum_{i=1}^{N}` | Σ with `N` above and `i=1` below (display), or Σᵢ₌₁ᴺ (inline) |
| `\prod_{i=1}^{N}` | Π, laid out the same way as Σ |
| `\int_a^b` | ∫ₐᵇ |
| `x^2`, `x^{n}` | x², xⁿ (superscript characters) |
| `x_i`, `x_{0}` | xᵢ, x₀ (subscript characters) |
| `\hat{y}`, `\bar{y}`, `\tilde{x}`, `\dot{v}` | ŷ, ȳ, x̃, v̇ |
| `\sqrt{x}` | √x, or √(x + 1) for longer arguments |
| `\lvert x \rvert` | \|x\| |
| `\mathbb{I}(c)` | 𝕀(c) |
| `\le`, `\ge`, `\ne`, `\approx` | ≤, ≥, ≠, ≈ |
| `\times`, `\cdot`, `\pm` | ×, ·, ± |
| `\in`, `\infty`, `\to` | ∈, ∞, → |
| `\alpha … \omega`, `\Delta`, `\delta` | α … ω, Δ, δ |
| `\max`, `\min`, `\operatorname{round}_9` | max, min, round₉ |
| `\begin{cases} … \end{cases}` | a left brace built from ⎧ ⎨ ⎩, one case per line |

If a superscript or subscript character doesn't exist in Unicode (for example a subscript `q` or `δ`), write it as `_q` or `^q` (use parentheses for longer ones, such as `_(max)`) rather than dropping it.

### Step 3: Show both forms
Always output, in this order:
1. **Rendered equation:** the typeset Unicode version from Step 2.
2. **LaTeX source:** the original or generated LaTeX in a `$$ ... $$` block, so it can be pasted into a document.
3. **Symbols:** one line per symbol, saying what it means (and which code variable it is, if the source was code).

## Example

**Input:** `\text{TolAcc}_{\delta} = \frac{100}{N}\sum_{i=1}^{N}\mathbb{I}\left(\lvert y_i-\hat{y}_i\rvert\le\delta\right)`

**Rendered equation:**
```
                   N
            100    ⎲
TolAcc_δ = ───── · ⎳  𝕀( |yᵢ − ŷᵢ| ≤ δ )
             N    i=1
```

**LaTeX source:**
$$\text{TolAcc}_{\delta} = \frac{100}{N}\sum_{i=1}^{N}\mathbb{I}\left(\lvert y_i-\hat{y}_i\rvert\le\delta\right)$$

**Symbols:** N = number of rows · yᵢ = true value · ŷᵢ = predicted value · δ = tolerance band · 𝕀 = 1 if the condition holds, else 0

## Prohibited Behaviors
- **NEVER** output only raw LaTeX when a rendered equation was asked for.
- **NEVER** write accents or operators as words (`hat(y)`, `sum`, `<=`) in the rendered equation.
- **NEVER** change the mathematics while formatting it. If the source looks wrong, render it as given and point out the issue separately.
