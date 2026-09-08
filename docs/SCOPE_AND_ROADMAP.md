# Scope and roadmap

What this repository does today, and what it deliberately does not.

## In scope, measured

Queueing sizing (Erlang B recursion, Erlang A via the M/M/c+M stationary
distribution), MILP shift assignment over a 15-minute weekly grid, three
operating patterns proven optimal, a greedy baseline for comparison, an
shift-swap screening, calendar-day grids and Excel workbooks. CBC and CP-SAT
both remain tested, but the comparison runner has been removed and the two
objectives differ.

## Out of scope, and why

These were implemented and removed rather than left in `src/` unbenchmarked.
A module that is not wired into `patterns.py` and not exercised in CI is a
promissory note, and reviewers are right to poke at it.

**Multi-channel concurrency.** Blending synchronous voice (1:1) with
asynchronous digital contacts (1:k) requires a capacity model the MILP does not
have: an agent handling three chats is not three agents. This is the
prerequisite for a fourth operating pattern (blended voice and asynchronous)
and is the single most distinctive extension available.

**Discrete-event simulation.** SimPy replay of a solved schedule against
stochastic arrivals, to measure realised service level rather than modelled
coverage. The right validation layer, but only meaningful once forecasting
exists to disagree with.

**Historical demand and forecasting.** The generator produces one seeded week.
Forecasting needs a long historical series, a time-respecting split, and
features restricted to what was knowable at decision time. Note that a forecast
trained on synthetically generated demand measures how much noise the generator
was given, not how predictable real demand is - a limitation to state plainly
in any result.

**Cost modelling.** The objective is denominated in penalty weights, not
currency. There is no wage data in this repository and no figure in it should
be read as money.

**Call taxonomy and routing.** Demand is a single homogeneous stream: contacts
and average handle time. There are no categories, intents, departments or
skills, so the model cannot answer which unit receives which contact.

## Known limits of what is here

- Synthetic data throughout. Every number is reproducible and none is observed.
- Difficulty tracks utilisation, not model size: an instance sized near its own
  ceiling is far harder than a larger one with slack.
- CBC omits surplus and wage costs; an optimal schedule need not minimize
  headcount or paid hours. The CP-SAT variant includes a surplus penalty.
- The active patterns use operational rest/hour rules, not statutory policy
  enforcement. The retained sector compiler is not the pattern solver's input.
- The swap endpoint is a partial screen: availability, skills and night caps
  are not part of its request schema.
