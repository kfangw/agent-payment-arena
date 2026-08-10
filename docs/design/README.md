# Design docs

These documents explain what each part of the repository does and why it is
built the way it is. The reasons given here are engineering reasons:
measurement validity, attack surface, failure modes, and reproducibility.
Every document follows the same outline: purpose, behavior, design decisions,
limits, and status.

The status section is what keeps these notes honest. Most of this repository
is not built yet, and a design note that describes unwritten code as though it
runs is worse than no note. Each document states which part exists. Planned
work lives in [ROADMAP.md](../../ROADMAP.md); scope lives in
[threat-model.md](../threat-model.md) and
[what-this-measures.md](../what-this-measures.md).

- [contract.md](contract.md): the x402 contract both gateway backends are held to, and why it is stated separately from either implementation.
- [backends.md](backends.md): the in-memory and HTTP backends, and the differential test that makes results from the fake one worth reading.
- [agents.md](agents.md): the four agents under evaluation and the defense each one represents.
- [attacks.md](attacks.md): the attack catalog, the benign twins it must ship with, and where an adversary is allowed to write.
- [delegator.md](delegator.md): the delegator response model, and why escalation is priced rather than assumed free.
- [metrics.md](metrics.md): the five quantities a run produces and why none of them may be reported alone.
- [reproducibility.md](reproducibility.md): cassettes, seeds, repetitions, and version recording.
