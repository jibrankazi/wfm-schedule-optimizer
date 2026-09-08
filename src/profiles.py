"""Service-level regimes and the duty taxonomy they operate on.

Two things vary between contact centres, and neither is the mathematics.

**The service contract.** How fast calls must be answered, how hard agents may
be run, and whether a queue is ever allowed to stand empty. A 95%-in-10s
target does not need slightly more staff than 80%-in-20s; it needs
substantially more, and this module makes that comparable rather than
assumed.

**The duty taxonomy.** Every operation names its non-queue work differently.
Hard-coding one organisation's codes into the solver makes the engine specific
to that organisation for no benefit, so codes are configuration.

Nothing here encodes a real employer's scheme. The defaults are generic
English words chosen to read clearly in a schedule grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .erlang import ServiceTarget


@dataclass(frozen=True)
class DutyCodes:
    """What the cells of a schedule grid are allowed to say.

    `primary_on_queue` is deliberately its own field rather than being pulled
    out of `on_queue` with `next(iter(...))`. Python randomises string hashing
    per process, so iteration order over a frozenset is not stable between
    runs - picking the first element would mark a drafted agent as overtime on
    one run and queue on the next.
    """

    primary_on_queue: str = "QUEUE"
    on_queue: frozenset[str] = frozenset({"QUEUE", "OVERTIME"})
    preemptable: tuple[str, ...] = ("QUALITY", "PROJECT", "CORRESPONDENCE", "TRAINING")
    unpaid_relief: frozenset[str] = frozenset({"MEAL"})
    paid_relief: frozenset[str] = frozenset({"REST"})
    absence: frozenset[str] = frozenset({"SICK", "VACATION", "LEAVE"})

    def __post_init__(self) -> None:
        if self.primary_on_queue not in self.on_queue:
            raise ValueError(
                f"primary_on_queue {self.primary_on_queue!r} must be one of "
                f"on_queue {sorted(self.on_queue)}"
            )
        if not self.preemptable:
            raise ValueError("preemptable must list at least one duty")
        overlap = self.on_queue.intersection(self.preemptable)
        if overlap:
            raise ValueError(f"codes cannot be both on-queue and preemptable: {sorted(overlap)}")
        duplicates = len(self.preemptable) - len(set(self.preemptable))
        if duplicates:
            raise ValueError("preemptable must not repeat a duty code")

    def normalise(self, value: object) -> str:
        """Grid cells arrive as whatever the source system wrote."""
        return str(value).strip().upper()

    def is_on_queue(self, value: object) -> bool:
        return self.normalise(value) in self.on_queue

    def all_codes(self) -> frozenset[str]:
        return frozenset(
            set(self.on_queue)
            | set(self.preemptable)
            | set(self.unpaid_relief)
            | set(self.paid_relief)
            | set(self.absence)
        )


@dataclass(frozen=True)
class ServiceProfile:
    """One operation's staffing contract.

    `min_presence_floor` is the constraint queueing theory cannot express. At
    03:00 a queue may offer a twentieth of an Erlang, and Erlang C will
    correctly answer "one agent, or none". An operation that must never go
    unanswered still staffs two, because the cost of an unmanned queue is not
    a service-level percentage. Floors are why overnight rosters are not a
    queueing problem.

    `abandons` records whether callers hang up rather than wait. It is not a
    proxy for how urgent the work is - it is the opposite. Callers to a
    commercial queue abandon readily. Callers with an emergency do not; they
    hold, and if disconnected they redial immediately, which is why modelling
    them with an abandonment term understates required capacity.
    """

    name: str
    target_sl: float
    target_time_sec: float
    max_occupancy: float
    min_presence_floor: int = 0
    abandons: bool = True
    mean_patience_sec: float = 90.0
    max_abandon_rate: float = 0.05
    hours: str = "weekday daytime"
    note: str = ""
    duty_codes: DutyCodes = field(default_factory=DutyCodes)

    def __post_init__(self) -> None:
        if not 0.0 < self.target_sl <= 1.0:
            raise ValueError("target_sl must be in (0, 1]")
        if self.target_time_sec < 0:
            raise ValueError("target_time_sec must be non-negative")
        if not 0.0 < self.max_occupancy <= 1.0:
            raise ValueError("max_occupancy must be in (0, 1]")
        if self.min_presence_floor < 0:
            raise ValueError("min_presence_floor must be non-negative")
        if self.abandons and self.mean_patience_sec <= 0:
            raise ValueError("an abandoning profile needs a positive mean_patience_sec")
        if not 0.0 <= self.max_abandon_rate <= 1.0:
            raise ValueError("max_abandon_rate must be in [0, 1]")

    @property
    def target(self) -> ServiceTarget:
        """The Erlang C sizing criteria, so existing code takes a profile unchanged."""
        return ServiceTarget(
            service_level=self.target_sl,
            target_time_sec=self.target_time_sec,
            max_occupancy=self.max_occupancy,
        )

    def label(self) -> str:
        return f"{self.target_sl:.0%}/{self.target_time_sec:.0f}s"

    def with_codes(self, codes: DutyCodes) -> ServiceProfile:
        return replace(self, duty_codes=codes)


# ---------------------------------------------------------------------------
# Shipped regimes
#
# These describe service contracts, not organisations. Any operation running
# to a given contract sizes the same way, whatever sector it sits in.
# ---------------------------------------------------------------------------

COMMERCIAL_INBOUND = ServiceProfile(
    name="Commercial inbound",
    target_sl=0.80,
    target_time_sec=20.0,
    max_occupancy=0.85,
    min_presence_floor=1,
    abandons=True,
    mean_patience_sec=90.0,
    hours="weekday daytime",
    note="The industry default. Callers wait a minute or two, then hang up.",
)

COMPLEX_SUPPORT = ServiceProfile(
    name="Complex support",
    target_sl=0.85,
    target_time_sec=30.0,
    max_occupancy=0.75,
    min_presence_floor=2,
    abandons=True,
    mean_patience_sec=240.0,
    hours="extended weekday",
    note=(
        "Longer, harder calls. The lower occupancy cap is the point: sustained "
        "high utilisation on cognitively demanding work drives attrition."
    ),
)

RAPID_RESPONSE = ServiceProfile(
    name="Rapid response intake",
    target_sl=0.95,
    target_time_sec=10.0,
    max_occupancy=0.70,
    min_presence_floor=4,
    abandons=False,
    mean_patience_sec=0.0,
    hours="24/7",
    note=(
        "Urgent intake. Callers do not abandon - they hold, and redial if cut "
        "off - so an abandonment term would understate required capacity. "
        "Sized on service level and floor alone."
    ),
)

BLENDED_BACK_OFFICE = ServiceProfile(
    name="Blended back office",
    target_sl=0.70,
    target_time_sec=60.0,
    max_occupancy=0.88,
    min_presence_floor=1,
    abandons=True,
    mean_patience_sec=300.0,
    hours="weekday daytime",
    note="Queue work fills the gaps between casework. Tolerant callers, high utilisation.",
)

PROFILES: dict[str, ServiceProfile] = {
    "commercial": COMMERCIAL_INBOUND,
    "complex": COMPLEX_SUPPORT,
    "rapid": RAPID_RESPONSE,
    "blended": BLENDED_BACK_OFFICE,
}


def get_profile(key: str) -> ServiceProfile:
    try:
        return PROFILES[key.strip().lower()]
    except KeyError:
        raise KeyError(f"unknown profile {key!r}; choose from {sorted(PROFILES)}") from None
