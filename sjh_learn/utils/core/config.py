"""core 层运行配置。"""

FORCE_RWA = False

RWA_DISABLED_MESSAGE = (
    "RWA support is currently disabled by default. "
    "Set FORCE_RWA=True only if you intentionally want to use the legacy RWA path."
)


def ensure_rwa_enabled() -> None:
    if not FORCE_RWA:
        raise RuntimeError(RWA_DISABLED_MESSAGE)


__all__ = ["FORCE_RWA", "RWA_DISABLED_MESSAGE", "ensure_rwa_enabled"]
