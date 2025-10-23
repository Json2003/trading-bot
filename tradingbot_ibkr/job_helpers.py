from typing import Any, Dict, Callable
def make_progress_handler(job_id: str,
                          job_db: Any,
                          notifier: Any) -> Callable[[float], None]:
    def _handler(progress: float) -> None:
        p = 0.0
        try:
            # clamp and store
            p = max(0.0, min(1.0, float(progress)))
            job_db.update_job_status(job_id, progress=p)
        except Exception:
            pass
        try:
            # best-effort notify
            notifier.send_job_progress(job_id, p)
        except Exception:
            pass
    return _handler