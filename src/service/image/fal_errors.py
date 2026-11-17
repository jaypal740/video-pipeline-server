import fal_client

# fal's error classes (fal.ai/docs/errors): a 5xx or a timeout is transient and the same request may be sent
# again; a 422 (content policy, no media generated, bad input) never succeeds unchanged, so the only way on
# is a different model.


def transient(exc: fal_client.FalClientError) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(exc, fal_client.FalClientTimeoutError) or (status is not None and status >= 500)


def refused(exc: Exception) -> bool:
    """A content checker would not process the request; the same picture and words are refused every time."""
    return "content_policy_violation" in str(exc)
