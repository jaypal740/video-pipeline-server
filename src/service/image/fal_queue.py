import logging
import time

import fal_client
import httpx

logger = logging.getLogger(__name__)

# A clip takes one to several minutes in fal's queue plus generation.
TIMEOUT_S = 900
POLL_S = 3.0


class FalQueue:
    """What fal's queued clip models (video and lip-sync) have in common: one client, polling a submitted request
    by its id, downloading the clip and uploading a local file."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client: fal_client.SyncClient | None = None

    def _wait(self, endpoint: str, request_id: str) -> dict:
        """Poll a submitted request until it completes; the id is known, so a timeout can be resumed."""
        client = self._get_client()
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            status = client.status(endpoint, request_id)
            if isinstance(status, fal_client.Completed):
                return client.result(endpoint, request_id)
            if time.monotonic() > deadline:
                raise fal_client.FalClientTimeoutError(TIMEOUT_S, request_id)
            time.sleep(POLL_S)

    @staticmethod
    def _download(url: str) -> bytes:
        reply = httpx.get(url, timeout=120, follow_redirects=True)
        reply.raise_for_status()
        if not reply.headers.get("content-type", "").startswith("video/"):
            raise RuntimeError(f"fal returned {reply.headers.get('content-type')!r} instead of a video from {url}")
        return reply.content

    def upload(self, data: bytes, content_type: str) -> str:
        """A short-lived fal URL for an image or an audio cut that lives in local storage; a failed upload is
        tried once more, since fal's storage sometimes refuses one at random (2026-11-02: "Invalid storage type")."""
        try:
            return self._get_client().upload(data, content_type)
        except (fal_client.FalClientError, httpx.HTTPError) as exc:
            logger.warning("fal upload failed (%s); trying once more", str(exc)[:160])
            return self._get_client().upload(data, content_type)

    def _get_client(self) -> fal_client.SyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("FAL_KEY is not set")
            self._client = fal_client.SyncClient(key=self._api_key, default_timeout=TIMEOUT_S)
        return self._client
