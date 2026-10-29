from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config


class Storage(Protocol):
    """Where files live. Keys are paths like runs/<run>/keyframes/s04.png or hosts/3-5/front.png."""

    name: str

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def url(self, key: str, seconds: int) -> str | None:
        """A link an outside service (fal) can fetch for `seconds`; None when the storage has no such link
        and the caller must upload the bytes itself."""
        ...


class LocalStorage:
    """Files under a folder; used by tests and local development."""

    name = "local"

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return (self._root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self._root / key).is_file()

    def url(self, key: str, seconds: int) -> str | None:
        return None

    def path(self, key: str) -> Path:
        return self._root / key


class R2Storage:
    """Cloudflare R2 through its S3 API. Presigned links are what fal downloads references from."""

    name = "r2"

    def __init__(self, account_id: str, access_key_id: str, secret_access_key: str, bucket: str) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3", endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name="auto",
            config=Config(signature_version="s3v4"))

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except self._client.exceptions.ClientError:
            return False
        return True

    def url(self, key: str, seconds: int) -> str | None:
        return self._client.generate_presigned_url("get_object", Params={"Bucket": self._bucket, "Key": key},
                                                   ExpiresIn=seconds)
