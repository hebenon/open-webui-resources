"""
Memory Enhancement Tool for Open WebUI Knowledge Bases

- Exposes a Tools class (public surface) with configuration Valves and the following async methods:
  - recall_memories(__user__, __event_emitter__, kb_id=None)
  - add_memory(items=None, __user__=None, __event_emitter__=None, kb_id=None)
  - update_memory(updates=None, __user__=None, __event_emitter__=None, kb_id=None)
  - delete_memory(ids_or_names=None, __user__=None, __event_emitter__=None, kb_id=None)

- Each method returns a JSON string (utf-8) describing success/failure and includes ids where available.
- Stores memories inside knowledge.data["memory_blocks"] while preserving unrelated metadata.
- Defensive: merges existing data when persisting so other knowledge attributes remain intact.
- Valves:
    - USE_MEMORY: enable/disable usage
    - DEBUG: enable debug logging
    - DEFAULT_KB_ID: fallback KB id
"""

import inspect
import json
import logging
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

# Attempt to import the Knowledge API surface provided by Open WebUI
try:
    from open_webui.models.knowledge import Knowledges  # type: ignore

    KNOWLEDGE_AVAILABLE = True
except Exception:
    KNOWLEDGE_AVAILABLE = False
    Knowledges = None  # type: ignore

# Attempt to import the Files API so we can create Markdown-backed records.
try:
    from open_webui.models.files import FileForm, Files  # type: ignore

    FILES_AVAILABLE = True
except Exception:
    FILES_AVAILABLE = False
    FileForm = None  # type: ignore
    Files = None  # type: ignore

log = logging.getLogger(__name__)


MEMORY_DATA_KEY = "memory_blocks"


class EventEmitter:
    """
    Thin wrapper to emit status events via a provided callable.

    Usage:
        emitter = EventEmitter(event_emitter_callable)
        await emitter.emit(description="Doing X", status="in_progress", done=False)
    """

    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None):
        self.event_emitter = event_emitter

    async def emit(
        self,
        description: str = "Unknown state",
        status: str = "in_progress",
        done: bool = False,
        **extra,
    ):
        """
        Emit a status message through the provided event_emitter if present.

        Adds `type: status` plus `data` payload.
        """
        if self.event_emitter:
            payload = {
                "type": "status",
                "data": {
                    "status": status,
                    "description": description,
                    "done": done,
                },
            }
            if extra:
                payload["data"].update(extra)
            await self.event_emitter(payload)


class Tools:
    """
    Memory tool public surface.

    Valves (configuration) are available at `tools.valves`.

    Methods (async):
      - recall_memories(__user__, __event_emitter__, kb_id=None) -> JSON str
      - add_memory(items=None, __user__=None, __event_emitter__=None, kb_id=None) -> JSON str
      - update_memory(updates=None, __user__=None, __event_emitter__=None, kb_id=None) -> JSON str
      - delete_memory(ids_or_names=None, __user__=None, __event_emitter__=None, kb_id=None) -> JSON str

    Notes on formats:
    - add_memory expects `items`: List[{"name": str, "content": str, "tags": Optional[List[str]]}]
      On success returns JSON `{"added": [...], "failed": [...], "message": "...", "kb_id": "<kb>"}` where each added item includes `id` (if available), `name`, `kb_id`, `created_at` (when available).
    - recall_memories returns JSON `{"blocks": [{"id","name","content","created_at"}, ...], "kb_id": "<kb>"}`.
    - update_memory accepts `updates`: List[{"id": "<id>", "content": "..."}] or List[{"name": "<name>", "content": "..."}]
    - delete_memory accepts `ids_or_names`: List[str] (ids or names)
    """

    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable memory usage."
        )
        DEBUG: bool = Field(
            default=True, description="Enable or disable debug mode (controls logging)."
        )
        DEFAULT_KB_ID: Optional[str] = Field(
            default=None,
            description="Default Knowledge Base ID. Used if no kb_id provided.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # -------------------------
    # Helper utilities (instance-level)
    # -------------------------
    @staticmethod
    def _get_knowledge_api():
        if not KNOWLEDGE_AVAILABLE or Knowledges is None:
            return None

        api = Knowledges
        try:
            if inspect.isclass(api):
                api = api()
        except Exception:
            api = Knowledges

        return api

    @staticmethod
    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    async def _get_knowledge_record(self, kb_id: str):
        api = self._get_knowledge_api()
        if not api:
            return None

        getter = getattr(api, "get_knowledge_by_id", None)
        if not callable(getter):
            return None

        record = await self._maybe_await(getter(kb_id))
        return record

    async def _save_knowledge_data(self, kb_id: str, data: dict):
        api = self._get_knowledge_api()
        if not api:
            raise RuntimeError("Knowledges API unavailable")

        updater = getattr(api, "update_knowledge_data_by_id", None)
        if not callable(updater):
            raise RuntimeError("update_knowledge_data_by_id not available")

        updated = await self._maybe_await(updater(kb_id, data))
        return updated

    @staticmethod
    def _get_files_api():
        if not FILES_AVAILABLE or Files is None:
            return None

        api = Files
        try:
            if inspect.isclass(api):
                api = api()
        except Exception:
            api = Files

        return api

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9-_ ]", "", name or "memory")
        sanitized = re.sub(r"\s+", "-", sanitized.strip())
        return sanitized.lower() or "memory"

    @staticmethod
    def _render_markdown_payload(
        block: dict,
        include_content: bool = True,
    ) -> str:
        """Render memory metadata plus content as Markdown with a simple front matter."""

        lines = ["---"]
        lines.append(f"id: {block.get('id')}")
        name_value = (block.get('name', '') or '').replace("\n", " ")
        lines.append(f"name: {name_value}")
        tags = block.get("tags") or []
        if isinstance(tags, list):
            tag_list = [str(tag) for tag in tags if tag is not None]
            lines.append(
                "tags: [" + ", ".join(f"'{tag}'" for tag in tag_list) + "]"
            )
        created_at = block.get("created_at")
        updated_at = block.get("updated_at")
        if created_at:
            lines.append(f"created_at: {created_at}")
        if updated_at:
            lines.append(f"updated_at: {updated_at}")
        file_id = block.get("file_id")
        if file_id:
            lines.append(f"file_id: {file_id}")
        lines.append("---")
        if include_content:
            lines.append("")
            content = block.get("content") or ""
            lines.append(str(content))
        return "\n".join(lines)

    def _build_memory_file_payload(
        self,
        block: dict,
        kb_id: str,
        user_id: Optional[str],
    ) -> Optional[Any]:
        files_api = self._get_files_api()
        if not files_api or FileForm is None or not user_id:
            return None

        markdown = self._render_markdown_payload(block)
        filename = f"{self._sanitize_filename(block.get('name', 'memory'))}-{block['id']}.md"
        meta = {
            "name": block.get("name") or filename,
            "content_type": "text/markdown",
            "size": len(markdown.encode("utf-8")),
            "memory": {
                "kb_id": kb_id,
                "memory_id": block.get("id"),
                "tags": block.get("tags"),
            },
        }
        form_data = FileForm(
            id=str(uuid.uuid4()),
            filename=filename,
            path="",
            data={"content": markdown, "status": "completed"},
            meta=meta,
        )

        try:
            file_record = files_api.insert_new_file(user_id, form_data)
            return file_record
        except Exception as exc:
            log.debug("Failed to create memory file: %s", exc)
            return None

    def _update_memory_file(
        self,
        file_id: str,
        block: dict,
        kb_id: str,
    ) -> bool:
        files_api = self._get_files_api()
        if not files_api:
            return False

        markdown = self._render_markdown_payload(block)
        try:
            files_api.update_file_data_by_id(
                file_id,
                {"content": markdown, "status": "completed"},
            )
            files_api.update_file_metadata_by_id(
                file_id,
                {
                    "name": block.get("name"),
                    "memory": {
                        "kb_id": kb_id,
                        "memory_id": block.get("id"),
                        "tags": block.get("tags"),
                    },
                },
            )
            return True
        except Exception as exc:
            log.debug("Failed to update memory file %s: %s", file_id, exc)
            return False

    def _delete_memory_file(self, file_id: str) -> bool:
        files_api = self._get_files_api()
        if not files_api:
            return False
        try:
            return bool(files_api.delete_file_by_id(file_id))
        except Exception as exc:
            log.debug("Failed to delete memory file %s: %s", file_id, exc)
            return False

    @staticmethod
    def _extract_user_id(record: Any, fallback_user: Optional[dict]) -> Optional[str]:
        if record is None:
            return fallback_user.get("id") if fallback_user else None

        candidate = None
        if isinstance(record, dict):
            candidate = record.get("user_id")
        else:
            candidate = getattr(record, "user_id", None)

        if not candidate and fallback_user:
            candidate = fallback_user.get("id")
        return candidate

    @staticmethod
    def _extract_data(record: Any) -> dict:
        if not record:
            return {}

        if isinstance(record, dict):
            raw = record.get("data") or {}
        else:
            raw = getattr(record, "data", None) or {}
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            elif hasattr(raw, "dict"):
                raw = raw.dict()

        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _coerce_blocks(data: dict) -> List[dict]:
        existing = data.get(MEMORY_DATA_KEY)
        if isinstance(existing, list):
            sanitized: List[dict] = []
            for entry in existing:
                if isinstance(entry, dict):
                    sanitized.append(entry)
            return sanitized
        return []

    @staticmethod
    def _serialize_block(block: dict) -> dict:
        allowed = {
            "id",
            "name",
            "content",
            "tags",
            "created_at",
            "updated_at",
            "file_id",
        }
        return {k: deepcopy(block.get(k)) for k in allowed}

    def _resolve_kb_id(
        self, __user__: Optional[dict], explicit_kb_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve kb id in order: explicit_kb_id, __user__['kb_id'], Valves.DEFAULT_KB_ID
        """
        if explicit_kb_id:
            return explicit_kb_id
        if __user__:
            kb = (
                __user__.get("kb_id")
                or __user__.get("knowledge_id")
                or __user__.get("knowledgeId")
            )
            if kb:
                return kb
        return self.valves.DEFAULT_KB_ID

    @staticmethod
    def _find_block_index(blocks: List[dict], identifier: str) -> Optional[int]:
        if not identifier:
            return None
        for idx, block in enumerate(blocks):
            if block.get("id") == identifier:
                return idx
        return None

    def _find_block_index_by_name(self, blocks: List[dict], name: str) -> Optional[int]:
        if not name:
            return None
        for idx, block in enumerate(blocks):
            if block.get("name") == name:
                return idx
        return None

    # -------------------------
    # Public API methods
    # -------------------------
    async def recall_memories(
        self,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        kb_id: Optional[str] = None,
    ) -> str:
        """Recall memory blocks from the Knowledge system.

        Reads data from ``Knowledges.get_knowledge_by_id(kb_id)`` and returns the
        list stored under ``knowledge.data["memory_blocks"]``.

        This implementation uses only the Knowledges API (no Memories fallback).
        """
        emitter = EventEmitter(__event_emitter__)

        if not self.valves.USE_MEMORY:
            msg = "Memory usage disabled (valve USE_MEMORY=False)."
            await emitter.emit(description=msg, status="memory_disabled", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        resolved_kb = self._resolve_kb_id(__user__, kb_id)
        if not resolved_kb:
            msg = "Knowledge base id not provided (no explicit kb_id, user.kb_id, or DEFAULT_KB_ID)."
            await emitter.emit(description=msg, status="missing_kb_id", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        await emitter.emit(
            description=f"Recalling memory blocks from knowledge base {resolved_kb}.",
            status="recall_in_progress",
            done=False,
            kb_id=resolved_kb,
        )

        if not KNOWLEDGE_AVAILABLE or Knowledges is None:
            msg = "Knowledges API not available in this runtime; cannot recall memories."
            await emitter.emit(
                description=msg, status="recall_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {"blocks": [], "message": msg, "kb_id": resolved_kb},
                ensure_ascii=False,
            )

        try:
            kb_obj = await self._get_knowledge_record(resolved_kb)
        except Exception as e:
            msg = f"Failed to retrieve knowledge {resolved_kb}: {repr(e)}"
            await emitter.emit(
                description=msg, status="recall_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {"blocks": [], "message": msg, "error": repr(e), "kb_id": resolved_kb},
                ensure_ascii=False,
            )

        if not kb_obj:
            msg = f"Knowledge base {resolved_kb} not found or empty."
            await emitter.emit(
                description=msg, status="recall_empty", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {"blocks": [], "message": msg, "kb_id": resolved_kb}, ensure_ascii=False
            )

        # --- Extract data field ---
        data_field = self._extract_data(kb_obj)
        blocks = self._coerce_blocks(data_field)
        normalized_blocks = [self._serialize_block(b) for b in blocks]

        await emitter.emit(
            description=f"Recalled {len(normalized_blocks)} memory blocks.",
            status="recall_complete",
            done=True,
            kb_id=resolved_kb,
        )

        return json.dumps(
            {
                "blocks": normalized_blocks,
                "count": len(normalized_blocks),
                "message": f"Recalled {len(normalized_blocks)} blocks.",
                "kb_id": resolved_kb,
            },
            ensure_ascii=False,
        )

    async def add_memory(
        self,
        items: Optional[List[Dict[str, Any]]] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        kb_id: Optional[str] = None,
    ) -> str:
        """Add named memory blocks into a knowledge base record."""

        emitter = EventEmitter(__event_emitter__)

        if not self.valves.USE_MEMORY:
            msg = "Memory usage disabled (valve USE_MEMORY=False)."
            await emitter.emit(description=msg, status="memory_disabled", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        if not items:
            msg = "Missing required parameter: items"
            await emitter.emit(description=msg, status="missing_items", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        resolved_kb = self._resolve_kb_id(__user__, kb_id)
        if not resolved_kb:
            msg = "Knowledge base id not provided (no explicit kb_id, user.kb_id, or DEFAULT_KB_ID)."
            await emitter.emit(description=msg, status="missing_kb_id", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        await emitter.emit(
            description=f"Adding {len(items)} entries to knowledge base {resolved_kb}.",
            status="add_in_progress",
            done=False,
            kb_id=resolved_kb,
        )

        if not KNOWLEDGE_AVAILABLE or Knowledges is None:
            msg = "Knowledges API not available in this runtime; cannot add memories."
            await emitter.emit(
                description=msg, status="add_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "added": [],
                    "failed": [{"reason": "knowledges_unavailable"}],
                    "message": msg,
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        try:
            kb_obj = await self._get_knowledge_record(resolved_kb)
        except Exception as e:
            msg = f"Failed to retrieve knowledge {resolved_kb}: {repr(e)}"
            await emitter.emit(
                description=msg, status="add_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "added": [],
                    "failed": [{"reason": "get_failed", "detail": repr(e)}],
                    "message": msg,
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        if not kb_obj:
            msg = (
                f"Knowledge {resolved_kb} not found. If you expect to create a new KB, "
                "call insert_new_knowledge first."
            )
            await emitter.emit(
                description=msg, status="add_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "added": [],
                    "failed": [{"reason": "kb_not_found"}],
                    "message": msg,
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        existing_data = self._extract_data(kb_obj)
        blocks = self._coerce_blocks(existing_data)
        existing_file_ids = list(existing_data.get("file_ids", []))

        created_entries: List[dict] = []
        failed: List[dict] = []
        now_ts = int(time.time())
        kb_owner = self._extract_user_id(kb_obj, __user__)
        newly_created_file_ids: List[str] = []

        for item in items:
            name = item.get("name")
            content = item.get("content")
            tags = item.get("tags", []) or []
            if not name or not content:
                failed.append({"item": item, "reason": "missing_name_or_content"})
                await emitter.emit(
                    description=f"Skipping invalid item (missing name/content): {item}",
                    status="add_skipped",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            block = {
                "id": str(uuid.uuid4()),
                "name": name,
                "content": content,
                "tags": tags,
                "created_at": now_ts,
                "updated_at": now_ts,
            }
            file_record = self._build_memory_file_payload(block, resolved_kb, kb_owner)

            if FILES_AVAILABLE and kb_owner and not file_record:
                failed.append(
                    {
                        "item": item,
                        "reason": "file_creation_failed",
                    }
                )
                await emitter.emit(
                    description=f"Failed to create file for memory '{name}'.",
                    status="add_failed",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            if file_record:
                file_id = getattr(file_record, "id", None) or getattr(
                    file_record, "file_id", None
                )
                if file_id:
                    block["file_id"] = file_id
                    newly_created_file_ids.append(file_id)

            blocks.append(block)
            created_entries.append(block)


        if not created_entries:
            msg = "No valid items to add."
            await emitter.emit(description=msg, status="add_skipped", done=True)
            return json.dumps(
                {
                    "added": [],
                    "failed": failed,
                    "message": msg,
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        combined_file_ids = list(dict.fromkeys(existing_file_ids + newly_created_file_ids))
        updated_data = {**existing_data, MEMORY_DATA_KEY: blocks}
        if combined_file_ids:
            updated_data["file_ids"] = combined_file_ids

        try:
            updated_kb = await self._save_knowledge_data(resolved_kb, updated_data)
        except Exception as e:
            msg = f"Failed to update knowledge data: {repr(e)}"
            await emitter.emit(
                description=msg, status="add_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "added": [],
                    "failed": failed + [{"reason": "update_failed", "detail": repr(e)}],
                    "message": msg,
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        returned_data = self._extract_data(updated_kb) if updated_kb else updated_data
        returned_blocks = self._coerce_blocks(returned_data)
        added: List[dict] = []
        for block in created_entries:
            found = next((b for b in returned_blocks if b.get("id") == block["id"]), None)
            source = found or block
            added.append(
                {
                    "id": source.get("id"),
                    "kb_id": resolved_kb,
                    "name": source.get("name"),
                    "content": source.get("content"),
                    "tags": source.get("tags", []),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at"),
                    "file_id": source.get("file_id"),
                    "method_used": {"function": "update_knowledge_data_by_id"},
                }
            )

        await emitter.emit(
            description=f"Added {len(added)} blocks.",
            status="add_complete",
            done=True,
            kb_id=resolved_kb,
        )
        return json.dumps(
            {
                "added": added,
                "failed": failed,
                "message": f"Added {len(added)} blocks.",
                "kb_id": resolved_kb,
            },
            ensure_ascii=False,
        )

    async def update_memory(
        self,
        updates: Optional[List[Dict[str, Any]]] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        kb_id: Optional[str] = None,
    ) -> str:
        """
        Update memory blocks.

        Request:
          - updates: list of update objects. Prefer {"id": "<id>", "content": "..."}. Accept {"name":"...","content":"..."} as fallback.
          - kb_id: optional explicit kb id.

        Response:
          {"results": [ {"id":"...","status":"updated" | "not_found_or_update_failed" | ... , "method_used": {...}}, ... ], "kb_id": "<kb>" }
        """
        emitter = EventEmitter(__event_emitter__)
        if not updates:
            msg = "Missing required parameter: updates"
            await emitter.emit(description=msg, status="missing_updates", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        resolved_kb = self._resolve_kb_id(__user__, kb_id)
        if not resolved_kb:
            msg = "Knowledge base id not provided"
            await emitter.emit(description=msg, status="missing_kb_id", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        await emitter.emit(
            description=f"Updating {len(updates)} blocks in {resolved_kb}",
            status="update_in_progress",
            done=False,
            kb_id=resolved_kb,
        )
        results = []

        data_changed = False
        try:
            kb_obj = await self._get_knowledge_record(resolved_kb)
        except Exception as e:
            msg = f"Failed to retrieve knowledge {resolved_kb}: {repr(e)}"
            await emitter.emit(
                description=msg, status="update_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "results": [
                        {
                            "status": "get_failed",
                            "detail": repr(e),
                        }
                    ],
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        data = self._extract_data(kb_obj)
        blocks = self._coerce_blocks(data)

        for upd in updates:
            bid = upd.get("id")
            name = upd.get("name")
            content = upd.get("content")
            tags = upd.get("tags")

            target_idx = None
            target_desc = {}

            if bid:
                target_idx = self._find_block_index(blocks, bid)
                target_desc = {"id": bid}
            elif name:
                target_idx = self._find_block_index_by_name(blocks, name)
                target_desc = {"name": name}
            else:
                results.append({"item": upd, "status": "invalid_payload"})
                await emitter.emit(
                    description=f"Invalid update payload: {upd}",
                    status="update_skipped",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            if target_idx is None:
                results.append({**target_desc, "status": "not_found"})
                await emitter.emit(
                    description=f"Item not found for update: {target_desc}",
                    status="update_failed",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            block = blocks[target_idx]
            updated_block = dict(block)
            if content is not None:
                updated_block["content"] = content
            if name and not bid:
                updated_block["name"] = name
            if tags is not None:
                updated_block["tags"] = tags
            updated_block["updated_at"] = int(time.time())

            if updated_block != block:
                blocks[target_idx] = updated_block
                data_changed = True
                file_id = updated_block.get("file_id")
                if file_id:
                    self._update_memory_file(file_id, updated_block, resolved_kb)
                results.append({**target_desc, "status": "updated"})
                await emitter.emit(
                    description=f"Updated block {target_desc}",
                    status="update_success",
                    done=False,
                    kb_id=resolved_kb,
                )
            else:
                results.append({**target_desc, "status": "no_change"})
                await emitter.emit(
                    description=f"No changes applied to {target_desc}",
                    status="update_skipped",
                    done=False,
                    kb_id=resolved_kb,
                )

        if data_changed:
            data[MEMORY_DATA_KEY] = blocks
            try:
                await self._save_knowledge_data(resolved_kb, data)
            except Exception as e:
                await emitter.emit(
                    description=f"Failed to persist updates: {repr(e)}",
                    status="update_failed",
                    done=False,
                    kb_id=resolved_kb,
                )
                results.append({"status": "persist_failed", "detail": repr(e)})

        await emitter.emit(
            description="Update pass complete",
            status="update_complete",
            done=True,
            kb_id=resolved_kb,
        )
        return json.dumps(
            {"results": results, "kb_id": resolved_kb}, ensure_ascii=False
        )

    async def delete_memory(
        self,
        ids_or_names: Optional[List[Any]] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        kb_id: Optional[str] = None,
    ) -> str:
        """
        Delete memory blocks by id or name.

        Request:
          - ids_or_names: list of ids (preferred) or names (strings)
          - kb_id: optional explicit kb id

        Response:
          {"results": [ {"id":"...","status":"deleted"}, {"name":"...","status":"not_found"} ], "kb_id": "<kb>"}
        """
        emitter = EventEmitter(__event_emitter__)
        if not ids_or_names:
            msg = "Missing required parameter: ids_or_names"
            await emitter.emit(
                description=msg, status="missing_ids_or_names", done=True
            )
            return json.dumps({"message": msg}, ensure_ascii=False)

        resolved_kb = self._resolve_kb_id(__user__, kb_id)
        if not resolved_kb:
            msg = "Knowledge base id not provided"
            await emitter.emit(description=msg, status="missing_kb_id", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        await emitter.emit(
            description=f"Deleting {len(ids_or_names)} blocks from {resolved_kb}",
            status="delete_in_progress",
            done=False,
            kb_id=resolved_kb,
        )
        results = []

        try:
            kb_obj = await self._get_knowledge_record(resolved_kb)
        except Exception as e:
            msg = f"Failed to retrieve knowledge {resolved_kb}: {repr(e)}"
            await emitter.emit(
                description=msg, status="delete_failed", done=True, kb_id=resolved_kb
            )
            return json.dumps(
                {
                    "results": [
                        {
                            "status": "get_failed",
                            "detail": repr(e),
                        }
                    ],
                    "kb_id": resolved_kb,
                },
                ensure_ascii=False,
            )

        data = self._extract_data(kb_obj)
        blocks = self._coerce_blocks(data)
        file_ids = list(data.get("file_ids", []))
        removed_file_ids: List[str] = []
        original_count = len(blocks)

        for identifier in ids_or_names:
            if not isinstance(identifier, str):
                results.append({"item": identifier, "status": "invalid_identifier"})
                await emitter.emit(
                    description=f"Invalid identifier: {identifier}",
                    status="delete_skipped",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            target_idx = self._find_block_index(blocks, identifier)
            if target_idx is not None:
                removed = blocks.pop(target_idx)
                file_id = removed.get("file_id")
                if file_id:
                    removed_file_ids.append(file_id)
                    self._delete_memory_file(file_id)
                results.append({"id": identifier, "status": "deleted"})
                await emitter.emit(
                    description=f"Deleted block id={identifier}",
                    status="delete_success",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            target_idx = self._find_block_index_by_name(blocks, identifier)
            if target_idx is not None:
                removed = blocks.pop(target_idx)
                file_id = removed.get("file_id")
                if file_id:
                    removed_file_ids.append(file_id)
                    self._delete_memory_file(file_id)
                results.append({"name": identifier, "status": "deleted", "id": removed.get("id")})
                await emitter.emit(
                    description=f"Deleted block name={identifier}",
                    status="delete_success",
                    done=False,
                    kb_id=resolved_kb,
                )
                continue

            results.append({"identifier": identifier, "status": "not_found"})
            await emitter.emit(
                description=f"Identifier not found for deletion: {identifier}",
                status="delete_failed",
                done=False,
                kb_id=resolved_kb,
            )

        if len(blocks) != original_count:
            data[MEMORY_DATA_KEY] = blocks
            if removed_file_ids:
                remaining = [fid for fid in file_ids if fid not in removed_file_ids]
                data["file_ids"] = remaining
            try:
                await self._save_knowledge_data(resolved_kb, data)
            except Exception as e:
                await emitter.emit(
                    description=f"Failed to persist deletions: {repr(e)}",
                    status="delete_failed",
                    done=False,
                    kb_id=resolved_kb,
                )
                results.append({"status": "persist_failed", "detail": repr(e)})

        await emitter.emit(
            description="Delete pass complete",
            status="delete_complete",
            done=True,
            kb_id=resolved_kb,
        )
        return json.dumps(
            {"results": results, "kb_id": resolved_kb}, ensure_ascii=False
        )

    async def inspect_knowledges_api(self) -> str:
        """
        Diagnostic helper: enumerate the callables on the Knowledges object and return their signatures.

        Response JSON:
          { "available": [ { "name": "<attr>", "is_callable": true, "is_coroutine": false, "signature": "(...)" }, ... ] }

        Use this to map the actual API surface to our expected candidate names.
        """
        # quick guard
        emitter = EventEmitter(None)
        if not KNOWLEDGE_AVAILABLE or Knowledges is None:
            msg = "Knowledges API not available in this runtime."
            await emitter.emit(description=msg, status="inspect_complete", done=True)
            return json.dumps({"message": msg}, ensure_ascii=False)

        results = []
        # If Knowledges is a class, list class attributes; if module/object, list its attributes
        for attr in sorted(dir(Knowledges)):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(Knowledges, attr)
                is_callable = callable(val)
                is_coro = inspect.iscoroutinefunction(val) or inspect.iscoroutine(val)
                sig = None
                if is_callable:
                    try:
                        sig = str(inspect.signature(val))
                    except Exception:
                        sig = "<signature unavailable>"
                results.append(
                    {
                        "name": attr,
                        "is_callable": is_callable,
                        "is_coroutine": is_coro,
                        "signature": sig,
                    }
                )
            except Exception as e:
                results.append({"name": attr, "error": repr(e)})
        await emitter.emit(
            description="Knowledges inspection complete",
            status="inspect_complete",
            done=True,
        )
        return json.dumps({"available": results}, ensure_ascii=False)

