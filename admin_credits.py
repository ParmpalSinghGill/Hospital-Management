"""Live credit / balance reports for configured API keys.

Deepgram: Manage API ``GET /v1/projects/{id}/balances`` (needs ``billing:read`` —
Owner/Admin key, or set ``DEEPGRAM_MANAGE_API_KEY``).
GLM / Zhipu: monitor quota ``/api/monitor/usage/quota/limit``.
OpenAI / Groq: connectivity when no public balance API exists.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def _mask(key: str | None) -> str:
    if not key:
        return "not set"
    k = key.strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}…{k[-4:]}"


def _base_report(provider: str, role: str, key: str | None, *, unit: str = "") -> dict[str, Any]:
    return {
        "provider": provider,
        "role": role,
        "key_status": "configured" if key else "missing",
        "key_hint": _mask(key),
        "credit": None,
        "unit": unit,
        "detail": "",
        "ok": False,
    }


def _sum_deepgram_balances(payload: dict[str, Any]) -> tuple[float | None, str, int]:
    balances = payload.get("balances") or []
    total = 0.0
    units = "USD"
    found = 0
    for b in balances:
        if not isinstance(b, dict):
            continue
        raw = b.get("amount")
        if raw is None:
            continue
        try:
            total += float(raw)
            found += 1
        except (TypeError, ValueError):
            continue
        u = b.get("units") or b.get("unit")
        if isinstance(u, str) and u.strip():
            units = u.strip().upper()
    if found == 0:
        return None, units, 0
    return round(total, 4), units, found


async def _deepgram_balances_http(
    client: httpx.AsyncClient, key: str, project_id: str
) -> tuple[int, dict[str, Any] | None, str]:
    headers = {"Authorization": f"Token {key}"}
    r = await client.get(
        f"https://api.deepgram.com/v1/projects/{project_id}/balances",
        headers=headers,
    )
    if r.status_code >= 400:
        return r.status_code, None, (r.text or "")[:220]
    try:
        return r.status_code, r.json(), ""
    except Exception:
        return r.status_code, None, "Invalid JSON from balances endpoint"


async def _deepgram_report() -> dict[str, Any]:
    stt_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    manage_key = os.getenv("DEEPGRAM_MANAGE_API_KEY", "").strip() or stt_key
    report = _base_report("Deepgram", "STT + TTS (Cascade)", stt_key or manage_key, unit="USD")
    if not manage_key:
        report["detail"] = "DEEPGRAM_API_KEY not set"
        return report

    using_manage = bool(os.getenv("DEEPGRAM_MANAGE_API_KEY", "").strip())
    headers = {"Authorization": f"Token {manage_key}"}
    last_err = ""
    try:
        # Prefer official SDK (same Manage balances API the console/docs use)
        try:
            from deepgram import DeepgramClient

            dg = DeepgramClient(api_key=manage_key)
            proj_resp = dg.manage.v1.projects.list()
            # SDK versions differ slightly in response shape
            if hasattr(proj_resp, "projects"):
                project_list = list(proj_resp.projects or [])
            elif isinstance(proj_resp, dict):
                project_list = list(proj_resp.get("projects") or [])
            else:
                project_list = []

            for project in project_list:
                if hasattr(project, "model_dump"):
                    project = project.model_dump()
                elif not isinstance(project, dict):
                    project = {
                        "project_id": getattr(project, "project_id", None),
                        "name": getattr(project, "name", None),
                    }
                project_id = project.get("project_id") or project.get("projectId")
                name = project.get("name") or project_id or "project"
                if not project_id:
                    continue
                try:
                    resp = dg.manage.v1.projects.billing.balances.list(str(project_id))
                    payload = (
                        resp.model_dump()
                        if hasattr(resp, "model_dump")
                        else (resp if isinstance(resp, dict) else {})
                    )
                    if hasattr(resp, "balances") and not payload.get("balances"):
                        payload = {
                            "balances": [
                                b.model_dump() if hasattr(b, "model_dump") else dict(b)
                                for b in (resp.balances or [])
                            ]
                        }
                    total, units, n = _sum_deepgram_balances(payload)
                    if total is None:
                        last_err = f"{name}: empty balances"
                        continue
                    report["credit"] = total
                    report["unit"] = units
                    report["ok"] = True
                    extra = " · manage key" if using_manage else ""
                    report["detail"] = f"Project: {name} · {n} balance row(s){extra}"
                    return report
                except Exception as sdk_err:
                    last_err = f"{name}: {sdk_err}"
                    continue
        except Exception as sdk_err:
            last_err = str(sdk_err)

        async with httpx.AsyncClient(timeout=25.0) as client:
            projects_r = await client.get(
                "https://api.deepgram.com/v1/projects", headers=headers
            )
            if projects_r.status_code >= 400:
                report["detail"] = (
                    f"Projects HTTP {projects_r.status_code}: {(projects_r.text or '')[:160]}"
                    + (f" · prior: {last_err}" if last_err else "")
                )
                return report
            project_list = (projects_r.json() or {}).get("projects") or []
            if not project_list:
                report["detail"] = "No Deepgram projects found for this key"
                report["ok"] = True
                return report

            for project in project_list:
                project_id = project.get("project_id") or project.get("projectId")
                name = project.get("name") or project_id or "project"
                if not project_id:
                    continue
                status, body, err = await _deepgram_balances_http(
                    client, manage_key, str(project_id)
                )
                if status >= 400 or body is None:
                    last_err = f"HTTP {status} on {name}" + (f": {err}" if err else "")
                    continue
                total, units, n = _sum_deepgram_balances(body)
                if total is None:
                    last_err = f"{name}: balances returned but no amount fields"
                    continue
                report["credit"] = total
                report["unit"] = units
                report["ok"] = True
                extra = " · manage key" if using_manage else ""
                report["detail"] = f"Project: {name} · {n} balance row(s){extra}"
                return report

            report["ok"] = True  # key lists projects; billing scope likely missing
            report["detail"] = (
                (last_err or "Balances forbidden")
                + ". Key needs Owner/Admin with billing:read — "
                "create one in Deepgram console, or set DEEPGRAM_MANAGE_API_KEY."
            )
    except Exception as e:
        report["detail"] = str(e)
    return report


async def _openai_report() -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    report = _base_report("OpenAI", "Realtime voice (+ optional chat LLM)", key)
    if not key:
        report["detail"] = "OPENAI_API_KEY not set"
        return report

    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            models = await client.get("https://api.openai.com/v1/models", headers=headers)
            if models.status_code >= 400:
                report["detail"] = f"Models HTTP {models.status_code}: {models.text[:160]}"
                return report

            # Legacy billing endpoints — often 401/403 for user API keys
            for url, label in (
                (
                    "https://api.openai.com/v1/dashboard/billing/credit_grants",
                    "credit_grants",
                ),
                (
                    "https://api.openai.com/dashboard/billing/credit_grants",
                    "credit_grants_alt",
                ),
            ):
                r = await client.get(url, headers=headers)
                if r.status_code != 200:
                    continue
                body = r.json() or {}
                total = body.get("total_available")
                if total is None:
                    total = body.get("total_granted")
                if total is not None:
                    try:
                        report["credit"] = round(float(total), 4)
                        report["unit"] = "USD"
                        report["ok"] = True
                        report["detail"] = f"From {label}"
                        return report
                    except (TypeError, ValueError):
                        pass

            report["ok"] = True
            report["detail"] = (
                "Key works. OpenAI no longer exposes remaining credit on user API keys — "
                "see platform.openai.com → Billing."
            )
    except Exception as e:
        report["detail"] = str(e)
    return report


async def _groq_report() -> dict[str, Any]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    report = _base_report("Groq", "CLI / optional Cascade LLM", key)
    if not key:
        report["detail"] = "GROQ_API_KEY not set"
        return report
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
            if r.status_code >= 400:
                report["detail"] = f"Models HTTP {r.status_code}: {r.text[:160]}"
                return report
            report["ok"] = True
            report["detail"] = (
                "Key works. Groq has no public remaining-credit API — check console.groq.com."
            )
    except Exception as e:
        report["detail"] = str(e)
    return report


async def _glm_quota(client: httpx.AsyncClient, key: str, url: str) -> dict[str, Any] | None:
    """Zhipu monitor quota — Authorization is the raw API key (not Bearer)."""
    headers = {
        "Authorization": key,
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
    }
    r = await client.get(url, headers=headers)
    if r.status_code >= 400:
        # Retry with Bearer for some gateways
        r = await client.get(url, headers={**headers, "Authorization": f"Bearer {key}"})
        if r.status_code >= 400:
            return None
    try:
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # Shapes: {success, code, data:{limits}} or {data:{limits}} or {limits}
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    limits = (payload or {}).get("limits") if isinstance(payload, dict) else None
    if not limits and data.get("success") is False:
        return None
    if not isinstance(limits, list) or not limits:
        return None

    token_limit = next((x for x in limits if isinstance(x, dict) and x.get("type") == "TOKENS_LIMIT"), None)
    if not token_limit:
        token_limit = next((x for x in limits if isinstance(x, dict)), None)
    if not token_limit:
        return None

    percentage = float(token_limit.get("percentage") or 0)
    remaining_pct = max(0.0, round(100.0 - percentage, 2))
    used = token_limit.get("currentValue")
    total = token_limit.get("usage")
    level = (payload or {}).get("level") if isinstance(payload, dict) else None
    detail_parts = []
    if level:
        detail_parts.append(f"Plan: {level}")
    if used is not None and total is not None:
        detail_parts.append(f"Used {used} / {total}")
    reset = token_limit.get("nextResetTime")
    if reset:
        detail_parts.append(f"Reset ts: {reset}")
    return {
        "credit": remaining_pct,
        "unit": "% remaining",
        "detail": " · ".join(detail_parts) or "From Zhipu quota monitor",
        "ok": remaining_pct > 0,
    }


async def _glm_report() -> dict[str, Any]:
    key = os.getenv("GLM_API_KEY", "").strip()
    base = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    report = _base_report("Zhipu GLM", "Optional Cascade / CLI LLM", key, unit="% remaining")
    if not key:
        report["detail"] = "GLM_API_KEY not set"
        return report

    quota_urls = [
        "https://open.bigmodel.cn/api/monitor/usage/quota/limit",
        "https://bigmodel.cn/api/monitor/usage/quota/limit",
        "https://api.z.ai/api/monitor/usage/quota/limit",
    ]
    # Prefer host matching GLM_BASE_URL when it points at z.ai
    if "z.ai" in base:
        quota_urls = [
            "https://api.z.ai/api/monitor/usage/quota/limit",
            "https://open.bigmodel.cn/api/monitor/usage/quota/limit",
            "https://bigmodel.cn/api/monitor/usage/quota/limit",
        ]

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            for url in quota_urls:
                parsed = await _glm_quota(client, key, url)
                if parsed:
                    report.update(parsed)
                    report["detail"] = f"{parsed['detail']} · {url.split('/')[2]}"
                    return report

            # Fallback: models connectivity
            headers = {"Authorization": f"Bearer {key}"}
            r = await client.get(f"{base}/models", headers=headers)
            if r.status_code >= 400:
                report["detail"] = (
                    f"Quota + models failed (models HTTP {r.status_code}). "
                    "Check GLM_API_KEY / BigModel console."
                )
                report["ok"] = r.status_code in (404, 405)
                return report
            report["ok"] = True
            report["detail"] = (
                "Key works, but quota monitor did not return limits. "
                "Check open.bigmodel.cn finance overview."
            )
    except Exception as e:
        report["detail"] = str(e)
    return report


async def _deepseek_report() -> dict[str, Any]:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    report = _base_report("DeepSeek", "Cascade LLM (default)", key)
    if not key:
        report["detail"] = "DEEPSEEK_API_KEY not set"
        return report

    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # Official balance endpoint
            bal = await client.get(f"{base}/user/balance", headers=headers)
            if bal.status_code == 200:
                try:
                    data = bal.json() or {}
                except Exception:
                    data = {}
                infos = data.get("balance_infos") or []
                parts = []
                primary = None
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    total = info.get("total_balance")
                    currency = (info.get("currency") or "").strip()
                    label = f"{total} {currency}".strip() if total is not None else ""
                    if label:
                        parts.append(label)
                    if primary is None and total is not None:
                        try:
                            primary = float(total)
                        except (TypeError, ValueError):
                            primary = None
                if primary is not None:
                    report["credit"] = primary
                    report["unit"] = (infos[0].get("currency") if infos else "") or "USD"
                report["ok"] = bool(data.get("is_available", True))
                report["detail"] = (
                    ", ".join(parts)
                    if parts
                    else ("Available" if report["ok"] else "No usable credit")
                )
                return report

            # Fallback: models list proves the key works
            models = await client.get(f"{base}/models", headers=headers)
            if models.status_code >= 400:
                # Some DeepSeek bases want /v1 prefix
                models = await client.get(f"{base}/v1/models", headers=headers)
            if models.status_code >= 400:
                report["detail"] = (
                    f"Balance HTTP {bal.status_code}; models HTTP {models.status_code}"
                )
                return report
            report["ok"] = True
            report["detail"] = (
                f"Key works (balance HTTP {bal.status_code}). "
                "Check platform.deepseek.com for remaining credit."
            )
    except Exception as e:
        report["detail"] = str(e)
    return report


async def collect_credit_reports() -> list[dict[str, Any]]:
    return [
        await _deepgram_report(),
        await _deepseek_report(),
        await _glm_report(),
        await _openai_report(),
        await _groq_report(),
    ]
