# HIPAA Security Risk Assessment — Revision 2 (post-remediation + Langfuse)

**HIPAA track, re-assessment.** The [first risk assessment](02-security-risk-assessment.md) scored the
un-remediated baseline (commit `5ff6708`). Since then the Phase 0/2 hardening was committed **and** a
Langfuse tracing integration was added — a new PHI egress path the original register never covered.
This revision re-scores against the **current committed code** and satisfies the §164.308(a)(1)(ii)(A)
requirement to update the analysis on a material change (NIST SP 800-30 method as before).

| | |
|---|---|
| System | DBC Care appointment assistant (web chat/voice, Telegram, Discord, MCP, CLI, Admin, **Langfuse tracing**) |
| Code baseline | commit `31c4f6a` — *"Add HIPAA hardening: auth, audit log, PHI redaction, rate limiting, tracing, and go-live checks"* |
| Assessment date | 2026-08-24 |
| Assessor | Claude (automated, code-level) — requires Security Officer review/sign-off |
| Prior assessment | [02-security-risk-assessment.md](02-security-risk-assessment.md) (7 Critical · 13 High · 4 Medium · 1 Low) |
| Current runtime posture | `HIPAA_MODE=1`, `SYNTHETIC_DATA_ONLY=1` (development with synthetic data; non-BAA vendors permitted by design) |

> **How to read this.** Ratings are **residual** — the risk that remains *with the current controls in
> place*, evaluated for a real-PHI production deployment (`SYNTHETIC_DATA_ONLY` unset). Where a control
> only takes effect once operators complete go-live steps, that is stated. The synthetic-dev posture is
> not itself a finding — it is an intended mode; R2-13 covers the one risk it introduces.

## 1. What changed since Revision 1

Verified in code (file:line in §4). The first assessment's 25 findings are re-mapped in §5.

- **Closed in code:** unauthenticated PHI endpoints (R-01/09), MCP exposure & demo OAuth (R-08/20),
  admin auth (R-06/15), CORS (R-01), PHI in logs (R-12), retention (R-13), audit trail (R-14),
  backups (R-16), file permissions (R-17), identity disclosure & ownership (R-10), vendor allow-list
  (R-03/04/05 — enforced in code; contracts still required), `Records` file (R-02, removed from tree).
- **New since R1:** Langfuse tracing (`tracing.py`) — conversation text + tool args/results traced to a
  self-hosted server. Assessed as **R2-01**.
- **New finding surfaced during this pass:** the default LLM callback prints full prompts/responses to
  stdout (**R2-02**).

## 2. Scope changes

Added to scope: `tracing.py` and the Langfuse server (self-hosted `http://localhost:53110` per `.env`);
the eight security modules (`security_config`, `admin_auth`, `audit_log`, `phi_redact`, `rate_limit`,
`retention`, `backup`, `go_live_check`). Everything else as Revision 1.

## 3. Residual risk register (current code)

Scoring as before: Likelihood × Impact, each 1–3. **Critical 9 · High 6 · Medium 3–4 · Low 1–2.**

| ID | Threat event | Vulnerability (evidence) | Existing controls | L | I | **Risk** | HIPAA ref |
|---|---|---|---|---|---|---|---|
| R2-01 | Patient names in conversation reach Langfuse in the clear | Trace masking (`tracing.py:53-65`) reuses `phi_redact`, which masks phones/emails by regex and identifier **keys** but **not a name typed in free-text** prompt/reply (`phi_redact.py:19-56`); default target is self-hosted but `LANGFUSE_BASE_URL` is plain HTTP and could be repointed | keys-gated + `HIPAA_MODE` mask + self-hosted default + 1-day retention (`retention.py:239-317`) | 2 | 2 | **4 Medium** | §164.312(e)(1), §164.502(b) |
| R2-02 | Full prompts + responses (all PHI) printed to stdout/console on every LLM call | `_init_model` default `callbacks=LLMInspectHandler` (`Model.py:135`) and every agent calls it without overriding (`hospital_agents.py:60,88,119,239,263,287`); handler `print()`s prompt and response (`Model.py:12-22`), bypassing the log redactor, regardless of `HIPAA_MODE` | none (stdout not redacted) | 3 | 2 | **6 High** | §164.312(b), §164.502(b) |
| R2-03 | Self-hosted Langfuse server itself is unsecured | `.env` base URL `http://localhost:53110` — no TLS, and nothing in-repo proves the instance authenticates access or encrypts traces at rest (out-of-repo infra) | project keys only; localhost default | 2 | 2 | **4 Medium** | §164.312(a)(1),(e); §164.308(b)(1) |
| R2-04 | Real PHI enters while `SYNTHETIC_DATA_ONLY=1` still set | The flag relaxes the vendor allow-list and PHI-channel guards at runtime (`security_config.py:45-53,173-176`); nothing at runtime detects real data — only the manual `go_live_check.py` gate does | go-live preflight (operator-run), start-up warning | 2 | 3 | **6 High** | §164.308(a)(1)(ii)(A) |
| R2-05 | Wrong-person disclosure via knowledge-only identity proofing | Identity = phone + full name the caller *states*; no OTP/possession factor (`tool_patient.py`, `tool_prescriptions.py:18-51`) | phone+name required, no record disclosed pre-verify, per-session rate-limit + lockout (`tool_patient.py:11-13,101-112`) | 2 | 2 | **4 Medium** | §164.312(d), §164.514(h) |
| R2-06 | PHI intercepted in transit | No TLS in repo; `cookie_secure`/HSTS/`ALLOWED_ORIGINS` assume an HTTPS reverse proxy supplied at deploy (`security_config.py:222`, `bot.py` headers) | Secure-cookie + HSTS *when* fronted by TLS; go-live requires it | 2 | 3 | **6 High** (Low once TLS terminated) | §164.312(e)(1),(e)(2)(ii) |
| R2-07 | Compromise of one MCP client key grants full tool access; no per-client revocation | Single shared static `MCP_API_KEY` (`mcp_server.py:371`), no rotation/scoping | bearer required, constant-time compare, WS refused, tunnel guard on, tools still enforce identity + audit | 1 | 2 | **2 Low** | §164.312(d) |
| R2-08 | Rate-limits / lockouts bypassed across workers or reset on restart | `RateLimiter`/`Lockout` are per-process in-memory (`rate_limit.py:1-4`) | effective for single-process deploy; documented to swap to Redis | 2 | 1 | **2 Low** | §164.308(a)(5)(ii)(C) |
| R2-09 | Backups written in clear text | `create_backup` writes plaintext when `BACKUP_ENC_KEY` unset (`backup.py:84-86`) | go-live FAILs without the key; Fernet when set | 1 | 3 | **3 Medium** | §164.312(a)(2)(iv), §164.310(d) |
| R2-10 | Vulnerable dependency exploited | app-facing `requirements.txt` uses `>=` floors; `requirements.lock`/`uv.lock` exist but the install file is unpinned; no scheduled `pip-audit` | lockfiles present | 2 | 2 | **4 Medium** | §164.308(a)(5)(ii)(B),(a)(8) |
| R2-11 | Doctor-schedule / directory reads not attributable | `/api/doctors`, `/api/departments`, `/api/doctors/{id}/schedule`, `/api/response-timings` are auth-gated but not audited (`admin_routes.py:278-319,495-501`) | session required; endpoints return no patient PHI | 1 | 1 | **1 Low** | §164.312(b) |
| R2-12 | Prompt-injection drives tool misuse | Patient text flows to the LLM which can call write tools | tools enforce phone+name + ownership + audit; MCP tools need bearer | 2 | 2 | **4 Medium** | §164.312(c)(1) |
| R2-13 | Vendors/policies/BAAs not yet in place | No executed BAAs; policies, Security Officer, training, IR plan pending (administrative track) | allow-list + go-live gate block the config; contracts outstanding | 2 | 3 | **6 High** | §164.308(b)(1),(a)(1-6), §164.316 |

**Summary:** 0 Critical · **4 High** (R2-02, R2-04, R2-06, R2-13) · 5 Medium · 3 Low.
Down from 7 Critical / 13 High at Revision 1. The remaining High items are: one code bug (R2-02, fixed & tested in this pass — see §6), one operator-discipline risk (R2-04), and two that need infrastructure/contracts
rather than code (R2-06 TLS, R2-13 BAAs & policies).

## 4. Control state — evidence (what is IMPLEMENTED now)

Condensed; every claim carries file:line.

**Access control** — admin sessions `_require_auth` 401/403 with roles (`admin_routes.py:71-78`), writes need `admin` role, HTML pages redirect to login (`admin_routes.py:90-105`); `/api/call-logs/recent` admin-only + audited, `/event`/`/end` need the per-call HMAC token (`log_routes.py:38-44,91-147`); MCP bearer + constant-time + WS-refused + demo-OAuth-refused-in-HIPAA, no `/tool-calls` route (`mcp_server.py:180-184,323-384`).

**Identity** — phone+name required, no on-file record disclosed before verification, name-mismatch gives no hint, `patient_id` alone rejected, appointment ownership enforced, per-session rate-limit+lockout (`db_patients.py:258-335`, `tool_patient.py:62-112`, `tool_booking.py:234-287`, `tool_prescriptions.py:18-51`).

**Vendors** — allow-list `{openai,groq}` BAA-capable, `{deepseek,glm}` never-PHI, `_init_model` raises on block, OpenAI `store=false`, Deepgram `mip_opt_out` (`security_config.py:150-169`, `Model.py:101`, `bot.py`); Telegram/Discord hand-off in prod (`telegram_bot.py:153`, `discord_bot.py:195`).

**Audit** — SHA-256 hash chain, `flock`+fsync per append, `verify_chain`, subjects are ids/`phone_hash` only; events span admin login/read/write/delete, every tool call (incl. `name_mismatch`/`not_owner`), MCP auth-deny, retention, backup (`audit_log.py:39-170` + call sites).

**PHI minimisation** — graph state → role/count summaries, tool logs mask phone / omit names, voice logs lengths, `tool_call.json` redacted, raw LLM dumps gated (`hospital_graph.py:63-205`, `tool_*.py`, `chat_log_tools.py`, `security_config.py:227-237`); global stdlib+loguru redaction filter (`phi_redact.py:123-146`).

**Retention/backup/perms** — daily purge of all PHI stores + Langfuse traces, audit retained 6y (`retention.py:39-49,239-317`); Fernet backups + integrity check (`backup.py:64-168`); umask 077 + 0600/0700 (`security_config.py:258-287`).

**Langfuse** — keys-gated + `LANGFUSE_TRACING_ENABLED`, `HIPAA_MODE` mask hook, self-hosted default, SDK failure = no-trace, 1-day retention (`tracing.py:44-103`, `agent_turn.py:480-490`, `retention.py:239-317`).

## 5. Mapping to Revision 1 findings

| R1 | Was | Now | Note |
|---|---|---|---|
| R-01, R-06, R-08, R-09, R-20 | Critical | **Low** | auth added everywhere; see §4 |
| R-03, R-04, R-05 | Critical | **Low in code / R2-13 for contracts** | allow-list + hand-off enforced; BAAs outstanding |
| R-02 | High | **Low** | file removed; history purge = operator step |
| R-07 | High | **R2-06 High** | Secure-cookie ready; TLS still external |
| R-10 | High | **R2-05 Medium** | disclosure closed; knowledge-only proofing remains |
| R-11 | High | **R2-09 Medium** | encrypted backups + `DATA_DIR`; disk-encryption is deploy |
| R-12 | High | **Low** (+ R2-02 new) | logs redacted; stdout callback is the new leak |
| R-13, R-14, R-16, R-17 | High/Med | **Low** | retention, audit, backups, perms implemented |
| R-15 | High | **Low** | per-user accounts + roles |
| R-18 | Medium | **R2-10 Medium** | lockfiles exist; install file unpinned |
| R-22, R-23 | High | **R2-13 High** | administrative track still open |
| — | — | **R2-01, R2-02, R2-03, R2-04, R2-07, R2-08, R2-11, R2-12** | new/re-expressed for current code |

## 6. Remediation for the new findings

| # | Finding | Action | Status |
|---|---|---|---|
| 6.1 | **R2-02** stdout prompt/response leak | `LLMInspectHandler` is now opt-in via `LLM_INSPECT_STDOUT=1` **and** refused whenever real PHI can be in play (`Model.inspect_handler()`); `_init_model` defaults to `callbacks=None` (`Model.py`) | **fixed & tested** (`tests/test_security.py::test_llm_inspect_handler_never_dumps_with_real_phi`) |
| 6.2 | R2-01 free-text names in traces | Options, in order: (a) keep self-hosted + `RETAIN_LANGFUSE_DAYS=1` (current); (b) add NER-based name masking to `phi_redact` (Presidio/spaCy) so `mask_for_trace` covers free text; (c) trace metadata only (no message content). Document the residual-risk acceptance until (b). | planned |
| 6.3 | R2-03 Langfuse server | Put the instance behind TLS + auth on an encrypted volume, or a BAA-covered host; never point `LANGFUSE_BASE_URL` off-host with real PHI; `go_live_check` should assert it is not a remote plain-HTTP URL | planned (add go-live check) |
| 6.4 | R2-04 synthetic→real transition | `go_live_check` already gates it; add a start-up refusal if `SYNTHETIC_DATA_ONLY=1` **and** the DB contains rows not marked synthetic (defence in depth) | planned |
| 6.5 | R2-09 plaintext backups | Consider refusing unencrypted backups when `HIPAA_MODE=1` and not synthetic | planned |
| 6.6 | R2-10 deps | Pin `requirements.txt` from `requirements.lock`; add `pip-audit` to CI | planned |

## 7. Residual after §6 fixes

| Finding | Now | After |
|---|---|---|
| R2-02 stdout leak | High | **Low** (fixed) |
| R2-01 trace free-text names | Medium | Low with NER masking; Medium if accepted with self-host + 1-day retention |
| R2-03 Langfuse server | Medium | Low behind TLS/auth/BAA |
| R2-04 synthetic→real | High | Low with the start-up guard |
| R2-06 TLS | High | Low once terminated at deploy |
| R2-13 BAAs/policies | High | Low after the administrative track (next step) |
| others | Low–Medium | unchanged / Low |

## 8. Acceptance & next step

| | |
|---|---|
| Risks accepted as-is | none yet — Security Officer to review R2-01/R2-05 acceptance rationale |
| Next review | after §6 fixes land, then before real-data go-live, then annually |
| Sign-off | ______________________ (Security Officer) date ________ |

**Next step — administrative & contractual track** (closes R2-13): execute BAAs (OpenAI ZDR, Deepgram,
Daily/hosting), designate Security & Privacy Officers, write the policy set (access control, incident
response & breach notification, retention/disposal, workstation, sanctions, contingency), workforce
training, and — because a Langfuse instance and the workstation now hold ePHI — add both to the device/
media inventory.
