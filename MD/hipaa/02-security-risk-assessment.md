# HIPAA Security Risk Assessment — DBC Care assistant

**HIPAA track, step 2.** Satisfies the *Risk Analysis* implementation specification, 45 CFR §164.308(a)(1)(ii)(A), and feeds *Risk Management*, §164.308(a)(1)(ii)(B). Method follows HHS guidance on Risk Analysis and NIST SP 800-30 Rev.1 (threat source → vulnerability → likelihood × impact → risk). Inputs: the [PHI data-flow map](01-phi-data-flow-map.md) (step 1) plus the additional checks listed in §2.

| | |
|---|---|
| System | DBC Care appointment assistant (web chat/voice, Telegram, Discord, MCP, CLI, Admin) |
| Code baseline | commit `5ff6708`, repo `ParmpalSinghGill/Hospital-Management` (**public**) |
| Assessment date | 2026-08-21 |
| Assessor | Claude (automated, code-level) — requires review and sign-off by the system owner / Security Officer |
| Status | **Initial assessment — not yet accepted.** Risk ratings are pre-remediation. |

> **Role assumption.** The assistant processes PHI on behalf of a healthcare provider (covered entity). Whoever operates it is therefore either the covered entity itself or a **business associate**, and every downstream vendor that receives PHI is a **subcontractor business associate** requiring a BAA. If the real deployment differs (e.g. a demo with only synthetic data and no real patients), the *likelihood* column does not change but the *impact* of several risks falls to Low; re-score before acceptance.

---

## 1. Scope

**In scope (ePHI systems):** every component that creates, receives, maintains, or transmits PHI — the seven stores, six entry channels, the LangGraph / Realtime processing path, the Admin surface, and the ten external recipients catalogued in step 1; the developer workstation that currently hosts the data; the GitHub repository; the Docker / Pipecat Cloud deployment path.

**Out of scope:** the covered entity's own EHR, billing, and physical facilities; end-user devices (patients' phones); vendor-internal controls (assessed only through contractual status).

**Not assessed (no evidence available):** physical security of the hosting location, workforce HR processes, existing organisational policies. These are marked *Not implemented* below until evidence is provided.

## 2. Evidence gathered beyond step 1

| Check | Result | Bearing |
|---|---|---|
| GitHub repo visibility (`api.github.com/repos/...` → HTTP 200 unauthenticated) | **Public** | `Records` (two name + phone pairs) is publicly readable since commit `052b2d8`; only PHI-bearing file ever committed (historical `dataset/appointments.json` was synthetic `Patient-NNNN` rows, removed in `1235a46`) |
| Web server CORS (`pipecat/runner/run.py:501`) | `allow_origins=["*"]` by default | unauthenticated PHI endpoints are callable cross-origin from any website |
| File modes | `.env` 0664, `.admin_token_secret` 0664, `hospital.db` 0644, `chats/` 0775 | readable by any local user / group member |
| Disk encryption (`lsblk`) | `/media/parmpal/Data` and `/` are plain partitions — no dm-crypt | data at rest on the workstation is unencrypted |
| Backups | none found (no backup dir, no `.bak`, no scheduled job) | single SQLite file is the only copy |
| Dependency pinning | 0 `==` pins in `requirements.txt` | builds are non-reproducible; no vulnerability scanning |
| Admin login (`admin_routes.py:179`) | no lockout, no rate limit, default `Admin/12345` | brute-force and default-credential exposure |

## 3. Scoring method

**Likelihood** that the threat exploits the vulnerability within 12 months, given current controls:

| | Definition |
|---|---|
| **High (3)** | Exploitable today with no special access, or the exposure is already occurring (e.g. PHI is sent to the vendor on every turn) |
| **Medium (2)** | Requires some access, skill, or an event (local user, network position, device loss) |
| **Low (1)** | Requires privileged access or an unlikely chain of events |

**Impact** on confidentiality, integrity, or availability of ePHI if exploited:

| | Definition |
|---|---|
| **High (3)** | Reportable breach affecting many patients, or loss of the only copy of ePHI |
| **Medium (2)** | Disclosure/alteration affecting one or few patients, or recoverable service loss |
| **Low (1)** | No PHI exposed; nuisance or compliance-documentation gap only |

**Risk = Likelihood × Impact** → **Critical 9 · High 6 · Medium 3–4 · Low 1–2.** Critical and High must be remediated or formally accepted before real PHI is processed.

## 4. Asset inventory (ePHI)

| ID | Asset | PHI | Criticality | Location |
|---|---|---|---|---|
| A1 | `dataset/hospital.db` + seed JSON | patients, appointments, prescriptions | Critical | workstation / container FS |
| A2 | `chats/sessions/*.json`, `chats/chat-*.jsonl` | transcripts, phone, patient_id, channel ids | Critical | FS |
| A3 | `chats/tool_call.json` | tool args/results | High | FS; MCP `/tool-calls` |
| A4 | `chats/messages/**` (LLM dumps) | full prompts incl. PHI | High | FS; Admin API |
| A5 | `app.log` | graph state, transcripts, tool args | High | FS |
| A6 | `metrics/bot_messages.csv`, `chats/timeline.jsonl` | user utterances | Medium | FS |
| A7 | Secrets: `.env`, `.admin_token_secret`, vendor keys | keys gating A1–A6 and vendor accounts | Critical | FS |
| A8 | Web app (`bot.py`, FastAPI, `/app`, `/admin`, `/api/*`) | in-transit PHI | Critical | host :7860 / Pipecat Cloud |
| A9 | MCP server (`mcp_server.py`) + cloudflared tunnel | tool traffic, `/tool-calls` | Critical | host :8000 / public URL |
| A10 | Telegram bot, Discord bot processes | chat text | High | host + vendor cloud |
| A11 | Voice pipeline (Daily, Deepgram, OpenAI Realtime) | raw audio | High | vendor cloud |
| A12 | LLM providers (DeepSeek, OpenAI, Groq, GLM) | conversation + tool results | High | vendor cloud |
| A13 | GitHub repository (public) | code; `Records` file | High | GitHub |
| A14 | Developer workstation (Linux, unencrypted disks) | hosts A1–A7 | Critical | physical |

## 5. Risk register

Ratings are *current state*. "Existing controls" lists only what the code actually enforces.

| ID | Threat event | Vulnerability (evidence) | Assets | Existing controls | L | I | **Risk** | HIPAA ref |
|---|---|---|---|---|---|---|---|---|
| R-01 | External attacker reads patient sessions and tool results over the network | `GET /api/call-logs/recent` (`log_routes.py:84`) and MCP `GET /tool-calls` (`mcp_server.py:244`) have no auth; CORS `*`; MCP URL published via cloudflared (`MD/mcp-cloudflare.md:56`) | A2, A3, A8, A9 | none | 3 | 3 | **9 Critical** | §164.312(a)(1), (d), (e)(1) |
| R-02 | PHI disclosed to the public via source control | `Records` (name + phone) tracked in a **public** GitHub repo since `052b2d8` | A13 | `.gitignore` covers other data dirs | 3 | 2 | **6 High** (already occurring — treat as a potential incident, see §8) | §164.308(a)(6), §164.402 |
| R-03 | Impermissible disclosure to vendors without a BAA | Full conversation + tool results sent to DeepSeek (default, no BAA, PRC storage), GLM (CN endpoint, no BAA), Groq and OpenAI (BAA obtainable but **not executed**); no ZDR flags (`Model.py`, `admin_settings.json:4-5,24`) | A12 | none | 3 | 3 | **9 Critical** | §164.308(b)(1), §164.314(a), §164.502(e) |
| R-04 | Raw patient audio processed by vendors without BAA / retention controls | Deepgram STT/TTS, OpenAI Realtime, Daily transport (`bot.py:155-169,272-286,437`) | A11 | none | 3 | 3 | **9 Critical** | §164.308(b)(1), §164.312(e) |
| R-05 | PHI transits consumer messaging platforms that will not sign a BAA; patient identity linked to platform account | Telegram/Discord bots carry names, phones, appointments; user ids persisted (`telegram_bot.py:104`, `discord_bot.py:110-117`) | A10, A2 | none | 3 | 3 | **9 Critical** | §164.308(b)(1), §164.312(e)(1) |
| R-06 | Unauthorised admin access → bulk export of all patients, prescriptions, transcripts | Default `Admin/12345` (`admin_routes.py:51-52`); token = HMAC(username), no expiry/revocation (`:70-76`); no lockout/rate limit (`:179`); HTML pages unauthenticated server-side (`:141-177`) | A1, A2, A4, A8 | HMAC token on JSON APIs | 3 | 3 | **9 Critical** | §164.312(a)(2)(i), (d); §164.308(a)(5)(ii)(D) |
| R-07 | Session/credential theft on the network | No TLS in repo; admin cookie lacks `Secure` (`admin_routes.py:186-193`); plain HTTP binds | A8, A9 | `httponly`, `samesite=lax` | 2 | 3 | **6 High** | §164.312(e)(1), (e)(2)(ii) |
| R-08 | Anyone can act as an authorised MCP client | `--oauth` uses `DemoOAuthProvider` that auto-approves and skips login (`mcp_demo_oauth.py:42-44`); OAuth off by default | A9 | none | 3 | 3 | **9 Critical** (when exposed) | §164.312(d) |
| R-09 | Transcript tampering / log poisoning | `POST /api/call-logs/event` writes arbitrary text into any session id, unauthenticated (`log_routes.py:64-75`) | A2 | none | 3 | 2 | **6 High** | §164.312(c)(1) |
| R-10 | Wrong-person disclosure via weak identity proofing | Phone match alone ⇒ `verified=True` (`db_patients.py:271`); on-file name returned before verification (`tool_patient.py:131-142`); collision reveals other person's name (`tool_patient.py:226-234`, `tool_booking.py:63-71`); phone-only cancel (`tool_booking.py:191-216`); no rate limit on lookups ⇒ phone enumeration | A1, A8–A10 | prompt-level instructions only | 3 | 2 | **6 High** | §164.312(d), §164.514(h) |
| R-11 | Loss/theft of workstation or disk exposes all ePHI | No disk encryption; SQLite and all logs plaintext; `.env` 0664 | A1–A7, A14 | OS login | 2 | 3 | **6 High** | §164.312(a)(2)(iv), §164.310(d)(1) |
| R-12 | Breach scope multiplied by PHI proliferation in logs | Full graph state at INFO (`hospital_graph.py` ×7), tool arg logging, `tool_call.json`, LLM dumps **on**, metrics CSV | A3–A6 | log rotation 20 MB×5 for `app.log` only | 3 | 2 | **6 High** | §164.308(a)(1)(ii)(B), §164.502(b) minimum necessary |
| R-13 | Indefinite retention increases breach exposure and violates disposal requirements | No retention/purge for `chats/`, `timeline.jsonl`, metrics, dumps; no media-disposal procedure | A1–A6 | manual delete via Admin | 3 | 2 | **6 High** | §164.310(d)(2)(i)-(ii) |
| R-14 | PHI access cannot be detected or investigated | No audit log of Admin reads; tool calls not attributed to a workforce user; `app.log` is a debug log, not an audit trail | A1–A8 | — | 3 | 2 | **6 High** | §164.312(b), §164.308(a)(1)(ii)(D) |
| R-15 | No unique user identity for workforce; shared account | Single `ADMIN_USER`; no roles; no emergency-access procedure | A8 | — | 3 | 2 | **6 High** | §164.312(a)(2)(i)-(ii), §164.308(a)(3)-(4) |
| R-16 | Data loss: corruption/deletion of the only copy | No backups; `clear_table()` irreversible (`db_core.py:263`); no contingency plan | A1, A2 | — | 2 | 3 | **6 High** | §164.308(a)(7)(ii)(A)-(B), §164.310(d)(2)(iv) |
| R-17 | Local user / co-tenant reads secrets and data | `.env`, `.admin_token_secret` 0664; `chats/` 0775 | A7, A2 | single-user workstation today | 1 | 3 | **3 Medium** | §164.312(a)(1) |
| R-18 | Vulnerable dependency exploited | No pins, no SCA/vuln scanning, `latest` Docker base, no patch process | A8–A10 | — | 2 | 2 | **4 Medium** | §164.308(a)(5)(ii)(B), (a)(8) |
| R-19 | Prompt injection causes tool misuse (cancel/reschedule/disclose) | Patient text flows straight to the LLM which calls write tools; MCP exposes write tools to any connected client | A1, A8, A9 | tools require phone; prompts instruct confirmation | 2 | 2 | **4 Medium** | §164.312(c)(1) |
| R-20 | Public MCP/tunnel abused to write bookings or enumerate data | cloudflared quick tunnel, `--allow-tunnel` disables DNS-rebinding guard (`mcp_server.py:120`); docs recommend "No Authentication" (`MD/mcp-cloudflare.md:43`) | A9 | none | 3 | 3 | **9 Critical** (while tunnel is up) | §164.312(a)(1), (e)(1) |
| R-21 | Hosting provider handles PHI without BAA | Pipecat Cloud deployment (`pcc-deploy.toml`), Cloudflare quick tunnel — no contracts | A8, A9 | — | 2 | 3 | **6 High** | §164.308(b)(1) |
| R-22 | Incident goes unnoticed / unreported | No incident response or breach-notification procedure; no monitoring | all | — | 3 | 2 | **6 High** | §164.308(a)(6), §164.400-414 |
| R-23 | Workforce mishandles PHI | No policies, training, sanctions, access-termination procedures; no designated Security Officer | all | — | 3 | 2 | **6 High** | §164.308(a)(1)(ii)(C), (a)(2), (a)(3), (a)(5) |
| R-24 | Real and synthetic data indistinguishable; test traffic uses real phone numbers | `MakeDataBase.py` rows identical in shape; session files contain developer phone numbers (`chats/sessions/sess-20260717-…`) | A1, A2 | `_is_seed_chat()` for chats only | 2 | 2 | **4 Medium** | §164.308(a)(1)(ii)(A) |
| R-25 | Integrity of ePHI not verifiable | No checksums/signing of DB or logs; SQLite writable by app user and any local user | A1, A2 | — | 1 | 2 | **2 Low** | §164.312(c)(2) |

**Summary:** 7 Critical · 13 High · 4 Medium · 1 Low. The Critical set is dominated by two themes — *PHI reachable without authentication* (R-01, R-06, R-08, R-20) and *PHI disclosed to vendors/platforms without a BAA* (R-03, R-04, R-05).

## 6. Security Rule safeguard gap analysis

Status: **Not implemented** · **Partial** · **Implemented** · **N/A**. (R) = required, (A) = addressable — addressable specs still need a documented decision.

### Administrative safeguards §164.308

| Standard / spec | Status | Evidence / gap |
|---|---|---|
| (a)(1)(ii)(A) Risk analysis (R) | Partial | This document; needs owner review and periodic update |
| (a)(1)(ii)(B) Risk management (R) | Not implemented | Remediation plan §7 not yet executed |
| (a)(1)(ii)(C) Sanction policy (R) | Not implemented | none |
| (a)(1)(ii)(D) Information system activity review (R) | Not implemented | no audit log exists to review (R-14) |
| (a)(2) Assigned security responsibility (R) | Not implemented | no Security Officer named |
| (a)(3) Workforce security — authorisation, clearance, termination (A) | Not implemented | single shared admin account (R-15) |
| (a)(4) Information access management (R/A) | Not implemented | no roles; any admin sees everything |
| (a)(5) Security awareness & training; (ii)(B) malware, (ii)(C) log-in monitoring, (ii)(D) password management (A) | Not implemented | default password; no lockout/monitoring (R-06) |
| (a)(6) Security incident procedures (R) | Not implemented | R-22 |
| (a)(7) Contingency plan — backup (R), DR (R), emergency mode (R), testing (A), criticality analysis (A) | Not implemented | no backups (R-16) |
| (a)(8) Evaluation (R) | Partial | this assessment; no recurring schedule |
| (b)(1) Business associate contracts (R) | Not implemented | zero BAAs across ten PHI recipients (R-03/04/05/21) |

### Physical safeguards §164.310

| Standard / spec | Status | Evidence / gap |
|---|---|---|
| (a) Facility access controls (A) | Not assessed | workstation-hosted; no evidence |
| (b) Workstation use (R); (c) Workstation security (R) | Not implemented | dev workstation holds all ePHI unencrypted (R-11) |
| (d)(1) Device & media controls — disposal (R), re-use (R), accountability (A), backup before move (A) | Not implemented | no disposal/retention procedure (R-13), no backups |

### Technical safeguards §164.312

| Standard / spec | Status | Evidence / gap |
|---|---|---|
| (a)(1) Access control — unique user ID (R) | Not implemented | shared admin; unauth endpoints (R-01, R-15) |
| (a)(2)(ii) Emergency access (R) | Not implemented | — |
| (a)(2)(iii) Automatic logoff (A) | Partial | admin cookie 12 h max-age only |
| (a)(2)(iv) Encryption/decryption at rest (A) | Not implemented | R-11 |
| (b) Audit controls (R) | Not implemented | R-14 |
| (c)(1) Integrity; (c)(2) mechanism to authenticate ePHI (A) | Not implemented | R-09, R-25 |
| (d) Person or entity authentication (R) | Not implemented | no patient authentication; demo OAuth; default admin creds (R-06, R-08, R-10) |
| (e)(1) Transmission security; (e)(2)(i) integrity controls (A); (e)(2)(ii) encryption (A) | Not implemented | no TLS in repo; vendors reached over HTTPS but no contractual cover (R-07) |

### Organisational & documentation §164.314, §164.316

| Standard | Status | Gap |
|---|---|---|
| §164.314(a) BA contract contents | Not implemented | no BAAs |
| §164.316(a) Policies & procedures; (b) documentation, 6-year retention, availability, updates | Not implemented | only these two documents exist |

## 7. Vendor / business associate analysis

Current status is **no BAA with anyone**. The table records what each vendor offers so Phase 1 decisions are grounded (verified 2026-08-21; terms change — re-verify before signing).

| Recipient | PHI received today | BAA available? | Conditions | Retention / ZDR controls | Decision needed |
|---|---|---|---|---|---|
| OpenAI API (chat + Realtime) | conversation, tool results, raw audio | **Yes** | request via baa@openai.com; case-by-case; only ZDR-eligible endpoints in scope (`/v1/chat/completions`, `/v1/realtime` qualify) | ZDR on approval; default 30-day abuse logs; no training on API data | candidate primary LLM + voice, **only after BAA + ZDR** |
| Deepgram (STT/TTS) | raw audio, reply text | **Yes** | enterprise contract via sales | set `mip_opt_out=true` (default opts into training programme), `redact=phi/pii`; transcripts not stored | keep under BAA; enable opt-out + redaction |
| Daily (WebRTC) | audio transport | **Yes** | Healthcare add-on (~$500/mo), HIPAA mode per domain | HIPAA domains scrub names/IDs from logs, no recordings unless customer-controlled | keep under BAA + HIPAA mode |
| Pipecat Cloud (hosting) | everything the bot handles | **Yes** | contact help@daily.co; one BAA covers Pipecat Cloud + Daily | HIPAA mode: no conversation logging | viable host if BAA executed |
| Groq | conversation | **Yes** (published BA Addendum) | requires Groq Services Agreement; excludes beta/preview/free tiers | no retention by default; self-serve ZDR | possible secondary LLM under BAA; otherwise remove |
| DeepSeek | conversation (**current default**) | **No** | — | data stored in PRC; used for training unless opted out by email | **remove for PHI** |
| Zhipu / GLM (`open.bigmodel.cn`) | conversation | **No** (none found) | China-hosted endpoint | no HIPAA terms | **remove for PHI** (self-hosted open weights would be the only path) |
| Cloudflare Tunnel | all MCP traffic | **Yes** | Enterprise plan only | Tunnel/Access/Gateway in scope under BAA | quick tunnels are out; Enterprise + Access, or no public MCP |
| Telegram Bot API | chat text | **No** | — | none | drop, or non-PHI channel only |
| Discord | chat text | **No** | — | not E2E encrypted | drop, or non-PHI channel only |
| Google Gemini | none at runtime | Vertex AI: yes (GCP BAA); AI Studio key: **no** (terms prohibit clinical use) | — | — | remove `GOOGLE_API_KEY` from `.env` or move to Vertex |

Sources: OpenAI data guide & BAA process; Deepgram trust/privacy + MIP docs; Daily HIPAA blog/pricing; Pipecat Cloud HIPAA doc; Groq Customer BAA & your-data docs; DeepSeek privacy policy; Z.ai privacy policy; Cloudflare HIPAA trust hub; Paubox summaries for Telegram/Discord; Google Cloud HIPAA + Gemini API terms.

## 8. Immediate incident note — `Records` in a public repository

`Records` contains two name + phone pairs and has been publicly readable on GitHub since commit `052b2d8`. Whether this is a reportable breach depends on whether those are real patients' identifiers (if they are developer test entries, document that determination and its basis). Required now regardless: (1) remove the file and **rewrite history** (`git filter-repo`) then force-push; (2) ask GitHub Support to purge cached views; (3) record the incident, the determination, and the remediation date in an incident log (§164.308(a)(6)(ii)); (4) if the identifiers are real and the provider is a covered entity, run the four-factor breach risk assessment under §164.402(2).

## 9. Remediation plan (risk management, §164.308(a)(1)(ii)(B))

Sequenced so that the highest exposure drops first and each phase is independently verifiable. "Closes" lists the risk IDs reduced to ≤ Medium when the phase is done.

### Phase 0 — Stop the bleeding (days; code-only)

| # | Action | Closes |
|---|---|---|
| 0.1 | Require admin auth on `GET /api/call-logs/recent`, and remove or auth-gate MCP `GET /tool-calls`; auth-gate `POST /api/call-logs/*` (per-session secret issued at call start) | R-01, R-09 |
| 0.2 | Restrict CORS to the app's own origin(s) (`--allowed-origins`) | R-01 |
| 0.3 | Remove `Records`; `git filter-repo` the history; force-push; log the incident (§8) | R-02 |
| 0.4 | Strip PHI from logs: delete the seven `pformat(state)` dumps; log tool calls by `patient_id` hash only; redact phone/name in `voice_bridge.py` and tool loggers; set `save_llm_messages=false` and make it a dev-only flag; stop writing `user_question` to metrics | R-12 |
| 0.5 | Admin hardening: refuse to start with default credentials; bcrypt-hashed password from secret store; signed token with expiry + server-side revocation list; `Secure` cookie; lockout after N failures; `_require_auth` on HTML routes | R-06 |
| 0.6 | MCP: make OAuth mandatory for HTTP transports; replace `DemoOAuthProvider` with a real IdP or pre-shared bearer tokens; delete the `--allow-tunnel` / "No Authentication" guidance from docs | R-08, R-20 |
| 0.7 | `chmod 600` on `.env`, `.admin_token_secret`, `hospital.db`; `chmod 700` on `chats/`, `dataset/`, `metrics/` | R-17 |

### Phase 1 — Decide the target architecture (weeks; decisions + contracts)

| # | Action | Closes |
|---|---|---|
| 1.1 | **Vendor decision.** Keep only PHI processors that sign a BAA (see §7). Practically: one LLM provider under BAA + ZDR (OpenAI Enterprise/API ZDR, Azure OpenAI, AWS Bedrock, or Google Vertex), Deepgram under BAA, Daily under its HIPAA offering; **remove DeepSeek and GLM as selectable providers** for any PHI deployment; keep Groq only if its BAA is executed | R-03, R-04 |
| 1.2 | **Channel decision.** Telegram and Discord cannot be made compliant for PHI. Options: (a) drop them; (b) keep them as *non-PHI* channels — general info + hand-off link to the authenticated web app, no lookup/book tools exposed; pick and document | R-05 |
| 1.3 | **Hosting decision.** Deploy on a host that signs a BAA (cloud VM/containers in AWS/GCP/Azure with BAA, or Pipecat Cloud only if a BAA is obtainable); retire cloudflared quick tunnels in favour of a named tunnel / load balancer under contract, or no public MCP at all | R-21, R-20 |
| 1.4 | Execute BAAs and record them in a BA inventory (§164.308(b), §164.314) | R-03–R-05, R-21 |

### Phase 2 — Core technical safeguards (weeks)

| # | Action | Closes |
|---|---|---|
| 2.1 | TLS everywhere: terminate HTTPS in front of FastAPI/MCP (reverse proxy or platform LB); HSTS; `Secure` cookies | R-07 |
| 2.2 | Encryption at rest: move SQLite → SQLCipher or a managed Postgres with encrypted volume; encrypt the log/chat volume; full-disk encryption on any workstation that holds PHI | R-11 |
| 2.3 | Identity & access: per-user admin accounts (unique IDs), roles (viewer/scheduler/admin), MFA, session timeout, emergency-access account procedure | R-15, R-06 |
| 2.4 | Audit logging: append-only audit table/stream for every PHI read/write (who, what patient, when, via which channel/tool); tamper-evident (hash chain or WORM storage); retain 6 years; weekly review procedure | R-14 |
| 2.5 | Patient identity proofing: require a second factor before disclosing or changing anything (DOB / OTP to the on-file phone); never return the on-file name/phone before verification; on phone collision return a generic message; name + phone for cancel/reschedule; rate-limit lookups per phone and per IP | R-10 |
| 2.6 | Retention & disposal: policy (e.g. transcripts 90 days, audit logs 6 years), scheduled purge job, secure-delete procedure for media; document data classes and schedules | R-13 |
| 2.7 | Backups & contingency: nightly encrypted backups off-host, restore test quarterly, documented DR and emergency-mode procedures | R-16 |
| 2.8 | Dependency hygiene: pin versions, enable Dependabot/`pip-audit`, pin Docker base digest, run container as non-root | R-18 |
| 2.9 | Prompt-injection hardening: tool-side authorisation checks (the tool, not the prompt, enforces who may cancel what); allow-list of tools per channel; MCP exposes read-only tools unless the client is authenticated as staff | R-19 |
| 2.10 | Mark synthetic rows (`is_synthetic` column / separate seed DB); prohibit real phone numbers in test traffic | R-24 |

### Phase 3 — Administrative program (ongoing)

| # | Action | Closes |
|---|---|---|
| 3.1 | Appoint Security Officer and Privacy Officer; publish contact | §164.308(a)(2) |
| 3.2 | Policies & procedures set: acceptable use, access control, password/MFA, workstation, incident response & breach notification, sanctions, retention/disposal, BA management, contingency, change management | R-22, R-23, §164.316 |
| 3.3 | Workforce training at onboarding + annual; signed acknowledgement | R-23 |
| 3.4 | Incident response runbook + breach risk-assessment template; tabletop test | R-22 |
| 3.5 | Recurring evaluation: re-run this risk assessment annually and on major change; vulnerability scan quarterly | §164.308(a)(8) |

### Projected residual risk after Phases 0–2

| Risk | Now | After |
|---|---|---|
| R-01, R-06, R-08, R-20 (unauth access) | Critical | Low |
| R-03, R-04 (vendors) | Critical | Low–Medium (depends on BAA + ZDR execution) |
| R-05 (Telegram/Discord) | Critical | Low if dropped / made non-PHI; **cannot** be reduced if they keep carrying PHI |
| R-07, R-11, R-12, R-13, R-14, R-15, R-16 | High | Low |
| R-10 | High | Medium (identity proofing over chat is inherently limited) |
| R-22, R-23 | High | Medium until the programme has run a full cycle |

## 10. Acceptance and review

| | |
|---|---|
| Risks accepted as-is | none yet |
| Next review | after Phase 0 (re-score R-01, R-02, R-06, R-08, R-09, R-12, R-17, R-20), then after Phase 2, then annually |
| Owner sign-off | ______________________ (Security Officer) date ________ |

## Next step

**Step 3 — Phase 0 implementation**: the seven code-only actions above. They require no vendor or architecture decisions and remove every currently-reachable exposure. Each can be delivered as a reviewed PR with tests.
