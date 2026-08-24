# OpenAI BAA + Zero Data Retention request — email template

Send **from the email that owns your OpenAI API organization** (the org must be on a paid
API plan in good standing). Fill every `[bracket]`; delete the notes in *italics*.
Before sending: note your **Organization ID** (platform.openai.com → Settings → Organization →
General) and make sure the org has a payment method and some API usage history.

---

**To:** baa@openai.com
**Subject:** BAA + Zero Data Retention request — [Your Company], API org `[org-XXXXXXXX]`

Hello OpenAI team,

I'm writing to request a **Business Associate Agreement (BAA)** and **Zero Data Retention (ZDR)**
for our OpenAI API organization.

**Organization**
- Company: [Your Company legal name]
- OpenAI organization ID: `[org-XXXXXXXX]`
- Account owner / contact: [Full name], [title], [email], [phone]
- Country / address: [address]
- Current plan & usage: paid API account since [month year]; approx. [N] requests/day today, expected [N] at launch

**Use case**
We operate **DBC Care**, an appointment-scheduling assistant for [Hospital / clinic name], a
[US-based healthcare provider / HIPAA covered entity]. Patients interact by web chat and voice to
book, reschedule and cancel appointments and to ask about their prescriptions. We act as the
hospital's **business associate**, and OpenAI would be our **subcontractor business associate**.

**What is sent to OpenAI**
- Conversation text that may contain PHI: patient name, phone number, address, appointment
  date/time, department/doctor, medication names.
- In voice mode, patient audio via the Realtime API.
- No images, no file uploads, no fine-tuning on patient data.

**Endpoints / models we will use (all ZDR-eligible, to our understanding)**
- `POST /v1/chat/completions` — model `[gpt-4o-mini / gpt-4.1-mini / other]`, with `store=false`
- `/v1/realtime` (WebSocket) — model `[gpt-realtime / gpt-realtime-mini]`
- We do **not** use Assistants, Files, Batch, fine-tuning, or the consumer ChatGPT products for PHI.

**Requests**
1. Execute a BAA covering organization `[org-XXXXXXXX]`.
2. Enable **Zero Data Retention** for the endpoints above on that organization.
3. Confirm in writing (a) which endpoints/models are in scope of the BAA, (b) that API data is
   not used for training, and (c) the retention period for any remaining abuse-monitoring logs
   under ZDR.
4. Confirm the data-processing regions available to us (we would prefer [US] processing).

**Safeguards on our side (for your eligibility review)**
- HIPAA Security Risk Assessment completed; remediation in place: PHI only sent to BAA-covered
  vendors (enforced in code), TLS in transit, encrypted storage, hash-chained audit logging of
  all PHI access, role-based admin access with MFA-ready accounts, retention/disposal schedule,
  encrypted backups, incident-response procedure.
- We are prepared to sign your standard BAA and to complete any security questionnaire.

Please let me know if you need anything else to proceed. I'm happy to jump on a call.

Thank you,

[Full name]
[Title], [Your Company]
[email] · [phone]
[Company website]

---

## After you send it

| When | What |
|---|---|
| Reply asks for details | Typical questions: expected volume, whether you're a covered entity or BA, which endpoints. Answer from the list above. |
| BAA arrives (DocuSign/PDF) | Have the signatory be an officer of your company. Read: scope (org ID, endpoints), breach-notification timelines, subcontractor terms, termination. |
| Approval email says ZDR enabled | Save the email + signed PDF in `compliance/contracts/openai/` (outside the repo). Keep for **6 years**. |
| Then, in the app | `PHI_ALLOWED_PROVIDERS=openai`, `HIPAA_MODE=1`, restrict API keys to that org, turn off any data-sharing toggles in the OpenAI org settings, record the BAA in the BA inventory (step 4). |
| If declined or slow | Fall back to **Azure OpenAI** (Microsoft's standard BAA, self-serve via the Azure portal) or **Groq** (published Customer BAA). Both are already allow-listable. |

*Do not send any real patient data until the signed BAA and the ZDR confirmation are in hand.*
