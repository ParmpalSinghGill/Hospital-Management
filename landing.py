"""ReceptionAi marketing landing page (served at `/`).

Kept out of bot.py so the markup can grow without bloating the voice entrypoint.
`render()` reads the live industry profile, so the page reflects the deployment's
actual configuration rather than static copy.

NOTE ON DEMO CONTENT: the testimonials, case-study metrics, partner names and
pricing below are sample content for a demo deployment. They are labelled as such
in the UI. Replace them with real figures before using this page publicly.
"""
from __future__ import annotations

import html

from industry_profile import all_profiles, current

# --- Sample social proof (clearly labelled in the UI; swap before going live) ---
_TESTIMONIALS = (
    (
        "Front desk used to miss a dozen calls over lunch. Now every one gets picked "
        "up and booked before we're back.",
        "Sample quote",
        "Multi-site clinic",
    ),
    (
        "Friday nights were the worst — phone ringing while we're seating people. "
        "It just handles the bookings now.",
        "Sample quote",
        "Neighbourhood restaurant",
    ),
    (
        "Guests calling at 2am to change a check-in date get an answer instead of "
        "voicemail.",
        "Sample quote",
        "Boutique hotel",
    ),
)

# The reply-time figure is the measured median time-to-first-audio across the 273
# recorded voice turns in metrics/bot_messages.csv (p90 3.2s). Recompute it from
# your own traffic rather than quoting this build's demo number.
_STATS = (
    ("24/7", "Answered", "No voicemail, no hold music, no missed bookings."),
    ("3", "Industries", "Clinics, restaurants and hotels from one engine."),
    ("6", "Channels", "Web chat, voice, Telegram, Discord, MCP and CLI."),
    ("~1.9s", "To reply", "Median time to first spoken word, measured on this build."),
)

_STEPS = (
    (
        "The call comes in",
        "Someone rings or opens the chat. ReceptionAi picks up straight away and "
        "asks what they need — no phone tree, no menu of options.",
    ),
    (
        "It works out what they want",
        "The caller talks normally. A router reads intent from natural speech and "
        "hands off to a specialist for booking, cancelling or rescheduling.",
    ),
    (
        "The booking is done",
        "It checks real availability, confirms the details back, writes the booking "
        "to your database and reads out the confirmation. End to end, no callback.",
    ),
)

_BENEFITS = (
    (
        "Never miss a booking",
        "Every call and message gets answered — nights, weekends and the middle of "
        "your lunch rush included.",
    ),
    (
        "Speaks like a person",
        "One short question at a time. No IVR menus, no reading options aloud, no "
        "asking the same thing twice.",
    ),
    (
        "Finishes the job",
        "It books, cancels and reschedules itself. It never says someone will call "
        "you back.",
    ),
    (
        "Knows your calendar",
        "Checks live availability before promising anything, and offers the nearest "
        "free time when a slot is taken.",
    ),
    (
        "Verifies who's calling",
        "Looks callers up by phone, confirms the name back, and never identifies "
        "someone from memory alone.",
    ),
    (
        "Everywhere at once",
        "The same assistant on your website, phone line, Telegram, Discord — sharing "
        "one database.",
    ),
)

_FAQS = (
    (
        "Does it work for my kind of business?",
        "It ships with hospital, restaurant and hotel profiles. The booking engine is "
        "the same underneath — a caller reserves time with a named resource — so the "
        "vocabulary swaps between doctors, tables and rooms without changing the logic.",
    ),
    (
        "What happens if it doesn't understand?",
        "It asks one short clarifying question rather than guessing. It never invents "
        "an availability, a name or a confirmation number — it only speaks values that "
        "came back from your data.",
    ),
    (
        "Can it handle changes and cancellations?",
        "Yes. Those are separate specialists. It finds the existing booking from a "
        "phone number, confirms once, then makes the change.",
    ),
    (
        "Does it replace my front desk?",
        "It covers the calls your team can't get to — the overflow, the after-hours, "
        "the ones that come in while you're with someone. The routine bookings stop "
        "landing in voicemail.",
    ),
    (
        "Which voice and language models does it use?",
        "Deepgram for speech, and your choice of DeepSeek, Groq, Zhipu GLM or OpenAI "
        "for the conversation. There's also an OpenAI Realtime speech-to-speech mode. "
        "You pick per service in Admin.",
    ),
    (
        "Where does the booking data live?",
        "In your own database. This build ships with SQLite for demos; the tools layer "
        "is a thin wrapper you can point at your existing system.",
    ),
)

_INDUSTRY_ART = {
    "hospital": "M12 3v18M3 12h18",
    "restaurant": "M6 3v8a2 2 0 0 0 4 0V3M8 11v10M17 3c-1.7 1.2-2.5 3-2.5 5.5S15.3 13 17 13v8",
    "hotel": "M3 20V7l9-4 9 4v13M9 20v-6h6v6",
}


def _industry_path(industry_id: str) -> str:
    return _INDUSTRY_ART.get(industry_id, _INDUSTRY_ART["hospital"])


def render() -> str:
    active = current()
    profiles = all_profiles()
    esc = html.escape

    # Hero transcript is written in the active industry's own words.
    demo_line = {
        "hospital": "I need to see someone about my tooth, it&rsquo;s been aching for days.",
        "restaurant": "Do you have a table for four on Friday around seven?",
        "hotel": "I&rsquo;d like a room from Friday, staying two nights.",
    }.get(active.id, "I&rsquo;d like to make a booking for Friday.")
    demo_reply = {
        "hospital": "Sorry you&rsquo;re dealing with that. What&rsquo;s your phone number?",
        "restaurant": "Happy to get you a table. What&rsquo;s your phone number?",
        "hotel": "Of course, I can set that up. What&rsquo;s your phone number?",
    }.get(active.id, "Of course. What&rsquo;s your phone number?")

    industry_cards = "\n".join(
        f"""          <article class="vertical{' is-live' if p['id'] == active.id else ''}">
            <svg class="vertical-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="{_industry_path(p['id'])}" stroke="currentColor" stroke-width="1.6"
                    stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h3>{esc(p['label'])}</h3>
            <p>{esc(p['tagline'])}</p>
            <dl class="vocab">
              <div><dt>Books</dt><dd>{esc(p['booking_plural'])}</dd></div>
              <div><dt>With</dt><dd>{esc(p['resource_plural'])}</dd></div>
              <div><dt>For</dt><dd>{esc(p['customer_label'])}s</dd></div>
            </dl>
            {'<span class="live-tag">Running now</span>' if p['id'] == active.id else ''}
          </article>"""
        for p in profiles
    )

    stat_items = "\n".join(
        f"""          <div class="stat">
            <div class="stat-figure">{esc(figure)}</div>
            <div class="stat-label">{esc(label)}</div>
            <p class="stat-note">{esc(note)}</p>
          </div>"""
        for figure, label, note in _STATS
    )

    step_items = "\n".join(
        f"""          <li class="step">
            <span class="step-index" aria-hidden="true">{i}</span>
            <div>
              <h3>{esc(title)}</h3>
              <p>{esc(body)}</p>
            </div>
          </li>"""
        for i, (title, body) in enumerate(_STEPS, start=1)
    )

    benefit_items = "\n".join(
        f"""          <article class="benefit">
            <h3>{esc(title)}</h3>
            <p>{esc(body)}</p>
          </article>"""
        for title, body in _BENEFITS
    )

    quote_items = "\n".join(
        f"""          <figure class="quote">
            <blockquote>{esc(text)}</blockquote>
            <figcaption><span class="quote-who">{esc(who)}</span>{esc(where)}</figcaption>
          </figure>"""
        for text, who, where in _TESTIMONIALS
    )

    faq_items = "\n".join(
        f"""          <details class="faq">
            <summary>{esc(q)}</summary>
            <p>{esc(a)}</p>
          </details>"""
        for q, a in _FAQS
    )

    example_chips = "\n".join(
        f'            <li>&ldquo;{esc(ex)}&rdquo;</li>' for ex in active.examples[:3]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ReceptionAi &mdash; the front desk that always picks up</title>
<meta name="description" content="An AI receptionist that answers every call and books,
cancels and reschedules for clinics, restaurants and hotels." />
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%2307211f'/%3E%3Cg stroke='%232fbfa8' stroke-width='2.4' stroke-linecap='round'%3E%3Cpath d='M8 13v6'/%3E%3Cpath d='M13 9v14'/%3E%3Cpath d='M18 11v10'/%3E%3Cpath d='M23 14v4'/%3E%3C/g%3E%3C/svg%3E" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  :root {{
    --ink: #07211f;
    --ink-2: #0f3b37;
    --ink-3: #16514b;
    --paper: #f3f8f6;
    --paper-2: #ffffff;
    --signal: #2fbfa8;
    --signal-deep: #178f7c;
    --amber: #f2a65a;
    --on-dark: #d8ece7;
    --on-dark-soft: #8fb3ad;
    --on-light: #143430;
    --on-light-soft: #5d7b76;
    --line-dark: rgba(216, 236, 231, 0.14);
    --line-light: rgba(20, 52, 48, 0.12);
    --display: "Fraunces", Georgia, serif;
    --body: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
    --measure: 62ch;
    --pad: clamp(1.15rem, 4vw, 2.5rem);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--on-light);
    font-family: var(--body);
    font-size: 17px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ width: min(1120px, 100%); margin: 0 auto; padding-inline: var(--pad); }}
  section {{ padding-block: clamp(3.5rem, 8vw, 6.5rem); }}
  .band-dark {{ background: var(--ink); color: var(--on-dark); }}
  .band-light {{ background: var(--paper); }}
  .band-paper {{ background: var(--paper-2); }}

  h1, h2, h3 {{ font-family: var(--display); font-weight: 700; text-wrap: balance; margin: 0; }}
  h1 {{ font-size: clamp(2.5rem, 6.4vw, 4.5rem); line-height: 1.02; letter-spacing: -0.033em; }}
  h2 {{ font-size: clamp(1.85rem, 3.8vw, 2.9rem); line-height: 1.1; letter-spacing: -0.025em; }}
  h3 {{ font-size: 1.16rem; line-height: 1.3; letter-spacing: -0.012em; }}
  p {{ margin: 0; max-width: var(--measure); }}

  .eyebrow {{
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--signal); margin-bottom: 0.9rem;
  }}
  .band-light .eyebrow, .band-paper .eyebrow {{ color: var(--signal-deep); }}
  .section-head {{ margin-bottom: clamp(2rem, 4vw, 3rem); }}
  .section-head p {{ margin-top: 0.85rem; color: var(--on-light-soft); font-size: 1.05rem; }}
  .band-dark .section-head p {{ color: var(--on-dark-soft); }}

  /* ---------- Nav ---------- */
  .nav {{
    position: sticky; top: 0; z-index: 50;
    background: rgba(7, 33, 31, 0.9);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--line-dark);
  }}
  .nav-inner {{ display: flex; align-items: center; gap: 1.5rem; height: 4.2rem; }}
  .logo {{ display: flex; align-items: center; gap: 0.6rem; text-decoration: none; color: var(--on-dark); }}
  .logo-mark {{ width: 2.1rem; height: 2.1rem; flex: none; }}
  .logo-word {{ font-family: var(--display); font-size: 1.18rem; font-weight: 700; letter-spacing: -0.02em; }}
  .logo-word b {{ color: var(--signal); font-weight: 700; }}
  .nav-links {{ display: none; gap: 1.6rem; margin-left: auto; }}
  .nav-links a {{
    color: var(--on-dark-soft); text-decoration: none; font-size: 0.92rem; font-weight: 500;
  }}
  .nav-links a:hover {{ color: var(--on-dark); }}
  .nav-cta {{ margin-left: auto; }}
  @media (min-width: 900px) {{
    .nav-links {{ display: flex; }}
    .nav-cta {{ margin-left: 0; }}
  }}

  /* ---------- Buttons ---------- */
  .btn {{
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.82rem 1.4rem; border-radius: 999px; border: 1px solid transparent;
    font: inherit; font-weight: 600; font-size: 0.95rem; text-decoration: none;
    cursor: pointer; transition: transform 0.16s ease, box-shadow 0.16s ease,
                                background 0.16s ease, border-color 0.16s ease;
  }}
  .btn-call {{
    background: var(--signal); color: #05211d;
    box-shadow: 0 12px 30px rgba(47, 191, 168, 0.28);
  }}
  .btn-call:hover {{ background: #45d3bb; transform: translateY(-2px); }}
  .btn-ghost {{ border-color: var(--line-dark); color: var(--on-dark); }}
  .btn-ghost:hover {{ border-color: var(--signal); color: var(--signal); }}
  .band-light .btn-ghost, .band-paper .btn-ghost {{ border-color: var(--line-light); color: var(--on-light); }}
  .band-light .btn-ghost:hover, .band-paper .btn-ghost:hover {{ border-color: var(--signal-deep); color: var(--signal-deep); }}
  .btn-sm {{ padding: 0.58rem 1.05rem; font-size: 0.88rem; }}
  .btn:focus-visible, a:focus-visible, summary:focus-visible {{
    outline: 2px solid var(--signal); outline-offset: 3px; border-radius: 6px;
  }}
  .ring {{
    width: 0.62rem; height: 0.62rem; border-radius: 50%; background: #05211d; flex: none;
    animation: ring 1.8s ease-out infinite;
  }}
  @keyframes ring {{
    0%   {{ box-shadow: 0 0 0 0 rgba(5, 33, 29, 0.55); }}
    70%  {{ box-shadow: 0 0 0 0.5rem rgba(5, 33, 29, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(5, 33, 29, 0); }}
  }}

  /* ---------- Hero ---------- */
  .hero {{
    background:
      radial-gradient(900px 460px at 12% -10%, #14514a 0%, transparent 62%),
      radial-gradient(760px 420px at 92% 4%, #0d3f4d 0%, transparent 58%),
      var(--ink);
    color: var(--on-dark);
    padding-block: clamp(3rem, 6vw, 4.5rem) clamp(3rem, 6vw, 4.75rem);
    overflow: hidden;
  }}
  .hero-grid {{ display: grid; gap: clamp(2.5rem, 5vw, 4rem); align-items: start; }}
  @media (min-width: 940px) {{
    .hero-grid {{ grid-template-columns: 1.05fr 0.95fr; align-items: center; }}
    /* Optically centre the call card against the taller text column. */
    .callcard {{ margin-top: 0.5rem; }}
  }}
  .status-pill {{
    display: inline-flex; align-items: center; gap: 0.5rem; margin-bottom: 1.4rem;
    padding: 0.36rem 0.85rem 0.36rem 0.6rem; border-radius: 999px;
    border: 1px solid var(--line-dark); background: rgba(47, 191, 168, 0.08);
    font-size: 0.8rem; font-weight: 600; color: var(--on-dark);
  }}
  .status-dot {{
    width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--signal);
    box-shadow: 0 0 0 0 rgba(47, 191, 168, 0.7); animation: ring 2s ease-out infinite;
  }}
  .hero h1 span {{ color: var(--signal); font-style: italic; }}
  .hero-lead {{
    margin-top: 1.3rem; font-size: clamp(1.02rem, 1.7vw, 1.18rem);
    color: var(--on-dark-soft); max-width: 34rem;
  }}
  .hero-actions {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 2rem; }}
  .hero-foot {{
    margin-top: 1.4rem; font-size: 0.85rem; color: var(--on-dark-soft);
  }}

  /* Call panel */
  .callcard {{
    border-radius: 22px; border: 1px solid var(--line-dark);
    background: linear-gradient(170deg, rgba(22, 81, 75, 0.72), rgba(7, 33, 31, 0.92));
    box-shadow: 0 40px 90px rgba(0, 0, 0, 0.42);
    overflow: hidden;
  }}
  .callcard-top {{
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.95rem 1.15rem; border-bottom: 1px solid var(--line-dark);
    font-size: 0.85rem; color: var(--on-dark-soft);
  }}
  .callcard-top .who {{ color: var(--on-dark); font-weight: 600; }}
  .callcard-timer {{ margin-left: auto; font-variant-numeric: tabular-nums; }}
  .wave {{
    display: flex; align-items: center; justify-content: center; gap: 4px;
    height: 5.5rem; padding-inline: 1.15rem;
    border-bottom: 1px solid var(--line-dark);
  }}
  .wave i {{
    display: block; width: 4px; border-radius: 2px; background: var(--signal); opacity: 0.85;
    animation: bounce 1.25s ease-in-out infinite;
  }}
  @keyframes bounce {{
    0%, 100% {{ height: 12%; opacity: 0.45; }}
    50%      {{ height: 88%; opacity: 1; }}
  }}
  .script {{ padding: 1.15rem; display: grid; gap: 0.7rem; }}
  .line {{ display: flex; gap: 0.65rem; align-items: flex-start; font-size: 0.92rem; }}
  .line .tag {{
    flex: none; width: 3.6rem; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding-top: 0.22rem; color: var(--on-dark-soft);
  }}
  .line.bot .tag {{ color: var(--signal); }}
  .line p {{ color: var(--on-dark); }}
  .line.bot p {{ color: var(--on-dark); }}
  .booked {{
    margin: 0 1.15rem 1.15rem; padding: 0.75rem 0.9rem; border-radius: 12px;
    background: rgba(47, 191, 168, 0.12); border: 1px solid rgba(47, 191, 168, 0.3);
    font-size: 0.85rem; color: var(--on-dark); display: flex; align-items: center; gap: 0.55rem;
  }}
  .booked svg {{ width: 1rem; height: 1rem; color: var(--signal); flex: none; }}

  /* ---------- Stats ---------- */
  /* Hero and stats are both dark — a hairline keeps them from reading as one block. */
  .stats-band {{ border-top: 1px solid var(--line-dark); padding-block: clamp(2.75rem, 5vw, 4rem); }}
  .stats {{ display: grid; gap: 1.6rem; }}
  @media (min-width: 640px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (min-width: 980px) {{ .stats {{ grid-template-columns: repeat(4, 1fr); }} }}
  .stat-figure {{
    font-family: var(--display); font-size: clamp(2.4rem, 4.5vw, 3.2rem); font-weight: 700;
    line-height: 1; letter-spacing: -0.03em; color: var(--signal);
    font-variant-numeric: tabular-nums;
  }}
  .stat-label {{
    margin-top: 0.5rem; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--on-dark);
  }}
  .stat-note {{ margin-top: 0.5rem; font-size: 0.9rem; color: var(--on-dark-soft); }}

  /* ---------- Steps ---------- */
  .steps {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 1.6rem; counter-reset: s; }}
  @media (min-width: 860px) {{ .steps {{ grid-template-columns: repeat(3, 1fr); gap: 2rem; }} }}
  .step {{ display: flex; gap: 1rem; align-items: flex-start; }}
  .step-index {{
    flex: none; width: 2.3rem; height: 2.3rem; border-radius: 50%;
    display: grid; place-items: center; font-family: var(--display); font-weight: 700;
    font-size: 1rem; color: var(--signal-deep);
    border: 1px solid var(--line-light); background: var(--paper-2);
  }}
  .step h3 {{ margin-bottom: 0.4rem; }}
  .step p {{ color: var(--on-light-soft); font-size: 0.95rem; }}

  /* ---------- Verticals ---------- */
  .verticals {{ display: grid; gap: 1.1rem; }}
  @media (min-width: 800px) {{ .verticals {{ grid-template-columns: repeat(3, 1fr); }} }}
  .vertical {{
    position: relative; padding: 1.6rem 1.4rem; border-radius: 16px;
    background: var(--paper-2); border: 1px solid var(--line-light);
  }}
  .vertical.is-live {{ border-color: rgba(47, 191, 168, 0.55); box-shadow: 0 20px 44px rgba(23, 143, 124, 0.12); }}
  .vertical-mark {{ width: 1.7rem; height: 1.7rem; color: var(--signal-deep); margin-bottom: 0.85rem; }}
  .vertical h3 {{ margin-bottom: 0.45rem; }}
  .vertical p {{ font-size: 0.93rem; color: var(--on-light-soft); }}
  .vocab {{
    margin: 1.1rem 0 0; padding-top: 0.9rem; border-top: 1px dashed var(--line-light);
    display: grid; gap: 0.35rem; font-size: 0.86rem;
  }}
  .vocab > div {{ display: flex; gap: 0.6rem; }}
  .vocab dt {{ width: 3.4rem; flex: none; color: var(--on-light-soft); }}
  .vocab dd {{ margin: 0; font-weight: 600; color: var(--on-light); }}
  .live-tag {{
    position: absolute; top: 1.1rem; right: 1.1rem;
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #05211d; background: var(--signal); padding: 0.2rem 0.5rem; border-radius: 999px;
  }}

  /* ---------- Benefits ---------- */
  .benefits {{ display: grid; gap: 1.4rem 2rem; }}
  @media (min-width: 700px) {{ .benefits {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (min-width: 1000px) {{ .benefits {{ grid-template-columns: repeat(3, 1fr); }} }}
  .benefit {{ padding-top: 1.2rem; border-top: 2px solid var(--signal); }}
  .benefit h3 {{ margin-bottom: 0.45rem; color: var(--on-dark); }}
  .benefit p {{ font-size: 0.94rem; color: var(--on-dark-soft); }}

  /* ---------- Quotes ---------- */
  .quotes {{ display: grid; gap: 1.1rem; }}
  @media (min-width: 860px) {{ .quotes {{ grid-template-columns: repeat(3, 1fr); }} }}
  .quote {{
    margin: 0; padding: 1.5rem 1.35rem; border-radius: 16px;
    background: var(--paper-2); border: 1px solid var(--line-light);
    display: flex; flex-direction: column; gap: 1rem;
  }}
  .quote blockquote {{
    margin: 0; font-family: var(--display); font-size: 1.05rem; line-height: 1.45;
    letter-spacing: -0.012em; color: var(--on-light);
  }}
  .quote figcaption {{ margin-top: auto; font-size: 0.83rem; color: var(--on-light-soft); }}
  .quote-who {{ display: block; font-weight: 700; color: var(--on-light); }}
  .sample-note {{
    margin-top: 1.6rem; font-size: 0.82rem; color: var(--on-light-soft);
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .sample-note b {{
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #5b3d13; background: var(--amber); padding: 0.18rem 0.48rem; border-radius: 999px;
  }}

  /* ---------- Try / examples ---------- */
  .try-grid {{ display: grid; gap: 2rem; align-items: center; }}
  @media (min-width: 880px) {{ .try-grid {{ grid-template-columns: 1fr 1fr; }} }}
  .examples {{ list-style: none; margin: 1.4rem 0 0; padding: 0; display: grid; gap: 0.6rem; }}
  .examples li {{
    padding: 0.8rem 1rem; border-radius: 12px; font-size: 0.95rem;
    background: rgba(47, 191, 168, 0.09); border: 1px solid rgba(47, 191, 168, 0.24);
    color: var(--on-dark);
  }}

  /* ---------- FAQ ---------- */
  .faqs {{ display: grid; gap: 0.6rem; max-width: 52rem; }}
  .faq {{
    border: 1px solid var(--line-light); border-radius: 12px; background: var(--paper-2);
    padding: 0 1.1rem;
  }}
  .faq summary {{
    cursor: pointer; list-style: none; padding: 1.05rem 0; font-weight: 600;
    display: flex; align-items: center; gap: 0.8rem;
  }}
  .faq summary::-webkit-details-marker {{ display: none; }}
  .faq summary::after {{
    content: "+"; margin-left: auto; font-size: 1.3rem; line-height: 1;
    color: var(--signal-deep); transition: transform 0.2s ease;
  }}
  .faq[open] summary::after {{ transform: rotate(45deg); }}
  .faq p {{ padding-bottom: 1.1rem; color: var(--on-light-soft); font-size: 0.95rem; }}

  /* ---------- Final CTA + footer ---------- */
  .final {{ text-align: center; }}
  .final h2 {{ margin-inline: auto; }}
  .final p {{ margin: 1rem auto 0; color: var(--on-dark-soft); }}
  .final .hero-actions {{ justify-content: center; }}
  footer {{
    background: var(--ink); color: var(--on-dark-soft);
    border-top: 1px solid var(--line-dark); padding-block: 2.5rem;
  }}
  .foot-inner {{ display: flex; flex-wrap: wrap; gap: 1rem 2rem; align-items: center; }}
  .foot-links {{ display: flex; flex-wrap: wrap; gap: 1.2rem; margin-left: auto; }}
  .foot-links a {{ color: var(--on-dark-soft); text-decoration: none; font-size: 0.88rem; }}
  .foot-links a:hover {{ color: var(--signal); }}
  .foot-note {{ margin-top: 1.2rem; font-size: 0.78rem; color: rgba(143, 179, 173, 0.7); }}

  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation: none !important; transition: none !important; }}
    html {{ scroll-behavior: auto; }}
  }}
</style>
</head>
<body>

<header class="nav">
  <div class="wrap nav-inner">
    <a class="logo" href="/">
      <svg class="logo-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <rect width="32" height="32" rx="9" fill="#0f3b37"/>
        <g stroke="#2fbfa8" stroke-width="2.4" stroke-linecap="round">
          <path d="M8 13v6"/><path d="M13 9v14"/><path d="M18 11v10"/><path d="M23 14v4"/>
        </g>
      </svg>
      <span class="logo-word">Reception<b>Ai</b></span>
    </a>
    <nav class="nav-links">
      <a href="#how">How it works</a>
      <a href="#industries">Industries</a>
      <a href="#why">Why it works</a>
      <a href="#faq">FAQ</a>
    </nav>
    <a class="btn btn-call btn-sm nav-cta" href="/app/?call=1">
      <span class="ring" aria-hidden="true"></span>Request a call
    </a>
  </div>
</header>

<main>
  <section class="hero">
    <div class="wrap hero-grid">
      <div>
        <span class="status-pill">
          <span class="status-dot" aria-hidden="true"></span>
          Live &middot; answering as {esc(active.label)}
        </span>
        <h1>The front desk that <span>always</span> picks up.</h1>
        <p class="hero-lead">
          ReceptionAi answers every call and message, works out what the caller needs
          from plain speech, and books, moves or cancels it &mdash; without a phone tree
          and without anyone calling them back.
        </p>
        <div class="hero-actions">
          <a class="btn btn-call" href="/app/?call=1">
            <span class="ring" aria-hidden="true"></span>Request a call
          </a>
          <a class="btn btn-ghost" href="/app/">Try the chat instead</a>
        </div>
        <p class="hero-foot">
          Talks to you in the browser &mdash; no signup, no phone number needed.
        </p>
      </div>

      <div class="callcard">
        <div class="callcard-top">
          <span class="status-dot" aria-hidden="true"></span>
          <span class="who">Incoming call</span>
          <span class="callcard-timer">00:12</span>
        </div>
        <div class="wave" aria-hidden="true">
          <i style="animation-delay:-1.05s"></i><i style="animation-delay:-.2s"></i>
          <i style="animation-delay:-.75s"></i><i style="animation-delay:-.45s"></i>
          <i style="animation-delay:-.95s"></i><i style="animation-delay:-.1s"></i>
          <i style="animation-delay:-.6s"></i><i style="animation-delay:-.35s"></i>
          <i style="animation-delay:-.85s"></i><i style="animation-delay:-.5s"></i>
          <i style="animation-delay:-.15s"></i><i style="animation-delay:-.7s"></i>
          <i style="animation-delay:-.3s"></i><i style="animation-delay:-.9s"></i>
          <i style="animation-delay:-.55s"></i><i style="animation-delay:-.25s"></i>
          <i style="animation-delay:-.8s"></i><i style="animation-delay:-.4s"></i>
        </div>
        <div class="script">
          <div class="line"><span class="tag">Caller</span><p>{demo_line}</p></div>
          <div class="line bot"><span class="tag">Desk</span><p>{demo_reply}</p></div>
        </div>
        <div class="booked">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Confirmed and written to your calendar
        </div>
      </div>
    </div>
  </section>

  <section class="band-dark stats-band">
    <div class="wrap">
      <div class="stats">
{stat_items}
      </div>
    </div>
  </section>

  <section id="how" class="band-light">
    <div class="wrap">
      <div class="section-head">
        <p class="eyebrow">How it works</p>
        <h2>Three steps, no phone tree.</h2>
        <p>The caller never presses 1. They just say what they need.</p>
      </div>
      <ol class="steps">
{step_items}
      </ol>
    </div>
  </section>

  <section id="industries" class="band-paper">
    <div class="wrap">
      <div class="section-head">
        <p class="eyebrow">Industries</p>
        <h2>One engine, three front desks.</h2>
        <p>
          Booking a doctor, a table and a room are the same problem: reserving time with
          a named resource. Only the words change &mdash; so the words are all we change.
        </p>
      </div>
      <div class="verticals">
{industry_cards}
      </div>
    </div>
  </section>

  <section id="why" class="band-dark">
    <div class="wrap">
      <div class="section-head">
        <p class="eyebrow">Why it works</p>
        <h2>Built to finish the call, not forward it.</h2>
      </div>
      <div class="benefits">
{benefit_items}
      </div>
    </div>
  </section>

  <section class="band-light">
    <div class="wrap">
      <div class="section-head">
        <p class="eyebrow">From the field</p>
        <h2>What a front desk sounds like afterwards.</h2>
      </div>
      <div class="quotes">
{quote_items}
      </div>
      <p class="sample-note">
        <b>Sample</b> Illustrative quotes for this demo build &mdash; replace with your own before publishing.
      </p>
    </div>
  </section>

  <section class="band-dark">
    <div class="wrap try-grid">
      <div>
        <p class="eyebrow">Try it now</p>
        <h2>Say one of these out loud.</h2>
        <p style="margin-top:0.85rem;color:var(--on-dark-soft)">
          Hit <strong style="color:var(--on-dark)">Request a call</strong>, allow the
          microphone, and talk to it the way a real caller would.
        </p>
        <div class="hero-actions">
          <a class="btn btn-call" href="/app/?call=1">
            <span class="ring" aria-hidden="true"></span>Request a call
          </a>
        </div>
      </div>
      <ul class="examples">
{example_chips}
      </ul>
    </div>
  </section>

  <section id="faq" class="band-paper">
    <div class="wrap">
      <div class="section-head">
        <p class="eyebrow">FAQ</p>
        <h2>The questions front desks ask.</h2>
      </div>
      <div class="faqs">
{faq_items}
      </div>
    </div>
  </section>

  <section class="band-dark final">
    <div class="wrap">
      <h2>Your next caller is about to hang up.</h2>
      <p>Pick up the phone and hear what they'd get instead.</p>
      <div class="hero-actions">
        <a class="btn btn-call" href="/app/?call=1">
          <span class="ring" aria-hidden="true"></span>Request a call
        </a>
        <a class="btn btn-ghost" href="/admin/">Open Admin</a>
      </div>
    </div>
  </section>
</main>

<footer>
  <div class="wrap">
    <div class="foot-inner">
      <a class="logo" href="/">
        <svg class="logo-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <rect width="32" height="32" rx="9" fill="#0f3b37"/>
          <g stroke="#2fbfa8" stroke-width="2.4" stroke-linecap="round">
            <path d="M8 13v6"/><path d="M13 9v14"/><path d="M18 11v10"/><path d="M23 14v4"/>
          </g>
        </svg>
        <span class="logo-word">Reception<b>Ai</b></span>
      </a>
      <div class="foot-links">
        <a href="/app/">Chat &amp; call</a>
        <a href="/app-lite/">Voice desk</a>
        <a href="/admin/">Admin</a>
        <a href="/api/tools">Tools API</a>
        <a href="/api/industry">Industry API</a>
      </div>
    </div>
    <p class="foot-note">
      ReceptionAi &mdash; booking, cancellation and rescheduling for clinics, restaurants
      and hotels. Currently configured as {esc(active.label)}.
    </p>
  </div>
</footer>

</body>
</html>
"""
