## Hospital Management LangGraph Agent

This project provides a simple LangGraph ReAct agent that uses Groq's `llama3-8b-8192` model and two tools:

- `book_appointment(patient_name, doctor, time)`
- `cancel_appointment(appointment_id)`

Both tools operate on an in-memory store for demonstration purposes.

### Prerequisites
- Python 3.10+
- A Groq API key with access to `llama3-8b-8192`

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key_here"
```

### Run
Interactive mode:
```bash
python Main.py
```

One-off prompt:
```bash
python Main.py "Book an appointment for John Doe with Dr. Smith at 3pm tomorrow"
```

Example prompts you can try interactively:
- "Book an appointment for Jane Roe with Dr. House at 10:00 on Friday"
- "Cancel appointment APT-0001"

Note: The storage is in-memory and will reset each time the process restarts.

