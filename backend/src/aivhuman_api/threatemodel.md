## Threat model

### Who would attack it?
- **A person who's work is being reviewed** who wants an AI-written document judged human-made, so it gets past a reviewer using this tool.
- **A disruptor** (e.g a student under deadline pressure) who wants the checker unavailable, so work can't be checked at all.
- **A curious or malicious user** who wants to read other people's past checks, which may contain their work.

### What are they trying to achieve?
- **Evasion:** get AI text scored low, by paraphrasing, mixing in human sentences, or swapping letters for Unicode lookalikes.
- **Denial of service:** make the server stop answering, which is cheap to cause because one GPT-2 scoring request costs far more to handle than to send.
- **Data theft:** pull other sessions' stored essays out of the check history.

### What can they actually do?
- Send any text up to any size, as often as they like, directly to the API with curl. The front-end checks don't stop them.
- Send many expensive requests at once (up to 100,000 characters each).
- Put crafted strings in the history search (`q`) and the `X-Session-Id` header.
- They can't upload files, make the server fetch URLs, run commands, or log in as anyone, because none of those features exist.

### Which risks apply

| Risk (section 2.7) | Applies? | Reason |
|---|---|---|
| Denial of service | **Yes, defended** | Inference is slow, so blocking the event loop or flooding it freezes the app for everyone. |
| SQL injection | **Yes, defended** | Checks are stored in SQLite and searchable by text. |
| Unsafe model loading | Yes, not defended | `torch.load(weights_only=False)` uses pickle. We trained the file ourselves and ship it in the repo, so the risk is low. Next step: a SHA-256 check at startup. |
| Missing rate limiting | Yes, not defended | One client can probe the model with requests sent one at a time. The DoS gate caps concurrency, not request rate. |
| Misconfiguration / dependencies | Yes, not defended | Applies to every app. Versions are pinned, but there's no ZAP scan. |
| Command injection | No | The backend never runs a system command. |
| SSRF | No | The backend never makes a request based on user input. |
| Unsafe file upload | No | Input is pasted text only. Body size is still capped at 2 MB. |
| Broken access control | No | There are no user accounts. History is scoped to a random session ID. Anyone holding that ID can read it, which is acceptable because there's no login and the IDs can't be guessed. |

### How this shaped our choices
DoS is the cheapest attack with the widest impact, because one person can take the checker offline for every user. That made it our first defence. The history feature is the only place untrusted text reaches a database, and leaking it exposes other people's essays, so SQL injection was the second. We haven't defended against evasion, which is the student's main goal. We list it under limitations, because it's a model problem rather than an application one.