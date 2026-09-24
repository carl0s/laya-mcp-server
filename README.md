# Laya MCP

An MCP server for [Laya](https://github.com/NandhaKishorM/laya), the open decision model from ConvAI Innovations. It lets Claude and any other MCP client ask typed questions about a piece of text or JSON and get back a structured answer with probabilities, in tens of milliseconds and without generating text.

It also exposes a small REST endpoint, so tools like n8n can use the same model without speaking MCP.

This is an unofficial project and is not affiliated with ConvAI Innovations.

```
state:     {"body": "I asked for a refund three weeks ago. If I don't hear back by Friday I'm cancelling."}
questions: intent (choice), urgency (score), churn_risk (noul)

answer:    intent = cancel (p 0.71), urgency = high, churn_risk = 0.88   (illustrative)
```

## Read this before relying on it

Laya is fast and cheap, but the base checkpoints are not accurate out of the box on most domains. In our own tests on accessibility checks, the base multilingual model was close to random on qualitative judgments, and some wrong answers came back with a confidence above 0.9. The Laya authors say the same thing in their model card: the checkpoints ship overconfident and should be recalibrated on your data before you trust the probabilities.

In practice:

- Measure it on 50 to 200 examples you have labeled yourself before putting it in any workflow.
- Use `include_all: true` in batch calls to get every answer, not only the uncertain ones, when you measure.
- Fit a temperature per question type with `scripts/fit_temperature.py` and load it with `CALIBRATION_PATH`. Calibration makes the confidence honest. It does not make the model more accurate; for that you need fine-tuning.
- Treat a high confidence as a reason to route automatically only after you have seen, on your data, that high confidence and correct answers go together.

Binary questions with concrete, observable criteria work best. Questions that need to understand the meaning of a text ("does this error message help the user fix the problem?") are where the base model struggles most.

## Tools

| Tool | What it does |
|---|---|
| `laya_classify` | Answers the questions for one state. Returns every answer with probabilities and confidence. |
| `laya_classify_many` | Same questions over up to `MAX_ITEMS` states. Returns counts per option (choice), means (score, noul) and the answers below the confidence threshold. `include_all: true` adds every answer. An `id` field in each item is echoed back. |
| `laya_info` | Loaded checkpoints, limits, threshold and calibration in use. |

All tools are read-only and idempotent.

### Question types

```json
{
  "intent": {
    "type": "choice",
    "instructions": "What does the customer want?",
    "criteria": { "refund": "Wants money back", "cancel": "Wants to cancel", "other": "Anything else" }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["low", "medium", "high", "critical"]
  },
  "churn_risk": {
    "type": "noul",
    "instructions": "The customer is threatening to leave."
  }
}
```

- `choice` picks one option. `criteria` is an object from option to description. Keep it under about 20 options.
- `score` places the state on an ordered scale. `criteria` is a list of levels. It is the weakest of the three types.
- `noul` returns the probability that the instruction is true for the state.

A full example is in `examples/questions.example.json`.

## Quick start

You need Docker and a machine with at least 4 GB of free RAM. No GPU is required.

```bash
git clone https://github.com/OWNER/laya-mcp.git
cd laya-mcp
cp .env.example .env
# set API_TOKENS to the output of: openssl rand -hex 32
docker compose up -d
docker compose logs -f   # the first start downloads about 1.3 GB of weights
```

When `curl localhost:8000/healthz` returns `{"status": "ok"}`, the server is ready. The MCP endpoint is `http://localhost:8000/mcp`.

Add it to Claude Code:

```bash
claude mcp add --transport http laya http://localhost:8000/mcp \
  --header "Authorization: Bearer YOUR_TOKEN"
```

## Authentication

Pick one mode with `AUTH_MODE`.

| Mode | Use it for | How clients authenticate |
|---|---|---|
| `bearer` (default) | Claude Code, n8n, scripts, any client that can send a header | `Authorization: Bearer <token>`, tokens listed in `API_TOKENS` |
| `github` | claude.ai custom connectors, which authenticate with OAuth | GitHub login, restricted to `ALLOWED_GITHUB_USERS` |
| `none` | Local development on `127.0.0.1` only | Nothing. The server logs a warning if it listens on another address. |

### Setting up GitHub OAuth for claude.ai

1. On GitHub, go to Settings, Developer settings, **OAuth Apps** (not GitHub Apps), New OAuth App.
2. Homepage URL: your public URL, for example `https://laya.example.com`. Authorization callback URL: the same URL followed by `/auth/callback`.
3. Copy the client ID, generate a client secret, and put both in `.env` together with `BASE_URL`, `ALLOWED_GITHUB_USERS` (GitHub usernames, comma separated) and `APP_SECRET` (`openssl rand -hex 32`).
4. In claude.ai, add a custom connector with the URL `https://laya.example.com/mcp`.

GitHub lets any account complete the login, so the allowlist is enforced on every tool call. OAuth client registrations and tokens are stored encrypted in `DATA_DIR` and survive restarts as long as the volume does. Do not change `APP_SECRET` after the first start, or every client will have to reconnect.

In `github` mode the REST endpoint is disabled unless you also set `API_TOKENS`.

## Deploy

The server is a single Docker image that needs roughly:

- RAM: 2.5 to 3 GB per loaded checkpoint, plus about 1 GB. One checkpoint fits in 4 GB.
- CPU: 2 vCPU are enough for interactive use. Set `OMP_NUM_THREADS` below the core count on shared hosts.
- Disk: about 5 GB for the image and the weights.
- A persistent volume on `/data`, so weights and OAuth state are not lost on restart.
- HTTPS in front of it if clients reach it over the internet.

Every push to `main` publishes an image to `ghcr.io/OWNER/laya-mcp` through the included GitHub Actions workflow, so you can deploy either from the repo or from the image.

### Any Docker host

Use `docker-compose.yml` as shown in the quick start, behind the reverse proxy you already have (Caddy, Traefik, nginx). The MCP transport is stateless and returns plain JSON, so it works behind proxies that do not handle streaming well.

### Hugging Face Spaces (free CPU tier)

1. Create a Space with the **Docker** SDK.
2. Push this repository to it and add this front matter at the top of the Space's `README.md`:

   ```yaml
   ---
   title: Laya MCP
   sdk: docker
   app_port: 8000
   ---
   ```

3. Set the variables from `.env.example` as Space secrets, with `DATA_DIR=/data` and `HF_HOME=/data/hf`.
4. The endpoint is `https://<user>-<space>.hf.space/mcp`.

Limits of the free tier: the Space sleeps after a period of inactivity, the first request after that waits for a cold start, and without paid persistent storage the weights are downloaded again and OAuth registrations are lost at every restart. With `AUTH_MODE=github` that means reconnecting in claude.ai. `bearer` mode has no such problem.

### CapRover

The repo includes a `captain-definition`, so you can create an app and deploy it from the repository with the "Deploy from Github" method, or deploy the published image.

In the app settings:

- Container HTTP port: `8000`.
- Persistent directory: `/data`.
- Environment variables from `.env.example`.
- In HTTP Settings, edit the nginx configuration and add these lines inside the `location /` block. CapRover proxies with HTTP/1.0 by default, which breaks MCP clients.

  ```nginx
  proxy_http_version 1.1;
  proxy_set_header Connection "";
  proxy_buffering off;
  proxy_read_timeout 120s;
  ```

Check that the host has enough free memory: a container killed with exit code 137 was out of memory.

### Other platforms

Any platform that builds a Dockerfile, gives you a persistent volume and at least 4 GB of RAM will work. There is nothing platform-specific in the image.

### Without Docker

```bash
uv venv --python 3.11
uv pip install torch -r requirements.txt   # add --index-url https://download.pytorch.org/whl/cpu for CPU-only torch on Linux
cp .env.example .env                      # set DATA_DIR=./data and HF_HOME=./data/hf
set -a; source .env; set +a
.venv/bin/python server.py
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AUTH_MODE` | `bearer` | `none`, `bearer` or `github` |
| `API_TOKENS` | | Comma-separated bearer tokens |
| `BASE_URL` | | Public URL of the server, required in `github` mode |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | | GitHub OAuth App credentials |
| `ALLOWED_GITHUB_USERS` | | GitHub usernames allowed to use the tools |
| `APP_SECRET` | | Signs tokens and encrypts OAuth state. Keep it stable. |
| `LAYA_CHECKPOINTS` | `multilingual` | `multilingual`, `english`, `typed-decisions`, comma separated. Use `multilingual` for anything that is not English. |
| `MAX_ITEMS` | `50` | Maximum items per batch call |
| `LOW_CONFIDENCE` | `0.7` | Answers below this are reported in `low_confidence` |
| `LOW_CONFIDENCE_FIELD` | `answer_confidence` | Field used when no calibration is loaded: `answer_confidence` or the stricter `confidence` |
| `CALIBRATION_PATH` | | JSON file with temperatures |
| `OMP_NUM_THREADS` | | CPU threads for inference |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Listen address |
| `DATA_DIR` | `./data` (`/data` in Docker) | OAuth state and anything else the server persists |
| `HF_HOME` | | Where the model weights are cached |
| `LOG_LEVEL` | `INFO` | |

## Calibration

1. Label some data yourself: at least 50 examples per question type, ideally a few hundred, with both outcomes represented.
2. Run them through `laya_classify_many` or `POST /v1/classify` with `include_all: true`.
3. Write one JSONL line per answered question:

   ```json
   {"type": "choice", "probabilities": {"pass": 0.91, "fail": 0.09}, "expected": "fail"}
   {"type": "noul", "noul": 0.83, "expected": true}
   ```

4. Fit the temperatures and load them:

   ```bash
   python scripts/fit_temperature.py labeled.jsonl > data/calibration.json
   # then set CALIBRATION_PATH=/data/calibration.json and restart
   ```

The script prints accuracy and expected calibration error before and after, per group. Keep a separate test set that you never use for fitting, and pick your routing thresholds on that one.

With calibration loaded, every answer gets a `calibrated` block next to the original fields, and the low-confidence threshold applies to `calibrated.answer_confidence`.

## REST API

`POST /v1/classify` with a bearer token.

```bash
curl -X POST https://laya.example.com/v1/classify \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d @examples/questions.example.json
```

Send `state` for one item, or `items` (plus optional `include_all`) for a batch. The responses have the same shape as the MCP tools. `GET /healthz` is open and returns 503 while the model is loading.

## Contributing

Issues and pull requests are welcome, especially:

- measured accuracy on public datasets, in languages other than English;
- calibration files for common question shapes;
- deployment notes for platforms not covered here.

## Credits and license

Laya is developed by [ConvAI Innovations](https://huggingface.co/convaiinnovations/laya) and released under Apache 2.0. This server is also released under the Apache License 2.0, see `LICENSE`.

Built on [FastMCP](https://github.com/jlowin/fastmcp).
