#!/usr/bin/env bash
# Source this from PBS jobs to configure proxy variables without storing
# credentials in the repository.
#
# Supported inputs:
#   PROXY_URL=http://user:pass@proxy.example.com:8080
# or:
#   PROXY_HOST=proxy.example.com
#   PROXY_PORT=8080
#   PROXY_USER=<username>
#   PROXY_PASS=<password>
#   PROXY_SCHEME=http
#
# If PROXY_USER or PROXY_PASS contains URL-special characters, set PROXY_URL
# directly with those values URL-encoded.
#
# PBS jobs use `#PBS -V`; the cluster's working HTTP_PROXY/HTTPS_PROXY is
# therefore normally inherited from the submission shell. Preserve that
# established path, but normalize shorthand such as `proxy:8080` before
# requests/httpx sees it.

# Idempotent: PBS wrappers source this before invoking the reusable .sh
# runners, while users may invoke a runner directly on a compute node.
if [[ "${PROXY_CONFIG_INITIALIZED:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
unset PROXY_CONFIG_VALID

if [[ -n "${PROXY_URL:-}" ]]; then
    proxy_url="$PROXY_URL"
elif [[ -n "${PROXY_HOST:-}" ]]; then
    proxy_scheme="${PROXY_SCHEME:-http}"
    proxy_host="$PROXY_HOST"

    if [[ "$proxy_host" == *"://"* ]]; then
        proxy_base="$proxy_host"
    else
        proxy_base="${proxy_scheme}://${proxy_host}"
    fi

    if [[ -n "${PROXY_PORT:-}" && "$proxy_base" != *":${PROXY_PORT}" ]]; then
        proxy_base="${proxy_base}:${PROXY_PORT}"
    fi

    if [[ -n "${PROXY_USER:-}" || -n "${PROXY_PASS:-}" ]]; then
        proxy_url="${proxy_base/:\/\//:\/\/${PROXY_USER:-}:${PROXY_PASS:-}@}"
    else
        proxy_url="$proxy_base"
    fi
elif [[ -n "${https_proxy:-}${http_proxy:-}${HTTPS_PROXY:-}${HTTP_PROXY:-}" ]]; then
    proxy_url="${https_proxy:-${http_proxy:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}}"
else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export PROXY_CONFIG_INITIALIZED=1
    echo "[proxy] disabled (set PROXY_URL/PROXY_HOST explicitly if required)"
    return 0 2>/dev/null || exit 0
fi

# Avoid a stale SOCKS/all-protocol proxy taking precedence in some clients.
unset all_proxy ALL_PROXY

proxy_python=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c "import socket, urllib.parse" >/dev/null 2>&1; then
        proxy_python="$candidate"
        break
    fi
done

if [[ -n "$proxy_python" ]]; then
    if ! normalized_proxy_url=$(PROXY_CHECK_URL="$proxy_url" "$proxy_python" - <<'PY'
import os
import sys
from urllib.parse import urlsplit

url = os.environ["PROXY_CHECK_URL"]
if any(char.isspace() for char in url):
    print("ERROR: proxy URL contains whitespace.", file=sys.stderr)
    raise SystemExit(1)
parsed = urlsplit(url if "://" in url else f"http://{url}")
if parsed.scheme not in {"http", "https"}:
    print(f"ERROR: unsupported proxy scheme: {parsed.scheme!r}", file=sys.stderr)
    raise SystemExit(1)
host = parsed.hostname
try:
    _ = parsed.port
except ValueError as exc:
    print(f"ERROR: invalid proxy port: {exc}", file=sys.stderr)
    raise SystemExit(1)
if not host:
    print("ERROR: proxy URL has no hostname.", file=sys.stderr)
    raise SystemExit(1)
print(parsed.geturl())
PY
    ); then
        echo "ERROR: invalid proxy configuration." >&2
        echo "Submit with PROXY_URL=http://<host>:<port> or correct PROXY_HOST." >&2
        return 1 2>/dev/null || exit 1
    fi
    # The validation step also normalizes shorthand such as "proxy:8080" to
    # a URI with an explicit scheme. requests/httpx reject the shorthand even
    # though urllib's resolver check can parse it.
    proxy_url="$normalized_proxy_url"
fi

export http_proxy="$proxy_url"
export https_proxy="$proxy_url"
export HTTP_PROXY="$proxy_url"
export HTTPS_PROXY="$proxy_url"
export PROXY_CONFIG_VALID=1
export PROXY_CONFIG_INITIALIZED=1

proxy_no_proxy="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export no_proxy="$proxy_no_proxy"
export NO_PROXY="$proxy_no_proxy"

proxy_display="${proxy_url#*://}"
proxy_display="${proxy_display#*@}"
proxy_display="${proxy_display%%/*}"
echo "[proxy] enabled: ${proxy_display}"

# --- diagnostics -----------------------------------------------------------
# Set PROXY_DEBUG=0 to silence. Probes are informational and never fail the job.
if [[ "${PROXY_DEBUG:-1}" == "1" ]]; then
    masked_url="$(printf '%s' "$proxy_url" | sed -E 's#(://[^:/@]+:)[^@]*@#\1****@#')"
    echo "[proxy] http_proxy=${masked_url}"
    echo "[proxy] no_proxy=${no_proxy:-}"
    if [[ -n "${PROXY_URL:-}" ]]; then proxy_src="PROXY_URL"
    elif [[ -n "${PROXY_HOST:-}" ]]; then proxy_src="PROXY_HOST"
    else proxy_src="inherited_env"; fi
    echo "[proxy] source=${proxy_src}"
    echo "[proxy-check] node hostname: $(hostname)"

    proxy_targets="${PROXY_TARGETS:-https://huggingface.co https://pypi.org https://download.pytorch.org}"
    proxy_host_only="$(printf '%s' "$proxy_display")"

    # DNS resolution of the proxy host.
    if command -v getent >/dev/null 2>&1; then
        if getent hosts "${proxy_host_only%%:*}" >/dev/null 2>&1; then
            echo "[proxy-check] DNS resolve ${proxy_host_only%%:*}: OK ($(getent hosts "${proxy_host_only%%:*}" | awk '{print $1}' | tr '\n' ' '))"
        else
            echo "[proxy-check] DNS resolve ${proxy_host_only%%:*}: FAILED (name not resolvable from this node)"
        fi
    fi

    # curl gives the clearest failure signal (exit code + HTTP status).
    #   exit 5 = cannot resolve proxy | 7 = cannot connect to proxy (port/firewall)
    #   exit 28 = timeout (often: this node has no route out) | http_code 407 = auth
    if command -v curl >/dev/null 2>&1; then
        for tgt in $proxy_targets; do
            out=$(curl -sS -x "$proxy_url" -o /dev/null \
                  -w 'http_code=%{http_code} time=%{time_total}s' \
                  --max-time 15 "$tgt" 2>&1) ; rc=$?
            echo "[proxy-check] curl via proxy $tgt -> ${out} curl_exit=${rc}"
        done
        # Direct (no proxy) egress test: does this node reach the internet at all?
        direct=$(curl -sS --noproxy '*' -o /dev/null \
                 -w 'http_code=%{http_code} time=%{time_total}s' \
                 --max-time 10 https://huggingface.co 2>&1) ; drc=$?
        echo "[proxy-check] curl DIRECT (no proxy) huggingface.co -> ${direct} curl_exit=${drc}"
    elif [[ -n "$proxy_python" ]]; then
        PROXY_CHECK_URL="$proxy_url" \
        PROXY_TARGETS="$proxy_targets" \
        "$proxy_python" - <<'PY' || true
import os, socket, time
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler, Request

purl = os.environ["PROXY_CHECK_URL"]
p = urlsplit(purl if "://" in purl else "http://" + purl)
host, port = p.hostname, p.port or (443 if p.scheme == "https" else 80)
print(f"[proxy-check] proxy host={host} port={port} scheme={p.scheme} "
      f"user={'set' if p.username else 'none'}")

t = time.time()
try:
    with socket.create_connection((host, port), timeout=8):
        print(f"[proxy-check] TCP connect to proxy OK ({time.time()-t:.1f}s)")
except Exception as e:
    print(f"[proxy-check] TCP connect to proxy FAILED ({time.time()-t:.1f}s): "
          f"{type(e).__name__}: {e}")

opener = build_opener(ProxyHandler({"http": purl, "https": purl}))
for tgt in os.environ["PROXY_TARGETS"].split():
    t = time.time()
    try:
        r = opener.open(Request(tgt, method="HEAD"), timeout=12)
        print(f"[proxy-check] {tgt} via proxy OK status={r.status} "
              f"({time.time()-t:.1f}s)")
        r.close()
    except Exception as e:
        print(f"[proxy-check] {tgt} via proxy FAILED ({time.time()-t:.1f}s): "
              f"{type(e).__name__}: {e}")
PY
    else
        echo "[proxy-check] no python found for connectivity probe"
    fi
fi
