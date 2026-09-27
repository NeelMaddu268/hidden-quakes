#!/usr/bin/env bash
# Copy audit for reader-facing text (ticket DEMO-03, H4).
#
#   scripts/check-copy.sh                 scan the default set (below)
#   scripts/check-copy.sh FILE...         scan these files instead
#
# CLAUDE.md rule 4: never write a count, magnitude, date, station code or percentage as a fact
# into UI copy, pitch text or docs; numbers render from pipeline output. docs/00-project.md
# lists the phrases we never use. This script greps for both and exits nonzero on any hit.
#
# Default set: README.md, docs/demo/*.md, and the H4 web copy under apps/web/src/app and
# apps/web/src/shell (.ts/.tsx, tests and .d.ts excluded).
#
# Digit rule (markdown): a line is an offender if any digit survives after removing
#   - fenced code blocks (``` ... ```) and inline code (`...`)
#   - placeholders: {name} and <from file: field> / <anything in angle brackets>
#   - URLs, ticket ids (ABC-12), EPSG codes, file names and paths with an extension,
#     docs/NN references, lane and priority ids (H1, P0), HackGT 13, 1D/2D/3D, 3DEP,
#     UTM zone 12N, CC BY licences, list markers (1. ), pitch timestamps (0:10), numbered
#     table index columns (| 12 |), GitHub handles, question references (Q27), pitch formats
#     ("2-minute pitch")
#   - years in citations (19xx/20xx) inside docs/ only, never in README.md
# A line ending in the marker <!-- copy-ok --> is exempt from both rules (use it only for a
# sentence that names a forbidden phrase in order to reject it) and is counted in the summary.
#
# Digit rule (tsx): JSX text nodes (text between > and < with no braces) and readable
# attributes (aria-label, title, alt, placeholder). The authoritative check for the shell is
# apps/web/src/shell/copy.test.ts, which parses the files; this is the grep-level twin so
# docs and code get one report.
#
# Forbidden phrases (case-insensitive, every scanned line): "confirmed earthquake",
# "caused by", "predict" (except the seismological "predicted arrival/P/S/time/travel"),
# "official", operator attribution ("induced by FORGE", "FORGE's fractures", "due to Cape
# Station", "attributed to the operator", ...), "scientists missed", "discovered earthquakes",
# "FORGE didn't know".
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ $# -gt 0 ]]; then
  files=("$@")
else
  files=(README.md)
  while IFS= read -r f; do files+=("$f"); done < <(ls docs/demo/*.md 2>/dev/null)
  while IFS= read -r f; do files+=("$f"); done < <(
    find apps/web/src/app apps/web/src/shell -type f \( -name '*.ts' -o -name '*.tsx' \) \
      ! -name '*.test.ts' ! -name '*.test.tsx' ! -name '*.d.ts' 2>/dev/null | sort
  )
  # REQ-H3-15 (c): the guided tour's captions, whose one copy file H3 handed to H4.
  [[ -f apps/web/src/scene/tour/copy.ts ]] && files+=(apps/web/src/scene/tour/copy.ts)
fi

# One ERE for every forbidden phrase; matched case-insensitively against each scanned line.
operators='(utah )?(forge|cape station|the operator|an operator|operators?)'
forbidden="confirmed earthquake"
forbidden+="|caused by"
forbidden+="|predict"
forbidden+="|\\bofficial\\b"
forbidden+="|(induced|triggered|generated|created) by ${operators}"
forbidden+="|forge'?s (fractures?|faults?|earthquakes?|quakes?|events?|seismicity|activity)"
forbidden+="|(due to|because of|attributed to|attributable to|blamed on) ${operators}"
forbidden+="|mapped forge"
forbidden+="|forge (didn'?t|did not|doesn'?t|does not) know"
forbidden+="|scientists missed"
forbidden+="|discovered earthquakes"

status=0
offenders=0
exempt=0

# Prints "file:line: reason: text" for one file; the awk exit code is 1 when it flagged anything.
scan() {
  local file="$1" kind="$2"
  awk -v FILE="$file" -v KIND="$kind" -v FORBIDDEN="$forbidden" -v INDOCS="$( [[ $file == docs/* ]] && echo 1 || echo 0 )" '
    function report(reason, text) {
      gsub(/^[ \t]+|[ \t]+$/, "", text)
      printf "%s:%d: %s: %s\n", FILE, NR, reason, text
      flagged++
    }
    function strip_allowed(s) {
      gsub(/`[^`]*`/, " ", s)                                  # inline code
      gsub(/https?:\/\/[^ )>\]]+/, " ", s)                      # URLs
      gsub(/<[^<>]*>/, " ", s)                                  # <placeholders>, html tags
      gsub(/\{[^{}]*\}/, " ", s)                                # {placeholders}
      gsub(/[A-Z][A-Z]+-[0-9]+/, " ", s)                        # ticket ids
      gsub(/EPSG:[0-9]+/, " ", s)                               # EPSG codes
      gsub(/[A-Za-z0-9_.\/()-]+\.(md|py|ts|tsx|js|yaml|yml|json|csv|sh|parquet|quakeml|png|nc|txt|xml|mseed|toml|lock)/, " ", s)  # file names
      gsub(/docs\/[0-9][0-9]/, " ", s)                          # docs/00 references
      gsub(/(^|[^A-Za-z0-9])[HP][0-9]([^A-Za-z0-9]|$)/, " ", s) # lane and priority ids
      gsub(/HackGT 13/, " ", s)
      gsub(/(^|[^A-Za-z0-9])[123]D([^A-Za-z0-9]|$)/, " ", s)   # 1D / 2D / 3D
      gsub(/3DEP/, " ", s)
      gsub(/UTM( zone)? 12N/, " ", s)
      gsub(/CC[- ]BY([- ][A-Z]+)* [0-9]\.[0-9]/, " ", s)        # licences
      gsub(/^[ \t]*[0-9]+\.[ \t]/, " ", s)                      # ordered-list markers
      gsub(/[0-9]+:[0-9][0-9]/, " ", s)                         # pitch timestamps (0:10)
      gsub(/^[ \t]*\|[ \t]*[0-9]+[ \t]*\|/, " ", s)              # numbered table index column (| 12 |)
      gsub(/@[A-Za-z0-9-]+/, " ", s)                             # GitHub handles
      gsub(/(^|[^A-Za-z0-9])Q[0-9]+([^A-Za-z0-9]|$)/, " ", s)    # question references (Q27)
      gsub(/[0-9]+-(second|minute) (pitch|video|flow)/, " ", s)  # pitch formats
      if (INDOCS == 1) gsub(/(^|[^A-Za-z0-9])(19|20)[0-9][0-9]([^A-Za-z0-9]|$)/, " ", s)  # years in citations
      return s
    }
    function check_forbidden(s,   t) {
      t = tolower(s)
      gsub(/predicted (arrivals?|p|s|times?|travel)[a-z-]*/, " ", t)
      if (t ~ FORBIDDEN) report("forbidden phrase", s)
    }
    BEGIN { fence = 0; flagged = 0; exempt = 0 }
    {
      line = $0
      if (line ~ /<!-- *copy-ok *-->/) { exempt++; next }
      if (KIND == "md") {
        if (line ~ /^[ \t]*```/) { fence = !fence; next }
        if (fence) next
        check_forbidden(line)
        rest = strip_allowed(line)
        if (rest ~ /[0-9]/) report("digit outside code or placeholder", line)
      } else {
        check_forbidden(line)
        # JSX text nodes: between a closing > and the next <, with no braces (expressions are data).
        s = line
        while (match(s, />[^<>{}]*[0-9][^<>{}]*</)) {
          seg = substr(s, RSTART + 1, RLENGTH - 2)
          if (seg ~ /[A-Za-z]/) report("digit in JSX text", seg)
          s = substr(s, RSTART + RLENGTH - 1)
        }
        if (match(line, /(aria-label|title|alt|placeholder|aria-description)="[^"]*[0-9][^"]*"/)) {
          report("digit in readable attribute", substr(line, RSTART, RLENGTH))
        }
      }
    }
    END {
      if (exempt > 0) printf "%s: %d line(s) exempted by <!-- copy-ok -->\n", FILE, exempt > "/dev/stderr"
      exit (flagged > 0 ? 1 : 0)
    }
  ' "$file"
}

for file in "${files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "check-copy: no such file: $file" >&2
    status=2
    continue
  fi
  case "$file" in
    *.md) kind=md ;;
    *.ts|*.tsx) kind=tsx ;;
    *) kind=md ;;
  esac
  out="$(scan "$file" "$kind" 2>/tmp/check-copy.$$)"
  rc=$?
  cat /tmp/check-copy.$$ >&2
  rm -f /tmp/check-copy.$$
  if [[ -n "$out" ]]; then
    echo "$out"
    n=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
    offenders=$((offenders + n))
  fi
  [[ $rc -eq 0 ]] || status=1
done

if [[ $offenders -gt 0 ]]; then
  echo "check-copy: $offenders offending line(s) in ${#files[@]} file(s)" >&2
else
  echo "check-copy: clean (${#files[@]} file(s))" >&2
fi
exit "$status"
