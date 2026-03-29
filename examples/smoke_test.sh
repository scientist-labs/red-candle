#!/usr/bin/env bash
set -uo pipefail

# Smoke test runner — executes each test suite in its own process
# for isolated memory management. Run all suites or pass names:
#
#   ./examples/smoke_test.sh              # run all
#   ./examples/smoke_test.sh core llm     # run specific suites
#   ./examples/smoke_test.sh --list       # list available suites

DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_SUITES=(core tokenizer embedding reranker ner structured tools llm vlm)

if [[ "${1:-}" == "--list" ]]; then
  echo "Available suites:"
  for suite in "${ALL_SUITES[@]}"; do
    echo "  $suite"
  done
  exit 0
fi

SUITES=("${@:-${ALL_SUITES[@]}}")

total_passed=0
total_failed=0
suite_results=()

for suite in "${SUITES[@]}"; do
  script="$DIR/smoke_test_${suite}.rb"
  if [[ ! -f "$script" ]]; then
    echo "Unknown suite: $suite (no $script)"
    suite_results+=("❌ $suite (not found)")
    total_failed=$((total_failed + 1))
    continue
  fi

  echo
  echo "╔══════════════════════════════════════════════════════════════════════════════╗"
  echo "║  Suite: $suite"
  echo "╚══════════════════════════════════════════════════════════════════════════════╝"
  echo

  if bundle exec ruby "$script"; then
    suite_results+=("✅ $suite")
    total_passed=$((total_passed + 1))
  else
    suite_results+=("❌ $suite")
    total_failed=$((total_failed + 1))
  fi
done

echo
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  SMOKE TEST SUMMARY"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
for result in "${suite_results[@]}"; do
  printf "║  %-74s ║\n" "$result"
done
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
printf "║  Suites passed: %-3d  failed: %-3d  total: %-3d                              ║\n" \
  "$total_passed" "$total_failed" "$((total_passed + total_failed))"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

exit $((total_failed > 0 ? 1 : 0))
