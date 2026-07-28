#!/bin/sh
set -eu

schema=sim.interference-acceptance/v1
profiles=acceptance/profiles-v1.psv

usage() {
  echo "usage: $0 capture --source COMMIT --target CAPABILITY --output FILE | verify --source COMMIT FILE" >&2
  exit 2
}

field() {
  name=$1
  value=$2
  printf '  (%s "%s")\n' "$name" "$value"
}

profile_for() {
  requested=$1
  awk -F '|' -v target="$requested" '
    $1 == target {
      for (i = 1; i <= NF; i++) printf "%s%s", $i, (i == NF ? ORS : FS)
      found = 1
    }
    END { if (!found) exit 1 }
  ' "$profiles"
}

verify() {
  source=$1
  artifact=$2
  test "$(sed -n '1p' "$artifact")" = "($schema"
  test "$(sed -n '$p' "$artifact")" = ")"
  grep -Fq "  (source \"$source\")" "$artifact"
  target=$(sed -n 's/^  (target "\([^"]*\)")$/\1/p' "$artifact")
  case "$target" in
    5080-laptop|5090|ryzen-ai-max-395) ;;
    *) echo "acceptance artifact has an unknown target" >&2; exit 1 ;;
  esac
  for required_field in profile adapter backend driver power thermal \
    crossover_cells portable_artifact_sha256 vendor_artifact_sha256; do
    grep -Eq "^  \\($required_field \"[^\"]+\"\\)$" "$artifact"
  done
  grep -Fq '  (evidence "physical-device")' "$artifact"
  grep -Fq '  (segmented_memory "true")' "$artifact"
  grep -Fq '  (result "measured-pass")' "$artifact"
  for case_id in attenuated-multi-segment long-world-distance exact-cancellation; do
    case_line=$(grep -F "(case (id \"$case_id\")" "$artifact")
    test "$(printf '%s\n' "$case_line" | wc -l)" -eq 1
    for required_metric in cells sources tiles segments repeats max_psi \
      component_tolerance phase_tolerance max_component_abs max_phase_abs; do
      printf '%s\n' "$case_line" | grep -Eq "\\($required_metric \"[0-9.e+-]+\"\\)"
    done
    printf '%s\n' "$case_line" | grep -Fq '(repeats "100")'
    printf '%s\n' "$case_line" | grep -Fq '(deterministic "true")'
    printf '%s\n' "$case_line" | grep -Fq '(determinism_required "true")'
    printf '%s\n' "$case_line" |
      grep -Fq '(intermediate_materializations "0")'
    printf '%s\n' "$case_line" |
      grep -Fq '(final_materializations "2")'
    printf '%s\n' "$case_line" | grep -Fq '(result "pass")'
  done
  grep -Fq '(projection (observables "amplitude,phase,magnitude-squared") (result "pass"))' "$artifact"
  grep -Fq '(failure (explicit_unavailable "pass") (automatic_cpu_choice "pass") (result "pass"))' "$artifact"
  if test "$target" = 5090 || test "$target" = ryzen-ai-max-395; then
    crossover_line=$(grep -F '(case (id "above-5080-crossover")' "$artifact")
    test "$(printf '%s\n' "$crossover_line" | wc -l)" -eq 1
    printf '%s\n' "$crossover_line" | grep -Fq '(cells "16641")'
    printf '%s\n' "$crossover_line" | grep -Fq '(repeats "3")'
    printf '%s\n' "$crossover_line" |
      grep -Eq '\(deterministic "(true|false)"\)'
    printf '%s\n' "$crossover_line" |
      grep -Fq '(determinism_required "false")'
    printf '%s\n' "$crossover_line" |
      grep -Fq '(intermediate_materializations "0")'
    printf '%s\n' "$crossover_line" |
      grep -Fq '(final_materializations "2")'
    printf '%s\n' "$crossover_line" | grep -Fq '(result "pass")'
    test "$(sed -n 's/^  (crossover_cells "\([0-9][0-9]*\)")$/\1/p' "$artifact")" -lt 16641
  fi
  count=$(grep -Fc '(result "pass"))' "$artifact")
  expected=5
  if test "$target" = 5090 || test "$target" = ryzen-ai-max-395; then
    expected=6
  fi
  test "$count" -eq "$expected"
  if grep -Eiq "hostname|username|user=|serial|uuid|/home/|\\\\" "$artifact"; then
    echo "acceptance artifact contains private identity data" >&2
    exit 1
  fi
}

capture() {
  source=$1
  target=$2
  output=$3
  profile=$(profile_for "$target")
  IFS='|' read -r capability public_target adapter profile_id backend driver power thermal crossover portable_sha vendor_sha <<EOF
$profile
EOF
  test "$capability" = "$target"

  raw="${output}.raw"
  mkdir -p "$(dirname "$output")"
  SIM_INTERFERENCE_WGPU_PHYSICAL=1 \
    SIM_INTERFERENCE_ACCEPTANCE_TARGET="$target" \
    cargo test -q -p sim-lib-interference-compute \
      hardware_tests::wgpu_matrix_repeats_same_profile_one_hundred_times_when_opted_in \
      -- --exact --nocapture --test-threads=1 >"$raw"
  cargo test -q -p sim-lib-interference-compute \
    hardware_tests::explicit_wgpu_absence_and_auto_cpu_choice_are_pre_submission \
    -- --exact --test-threads=1 >/dev/null
  cargo test -q -p sim-lib-interference-solve --test projection_conformance \
    -- --test-threads=1 >/dev/null
  measured=$(sed -n 's/^acceptance-case=//p' "$raw")
  test -n "$measured"

  {
    echo "($schema"
    field source "$source"
    field target "$public_target"
    field profile "$profile_id"
    field adapter "$adapter"
    field backend "$backend"
    field driver "$driver"
    field evidence physical-device
    field power "$power"
    field thermal "$thermal"
    field crossover_cells "$crossover"
    field portable_artifact_sha256 "$portable_sha"
    field vendor_artifact_sha256 "$vendor_sha"
    field segmented_memory true
    echo "  (cases"
    printf '%s\n' "$measured" | sed 's/^/    /'
    echo '    (projection (observables "amplitude,phase,magnitude-squared") (result "pass"))'
    echo '    (failure (explicit_unavailable "pass") (automatic_cpu_choice "pass") (result "pass"))'
    echo "  )"
    field result measured-pass
    echo ")"
  } >"$output"
  rm -f "$raw"
  verify "$source" "$output"
}

command=${1:-}
shift || true
case "$command" in
  capture)
    test "${1:-}" = --source || usage
    source=${2:-}
    test "${3:-}" = --target || usage
    target=${4:-}
    test "${5:-}" = --output || usage
    output=${6:-}
    test "$#" -eq 6 || usage
    capture "$source" "$target" "$output"
    ;;
  verify)
    test "${1:-}" = --source || usage
    source=${2:-}
    artifact=${3:-}
    test "$#" -eq 3 || usage
    verify "$source" "$artifact"
    ;;
  *)
    usage
    ;;
esac
