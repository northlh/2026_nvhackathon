
# ---------------------------
# Determine writable TMPDIR
# ---------------------------
if [[ -z "${TMPDIR:-}" ]]; then
  for base in "/gscratch/$USER" "/scratch/$USER" "/localscratch/$USER"; do
    if [[ -d "$base" && -w "$base" ]]; then
      export TMPDIR="$base/tmp"
      mkdir -p "$TMPDIR" || {
        echo "❌ Could not create TMPDIR inside $base"
        exit 1
      }
      echo "✅ TMPDIR auto-set to $TMPDIR"
      break
    fi
  done
fi

if [[ -z "${TMPDIR:-}" ]]; then
  echo "❌ TMPDIR not set and no writable scratch directory found"
  exit 1
fi

# Final validation
touch "$TMPDIR/.write_test.$$" 2>/dev/null || {
  echo "❌ TMPDIR not writable: $TMPDIR"
  exit 1
}
rm -f "$TMPDIR/.write_test.$$"
