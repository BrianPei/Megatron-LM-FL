#!/bin/bash

yq_is_usable() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" --version >/dev/null 2>&1
}

resolve_yq_bin() {
    local script_dir="${1:?script_dir is required}"
    local -a candidates=()

    if [[ -n "${YQ_BIN:-}" ]]; then
        candidates+=("$YQ_BIN")
    fi

    if command -v yq >/dev/null 2>&1; then
        candidates+=("$(command -v yq)")
    fi

    candidates+=("/usr/local/bin/yq" "$script_dir/yq_compat.py")

    local candidate
    for candidate in "${candidates[@]}"; do
        if yq_is_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "Unable to find a working yq executable. Install yq, set YQ_BIN, or use ${script_dir}/yq_compat.py." >&2
    return 1
}
