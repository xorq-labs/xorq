#!/usr/bin/env bash
# Tests .envrcs/.envrc.gitignore and .envrcs/.envrc.from-template in scratch
# git repos, and that every bundle .envrcs/.envrc.secrets.template names
# loads through .envrcs/.envrc.sops.
#
# Each case loads a fragment through a source_env that pushd's into .envrcs/
# as direnv's does, so a path missing its $direnv_root prefix fails here as
# it would under direnv. It runs the fragment in direnv's default mode, not
# strict_env, so a failing command inside it reaches the case's checks
# rather than ending the run; the strict_env cases at the end load under
# strict_env as well. watch_file, log_status and log_error are stubs (the
# first records its arguments in $direnv_root.watch); direnv itself is not
# needed. Needs bash, git, awk and install.
#
# Usage: .envrcs/test-fragments.sh   (tests the working tree it sits in)
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd -P)
work=$(cd "$(mktemp -d)" && pwd -P)
trap 'chmod -R u+rw "$work" 2>/dev/null; rm -rf "$work"' EXIT
failures=0
# No host git config and no host excludes file: either would make
# check-ignore pass for a file the repo's own rules do not ignore. With no
# config, git still reads $XDG_CONFIG_HOME/git/ignore, so that points here.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 XDG_CONFIG_HOME=$work/xdg

fragments_into() { # <dir>: the fragments and templates under test
    mkdir -p "$1/.envrcs"
    cp "$repo/.gitignore.template" "$1/"
    cp "$repo"/.envrcs/.envrc.{gitignore,from-template,secrets.template,user.template,sops} "$1/.envrcs/"
}

# A main checkout at $1 holding the fragments and templates under test.
# git init, not a clone: the cases need .git as a directory, a file, or
# absent, and nothing else from the repo.
scratch_repo() {
    git init -q "$1"
    git -C "$1" -c user.name=test -c user.email=test@example.invalid \
        commit -q --allow-empty -m init
    fragments_into "$1"
}

# Load fragment $2 with direnv_root=$1, in mode $3 (default: direnv's
# default; "strict" for strict_env); stderr to $1.err. Prints, after the
# load, which of the fragments' own variables are still set. Exits non-zero
# if the load itself did, which only strict mode can do.
#
# A fresh bash process, not a subshell: `check` calls this inside an `if`,
# and bash ignores `set -e` for everything run from an `if` condition,
# subshells included, so a subshell's strict mode would test nothing.
# shellcheck disable=SC2016 # the script is for the inner bash to expand
loader='
    case $3 in
    strict) set -euo pipefail ;;
    *) set +eu +o pipefail ;;
    esac
    export direnv_root=$1
    watch_file() { printf "%s\n" "$@" >>"$direnv_root.watch"; }
    log_status() { echo "log_status: $*" >&2; }
    log_error() { echo "log_error: $*" >&2; }
    source_env() {
        pushd "$(dirname "$1")" >/dev/null
        . "./${1##*/}"
        popd >/dev/null
    }
    cd "$direnv_root"
    source_env ".envrcs/$2"
    for v in _gitignore_tmp _gitignore_rc _from_template_f _from_template_rc _from_template_dirs _from_template_todo; do
        [ -z "${!v+set}" ] || echo "$v"
    done
'
load() {
    bash -c "$loader" load "$1" "$2" "${3:-}" 2>"$1.err"
}

check() { # <name> <command>...
    if "${@:2}"; then
        echo "ok   $1"
    else
        echo "FAIL $1"
        failures=$((failures + 1))
    fi
}

no_tmp() { test -z "$(find "$1" -maxdepth 1 -name '.gitignore.tmp.*')"; }
ignored() { git -C "$1" check-ignore -q "$2"; }
mode_600() { [ "$(stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1")" = 600 ]; }
no_errors() { ! grep -q '^log_error' "$1"; }

# --- .envrc.gitignore ---

r=$work/g1
scratch_repo "$r"
leaked=$(load "$r" .envrc.gitignore)
check "gitignore: a first build reports nothing" test ! -s "$r.err"
check "gitignore: no .gitignore.local gives the template" cmp -s "$r/.gitignore" "$r/.gitignore.template"
check "gitignore: leaves no variable set" test -z "$leaked"
check "gitignore: the result ignores itself" git -C "$r" check-ignore -q .gitignore
# xorq's template keeps .gitignore unanchored (upstream anchors it): tools
# write untracked nested ones, and a contributor's existing copy of the
# template then loses no rule on the first rebuild.
check "gitignore: ... and a nested .gitignore too" git -C "$r" check-ignore -q sub/.gitignore
for f in .gitignore.local .gitignore.orig .gitignore.tmp.1; do
    check "gitignore: the result ignores $f at the root" ignored "$r" "$f"
    check "gitignore: ... but not sub/$f" eval "! git -C '$r' check-ignore -q sub/$f"
done
check "gitignore: a rebuild leaves no temp file" no_tmp "$r"
check "gitignore: watches the template and .gitignore.local" test "$(cat "$r.watch")" = \
    "$(printf '%s\n' "$r/.gitignore.template" "$r/.gitignore.local")"

r=$work/g2
scratch_repo "$r"
printf '/one' >"$r/.gitignore.template"
printf '/two\n' >"$r/.gitignore.local"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: template without a final newline, then .gitignore.local" \
    test "$(cat "$r/.gitignore")" = "$(printf '/one\n/two')"

r=$work/g6
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
echo '/added-later' >>"$r/.gitignore.template"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: a template edit reaches an existing .gitignore on reload" \
    cmp -s "$r/.gitignore" "$r/.gitignore.template"
check "gitignore: ... and keeps no .gitignore.orig, since it drops no rule" test ! -e "$r/.gitignore.orig"

# An untracked .gitignore with a rule the rebuild lacks: moved aside, not lost.
r=$work/k1
scratch_repo "$r"
printf '%s\n' '# by hand' '/.gitignore' 'scratch/' >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: a dropped rule's .gitignore is rebuilt" cmp -s "$r/.gitignore" "$r/.gitignore.template"
check "gitignore: ... and the old one kept as .gitignore.orig" grep -qx 'scratch/' "$r/.gitignore.orig"
check "gitignore: ... and that is reported as status" grep -q '^log_status: .*\.gitignore\.orig' "$r.err"
check "gitignore: ... not as an error" no_errors "$r.err"

# Only comments and blank lines differ: nothing is dropped, nothing kept.
r=$work/k2
scratch_repo "$r"
# The template's own rules under another header, so the case holds for any
# template (upstream's fixture names its anchored /.gitignore).
{ printf '%s\n' '# an older header' ''; grep -v '^#' "$r/.gitignore.template"; } >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: a comment-only difference keeps no .gitignore.orig" test ! -e "$r/.gitignore.orig"

# Lines compared as git reads them: CR and trailing blanks are not a dropped
# rule; an indented '#' is a rule, not a comment.
r=$work/k4
scratch_repo "$r"
sed 's/$/\r/' "$r/.gitignore.template" >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: CRLF line endings keep no .gitignore.orig" test ! -e "$r/.gitignore.orig"
r=$work/k5
scratch_repo "$r"
sed 's/$/  /' "$r/.gitignore.template" >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: trailing blanks keep no .gitignore.orig" test ! -e "$r/.gitignore.orig"
r=$work/k6
scratch_repo "$r"
printf '%s\n' '/.gitignore' '  # a pattern to git' >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: an indented '#' line is a rule, kept in .gitignore.orig" test -e "$r/.gitignore.orig"
# A blank line the template lacks is not a rule.
r=$work/k7
scratch_repo "$r"
echo '/.gitignore' >"$r/.gitignore.template"
printf '%s\n' '/.gitignore' '' >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: a blank line keeps no .gitignore.orig" test ! -e "$r/.gitignore.orig"

# awk that cannot say: exits other than 0 and 1 (here 127, as for no awk),
# or an unreadable .gitignore read as empty (busybox awk exits 1). Either
# way the old file, and its hand rule, are kept.
mkdir "$work/awk-127" "$work/awk-1"
printf '#!/bin/sh\nexit 127\n' >"$work/awk-127/awk"
printf '#!/bin/sh\nexit 1\n' >"$work/awk-1/awk"
chmod +x "$work/awk-127/awk" "$work/awk-1/awk"
r=$work/k8
scratch_repo "$r"
echo 'scratch/' >"$r/.gitignore"
PATH=$work/awk-127:$PATH load "$r" .envrc.gitignore >/dev/null
check "gitignore: awk failing keeps the old .gitignore" test "$(cat "$r/.gitignore")" = scratch/
check "gitignore: ... and says it could not compare" grep -q '^log_error: could not compare' "$r.err"
check "gitignore: ... and leaves no temp file" no_tmp "$r"
r=$work/k9
scratch_repo "$r"
echo 'scratch/' >"$r/.gitignore"
chmod 000 "$r/.gitignore"
if [ -r "$r/.gitignore" ]; then
    echo "skip gitignore: unreadable .gitignore (running as root)"
else
    PATH=$work/awk-1:$PATH load "$r" .envrc.gitignore >/dev/null
    chmod 600 "$r/.gitignore"
    check "gitignore: an unreadable .gitignore is kept" test "$(cat "$r/.gitignore")" = scratch/
    check "gitignore: ... and is reported" grep -q '^log_error: could not compare' "$r.err"
fi

# A rename that fails: an mv shim refuses any move whose arguments contain
# $fail_mv, and runs the real mv otherwise. The old .gitignore survives
# each failure, and no temp file is left.
real_mv=$(command -v mv)
mkdir "$work/mv-fail"
printf '#!/bin/sh\ncase "$*" in *"$fail_mv"*) exit 1 ;; esac\nexec %s "$@"\n' \
    "$real_mv" >"$work/mv-fail/mv"
chmod +x "$work/mv-fail/mv"
r=$work/v1
scratch_repo "$r"
cp "$r/.gitignore.template" "$r/.gitignore"
echo '/added' >>"$r/.gitignore.template"
fail_mv=/.gitignore.tmp. PATH=$work/mv-fail:$PATH load "$r" .envrc.gitignore >/dev/null
check "gitignore: a failed rename keeps the old .gitignore" \
    eval "! grep -qx /added '$r/.gitignore'"
check "gitignore: ... and is reported" grep -q '^log_error: .*rename failed' "$r.err"
check "gitignore: ... and leaves no temp file" no_tmp "$r"
r=$work/v2
scratch_repo "$r"
echo 'scratch/' >"$r/.gitignore"
fail_mv=/.gitignore.tmp. PATH=$work/mv-fail:$PATH load "$r" .envrc.gitignore >/dev/null
check "gitignore: a failed rename after moving the old one aside puts it back" \
    test "$(cat "$r/.gitignore" 2>/dev/null)" = scratch/ -a ! -e "$r/.gitignore.orig"
check "gitignore: ... and is reported" grep -q '^log_error: .*rename failed' "$r.err"
check "gitignore: ... and leaves no temp file" no_tmp "$r"
r=$work/v3
scratch_repo "$r"
echo 'scratch/' >"$r/.gitignore"
fail_mv=/.gitignore.orig PATH=$work/mv-fail:$PATH load "$r" .envrc.gitignore >/dev/null
check "gitignore: failing to move the old one aside keeps it" test "$(cat "$r/.gitignore")" = scratch/
check "gitignore: ... and is reported" grep -q '^log_error: could not move .gitignore to' "$r.err"
check "gitignore: ... and leaves no temp file" no_tmp "$r"

# git that cannot say whether .gitignore is tracked: not rebuilt.
r=$work/v4
scratch_repo "$r"
echo '/old' >"$r/.gitignore"
echo garbage >"$r/.git/index"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: git failing on the tracked check keeps the old .gitignore" test "$(cat "$r/.gitignore")" = /old
check "gitignore: ... and is reported" grep -q '^log_error: git could not say whether .gitignore is tracked' "$r.err"

# A .gitignore that is a directory: not rebuilt into.
r=$work/k10
scratch_repo "$r"
mkdir "$r/.gitignore"
leaked=$(load "$r" .envrc.gitignore)
check "gitignore: ... a refused .gitignore directory leaves no variable set" test -z "$leaked"
check "gitignore: a .gitignore directory gets nothing inside" test -z "$(ls -A "$r/.gitignore")"
check "gitignore: ... and is reported" grep -q 'not a regular file' "$r.err"

# A .gitignore.orig already there: never overwritten; nothing is rebuilt.
r=$work/k3
scratch_repo "$r"
echo 'first/' >"$r/.gitignore.orig"
echo 'second/' >"$r/.gitignore"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: an existing .gitignore.orig is not overwritten" test "$(cat "$r/.gitignore.orig")" = first/
check "gitignore: ... and the .gitignore with the dropped rule is kept" test "$(cat "$r/.gitignore")" = second/
check "gitignore: ... and that is reported" grep -q '^log_error: .*\.gitignore\.orig exists' "$r.err"
check "gitignore: ... and leaves no temp file" no_tmp "$r"

r=$work/g3
scratch_repo "$r"
echo '/old' >"$r/.gitignore"
rm "$r/.gitignore.template"
load "$r" .envrc.gitignore >/dev/null
check "gitignore: no template keeps the repo's own .gitignore" test "$(cat "$r/.gitignore")" = /old
check "gitignore: no template is a silent no-op" test ! -s "$r.err"

r=$work/g4
scratch_repo "$r"
echo '/old' >"$r/.gitignore"
chmod 000 "$r/.gitignore.template"
if [ -r "$r/.gitignore.template" ]; then
    echo "skip gitignore: unreadable template (running as root: chmod 000 does not stop reads)"
else
    load "$r" .envrc.gitignore >/dev/null
    check "gitignore: unreadable template keeps the old .gitignore" test "$(cat "$r/.gitignore")" = /old
    check "gitignore: unreadable template is reported" grep -q 'not rebuilt' "$r.err"
    check "gitignore: unreadable template leaves no temp file" no_tmp "$r"
fi

r=$work/g5
scratch_repo "$r"
echo '/tracked' >"$r/.gitignore"
git -C "$r" add .gitignore
load "$r" .envrc.gitignore >/dev/null
check "gitignore: a tracked .gitignore is not overwritten" test "$(cat "$r/.gitignore")" = /tracked
check "gitignore: a tracked .gitignore is reported" grep -q 'is tracked' "$r.err"

# --- .envrc.from-template ---
# Each case runs .envrc.gitignore first, as the root .envrc does, so the two
# files are ignored unless the case says otherwise.

r=$work/t1
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
leaked=$(load "$r" .envrc.from-template)
for f in secrets user; do
    check "from-template: main checkout creates .envrc.$f from its template" \
        cmp -s "$r/.envrcs/.envrc.$f" "$r/.envrcs/.envrc.$f.template"
    check "from-template: .envrc.$f is mode 600" mode_600 "$r/.envrcs/.envrc.$f"
done
check "from-template: leaves no variable set" test -z "$leaked"
check "from-template: main checkout reports nothing" test ! -s "$r.err"

r=$work/t2
scratch_repo "$r"
git -C "$r" worktree add -q --detach "$work/t2-wt"
mkdir -p "$work/t2-wt/.envrcs"
cp "$r"/.gitignore.template "$work/t2-wt/"
cp "$r"/.envrcs/.envrc.* "$work/t2-wt/.envrcs/"
load "$work/t2-wt" .envrc.gitignore >/dev/null
leaked=$(load "$work/t2-wt" .envrc.from-template)
check "from-template: linked worktree creates neither file" \
    test ! -e "$work/t2-wt/.envrcs/.envrc.secrets" -a ! -e "$work/t2-wt/.envrcs/.envrc.user"
check "from-template: ... and says so" grep -q 'linked git worktree' "$work/t2-wt.err"
check "from-template: ... as status, not as an error" no_errors "$work/t2-wt.err"
check "from-template: ... and leaves no variable set" test -z "$leaked"
cp "$work/t2-wt/.envrcs/.envrc.secrets.template" "$work/t2-wt/.envrcs/.envrc.secrets"
cp "$work/t2-wt/.envrcs/.envrc.user.template" "$work/t2-wt/.envrcs/.envrc.user"
leaked=$(load "$work/t2-wt" .envrc.from-template)
check "from-template: a linked worktree that has both files says nothing" test ! -s "$work/t2-wt.err"
check "from-template: ... and leaves no variable set" test -z "$leaked"
rm -f "$work/t2-wt/.envrcs/.envrc.secrets" "$work/t2-wt/.envrcs/.envrc.user"
GIT_DIR=$r/.git load "$work/t2-wt" .envrc.from-template >/dev/null
check "from-template: a linked worktree under an exported GIT_DIR creates neither file" \
    test ! -e "$work/t2-wt/.envrcs/.envrc.secrets" -a ! -e "$work/t2-wt/.envrcs/.envrc.user"

# Fragments in a subdirectory of a checkout, and in a submodule: each is in
# its repo's main checkout, so both files are created.
r=$work/m1
git init -q "$r"
fragments_into "$r/proj"
load "$r/proj" .envrc.gitignore >/dev/null
load "$r/proj" .envrc.from-template >/dev/null
check "from-template: a subdirectory of a checkout creates both files" \
    test -f "$r/proj/.envrcs/.envrc.secrets" -a -f "$r/proj/.envrcs/.envrc.user"
r=$work/m2
scratch_repo "$work/m2-sub"
git init -q "$r"
git -C "$r" -c protocol.file.allow=always submodule add -q "$work/m2-sub" sm 2>/dev/null
fragments_into "$r/sm"
load "$r/sm" .envrc.gitignore >/dev/null
load "$r/sm" .envrc.from-template >/dev/null
check "from-template: a submodule creates both files" \
    test -f "$r/sm/.envrcs/.envrc.secrets" -a -f "$r/sm/.envrcs/.envrc.user"

r=$work/t3
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
: >"$r/.envrcs/.envrc.user"
load "$r" .envrc.from-template >/dev/null
check "from-template: an existing empty .envrc.user is kept empty" test ! -s "$r/.envrcs/.envrc.user"

r=$work/t4
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
rm "$r/.envrcs/.envrc.secrets.template"
load "$r" .envrc.from-template >/dev/null
check "from-template: a missing template is skipped" test ! -e "$r/.envrcs/.envrc.secrets"
check "from-template: a missing template skips only that file" test -f "$r/.envrcs/.envrc.user"
check "from-template: a missing template reports nothing" test ! -s "$r.err"

r=$work/t5
scratch_repo "$r"
printf '%s\n' '!.envrcs/.envrc.secrets' >"$r/.gitignore.local"
load "$r" .envrc.gitignore >/dev/null
load "$r" .envrc.from-template >/dev/null
check "from-template: a file git would not ignore is not created" test ! -e "$r/.envrcs/.envrc.secrets"
check "from-template: ... and is reported as not ignored" grep -q 'envrc.secrets is not ignored' "$r.err"
check "from-template: ... and the ignored one is still created" test -f "$r/.envrcs/.envrc.user"

r=$work/t6
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
rm -rf "$r/.git"
load "$r" .envrc.from-template >/dev/null
check "from-template: no .git creates neither file" \
    test ! -e "$r/.envrcs/.envrc.secrets" -a ! -e "$r/.envrcs/.envrc.user"
# git's own reason, once per file from check-ignore; none from the guard.
check "from-template: ... and git's error appears once per file" \
    test "$(grep -c '^fatal:' "$r.err")" = 2

# git that cannot answer: a corrupt index makes check-ignore exit 128.
r=$work/t7
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
echo garbage >"$r/.git/index"
load "$r" .envrc.from-template >/dev/null
check "from-template: git failing creates neither file" \
    test ! -e "$r/.envrcs/.envrc.secrets" -a ! -e "$r/.envrcs/.envrc.user"
check "from-template: ... and says git could not tell, not 'not ignored'" \
    grep -q 'git could not say' "$r.err"

# --- .envrc.secrets.template ---
# Every bundle the template names exists, so every line runs use_sops. sops
# is a stub that prints the bundle as it is: for a line with no real newline
# that is what sops's dotenv store emits on decrypt (stores/dotenv/store.go,
# LoadPlainFile and EmitPlainFile, at v3.13.3). direnv's dotenv parser is
# the real one when direnv is on PATH; without it, a stub stands in and
# the case checks only that no line is refused before or after sops. The
# stub refuses a call without --input-type dotenv --output-type dotenv, as
# real sops refuses to emit a dotenv bundle as binary. The values are read
# from a child process: direnv hands the user's shell exported ones only.

r=$work/x1
scratch_repo "$r"
mkdir "$r/bin"
# shellcheck disable=SC2016 # the stub's script is for sh to expand
printf '%s\n' '#!/bin/sh' \
    'for t in --input-type --output-type; do' \
    '    case " $* " in *" $t dotenv "*) ;; *) echo "stub sops: no $t dotenv: $*" >&2; exit 1 ;; esac' \
    'done' 'for a; do f=$a; done' 'echo "$f" >>"$SOPS_CALLS"' 'cat "$f"' >"$r/bin/sops"
pem=$(printf '%s\n' '-----BEGIN PRIVATE KEY-----' 'AAAA/b+c=' '-----END PRIVATE KEY-----')
real_direnv=$(command -v direnv || true)
if [ -z "$real_direnv" ]; then
    printf '%s\n' '#!/bin/sh' 'echo "export DIRENV_STUB=1;"' >"$r/bin/direnv"
fi
chmod +x "$r"/bin/*
sed -n 's/^use sops_if_exists \([^ ]*\).*/\1/p' "$r/.envrcs/.envrc.secrets.template" >"$r.bundles"
while read -r b; do
    echo "PLAIN_${b//[^a-z]/_}='x'" >"$r/.envrcs/$b"
done <"$r.bundles"
{
    printf 'SNOWFLAKE_PRIVATE_KEY="%s"\n' "$(awk '{printf "%s\\n", $0}' <<<"$pem")"
    printf '%s\n' "SNOWFLAKE_PRIVATE_KEY_PWD='p w\$d\"x'"
} >"$r/.envrcs/.env.secrets.snowflake.keypair.sops-encrypted"
PATH=$r/bin:$PATH SOPS_CALLS=$r.sops bash -c '
    export direnv_root=$1
    watch_file() { :; }
    log_error() { echo "log_error: $*" >&2; }
    use() { "use_$1" "${@:2}"; }
    source_env() {
        pushd "$(dirname "$1")" >/dev/null
        . "./${1##*/}"
        popd >/dev/null
    }
    cd "$direnv_root"
    source_env .envrcs/.envrc.secrets.template
    sh -c '"'"'printf "%s" "$SNOWFLAKE_PRIVATE_KEY"'"'"' >"$direnv_root.key"
    sh -c '"'"'printf "%s" "$SNOWFLAKE_PRIVATE_KEY_PWD"'"'"' >"$direnv_root.pwd"
' secrets "$r" 2>"$r.err"
check "secrets: the template names the keypair bundle" \
    grep -qx .env.secrets.snowflake.keypair.sops-encrypted "$r.bundles"
check "secrets: every line of the template loads with no error" no_errors "$r.err"
check "secrets: ... and every bundle reaches sops" \
    test "$(sed 's,.*/,,' "$r.sops" 2>/dev/null)" = "$(cat "$r.bundles")"
if [ -n "$real_direnv" ]; then
    check "secrets: the quoted \\n PEM loads with its newlines" \
        test "$(cat "$r.key")" = "$pem"
    check "secrets: ... and the single-quoted passphrase as written" \
        test "$(cat "$r.pwd")" = "p w\$d\"x"
else
    echo "skip secrets: PEM and passphrase values (no direnv on PATH)"
fi

# --- both, under strict_env ---

r=$work/s1
scratch_repo "$r"
check "strict_env: .envrc.gitignore loads" load "$r" .envrc.gitignore strict
check "strict_env: .envrc.from-template loads" load "$r" .envrc.from-template strict
check "strict_env: ... and creates both files" \
    test -f "$r/.envrcs/.envrc.secrets" -a -f "$r/.envrcs/.envrc.user"
r=$work/s2
scratch_repo "$r"
load "$r" .envrc.gitignore >/dev/null
printf '%s\n' '!.envrcs/.envrc.secrets' >>"$r/.gitignore"
check "strict_env: a refused file does not abort the load" load "$r" .envrc.from-template strict
r=$work/s3
scratch_repo "$r"
echo 'scratch/' >"$r/.gitignore"
check "strict_env: keeping a dropped rule's .gitignore loads" load "$r" .envrc.gitignore strict
check "strict_env: ... and keeps it" grep -qx 'scratch/' "$r/.gitignore.orig"

if [ "$failures" -ne 0 ]; then
    echo "$failures failed"
    exit 1
fi
echo "all passed"
