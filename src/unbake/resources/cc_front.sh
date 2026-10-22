# The compiler the Makefile runs: `sh cc_front.sh CACHE_DIR REAL_CC ARGS...` (resources/cc_cache.py explains the cache).
# A build is two make runs (process.make): the first, with ABU_CC_COLLECT naming a file, only records each compile
# command here and fails it (exit 125); one batch then answers all of them from the cache in a few processes, leaving
# `<out>.cc` and its answer key `<out>.cc.key` beside each answered unit; the second run lands here and moves the
# answer into place only when the hash of its own command line and input equals that key (never by date: a leftover
# answer for another text or other flags is dropped), and starts the python front only for a unit the cache could not
# answer (the compiler must run).
folder=${0%/*} cache=$1 compiler=$2
shift 2
if [ -n "$ABU_CC_COLLECT" ]; then
  fmt=%s
  for a; do fmt=$fmt'\037%s'; done
  printf "$fmt\n" "$compiler" "$@" >> "$ABU_CC_COLLECT"  # one write: lines this short never interleave
  exit 125
fi
out= prev= src=
for a; do
  [ "$prev" = -o ] && out=$a
  case $a in *.i) src=$a ;; esac
  prev=$a
done
if [ -n "$out" ] && [ -n "$src" ] && [ -f "$out.cc.key" ]; then
  fmt=%s
  for a; do fmt=$fmt'\037%s'; done
  have=$({ printf "$fmt\n" "$compiler" "$@"; cat "$src"; } | sha256sum)
  if [ "${have%% *}" = "$(cat "$out.cc.key")" ] && [ -f "$out.cc" ]; then
    rm -f "$out.cc.key"
    exec mv -f "$out.cc" "$out"
  fi
fi
[ -z "$out" ] || rm -f "$out.cc" "$out.cc.key"
exec python3 -I -S -c "import sys;sys.path.insert(0,sys.argv.pop(1));import cc_cache;sys.exit(cc_cache.main(sys.argv))" \
  "$folder" "$cache" "$compiler" "$@"
