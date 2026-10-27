# The compiler the Makefile runs: `sh cc_front.sh CACHE_DIR REAL_CC ARGS...` (resources/cc_cache.py explains the cache).
# A build is two make runs (process.make): the first, with ABU_CC_COLLECT naming a file, only records each compile
# command here and fails it (exit 125); one batch then answers all of them from the cache in a few processes, leaving
# `<out>.cc` and its answer key `<out>.cc.key` beside each answered unit; the second run lands in the python front
# (cc_cache.py `run`), which moves the answer into place only when the hash of its own command line and input equals
# that key (never by date: a leftover answer for another text or other flags is dropped) and runs the compiler for a
# unit the cache could not answer. The object rule's own process (cc_cache.py `--object`, stamprec `object`) does the
# record and the move itself and starts this only for a unit the cache could not answer.
folder=${0%/*} cache=$1 compiler=$2
shift 2
if [ -n "$ABU_CC_COLLECT" ]; then
  fmt=%s
  for a; do fmt=$fmt'\037%s'; done
  printf "$fmt\n" "$compiler" "$@" >> "$ABU_CC_COLLECT"  # one write: lines this short never interleave
  exit 125
fi
exec python3 -I -S -c "import sys;sys.path.insert(0,sys.argv.pop(1));import cc_cache;sys.exit(cc_cache.main(sys.argv))" \
  "$folder" "$cache" "$compiler" "$@"
