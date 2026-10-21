# The compiler the Makefile runs: `sh cc_front.sh CACHE_DIR REAL_CC ARGS...` (resources/cc_cache.py explains the cache).
# A build is two make runs (process.make): the first, with ABU_CC_COLLECT naming a file, only records each compile
# command here and fails it (exit 125); one batch then answers all of them from the cache in a few processes, leaving
# `<out>.cc` beside each answered unit; the second run lands here, moves that file into place, and starts the python
# front only for a unit the cache could not answer (the compiler must run). No python starts for the rest.
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
if [ -n "$out" ] && [ -n "$src" ] && [ "$out.cc" -nt "$src" ]; then
  exec mv -f "$out.cc" "$out"
fi
exec python3 -I -S -c "import sys;sys.path.insert(0,sys.argv.pop(1));import cc_cache;sys.exit(cc_cache.main(sys.argv))" \
  "$folder" "$cache" "$compiler" "$@"
