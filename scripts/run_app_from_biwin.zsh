#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
dataset_root="${BIWIN_DATASET_ROOT:-/Volumes/BIWIN/CollidingGalaxiesSFR}"
cache_root="${COLLIDE_VIS_CACHE:-/Volumes/BIWIN/collide-vis-001-cache}"

if [[ ! -d "$dataset_root/output" ]]; then
	print "Dataset output directory not found: $dataset_root/output" >&2
	exit 66
fi

cd "$project_dir"

app_binary=""
for candidate in \
	"bin/collide-vis-001Debug.app/Contents/MacOS/collide-vis-001Debug" \
	"bin/collide-vis-001_debug.app/Contents/MacOS/collide-vis-001_debug"
do
	if [[ -x "$candidate" ]]; then
		app_binary="$candidate"
		break
	fi
done

if [[ -z "$app_binary" ]]; then
	print "Built app binary was not found. Build the Debug target first, then rerun this script." >&2
	exit 69
fi

mkdir -p "$cache_root"

print "Starting collide-vis-001 full-res from Terminal."
print "Dataset: $dataset_root"
print "Cache root: $cache_root"
print "Binary: $app_binary"
print ""
print "This version builds a full particle cache. Expect roughly 175-180GB of cache writes."
print "Cache generation progress will appear below."

COLLIDE_VIS_DATA="$dataset_root" \
COLLIDE_VIS_CACHE="$cache_root" \
	"$project_dir/$app_binary"
