#!/bin/zsh
set -euo pipefail

dataset_root="${BIWIN_DATASET_ROOT:-/Volumes/BIWIN/CollidingGalaxiesSFR}"
output_root="$dataset_root/output"
expected_start="${1:-0}"
expected_end="${2:-3966}"

if [[ ! -d "$output_root" ]]; then
	print "Output directory not found: $output_root" >&2
	exit 66
fi

tmp_file="$(mktemp -t collide-vis-frames.XXXXXX)"
trap 'rm -f "$tmp_file"' EXIT

while IFS= read -r file; do
	base="${file:t}"
	frame_text="${base#snapshot_}"
	frame_text="${frame_text%.hdf5}"
	if [[ "$frame_text" == <-> ]]; then
		print "$((10#$frame_text))\t$file"
	fi
done < <(find "$output_root" -name 'snapshot_*.hdf5' -type f -print) > "$tmp_file"

total_count="$(wc -l < "$tmp_file" | tr -d ' ')"
unique_count="$(cut -f1 "$tmp_file" | sort -n | uniq | wc -l | tr -d ' ')"
duplicate_count="$(cut -f1 "$tmp_file" | sort -n | uniq -d | wc -l | tr -d ' ')"
expected_count=$((expected_end - expected_start + 1))

print "Dataset: $dataset_root"
print "Snapshots total files: $total_count"
print "Snapshots unique frames: $unique_count / $expected_count expected (${expected_start}...${expected_end})"
print "Duplicate frame numbers: $duplicate_count"

awk -v start="$expected_start" -v end="$expected_end" '
	{ seen[$1] = 1 }
	END {
		missing = 0
		rangeStart = -1
		rangeEnd = -1
		for(i = start; i <= end; ++i) {
			if(!(i in seen)) {
				++missing
				if(rangeStart < 0) {
					rangeStart = i
					rangeEnd = i
				} else if(i == rangeEnd + 1) {
					rangeEnd = i
				} else {
					if(printed < 16) {
						printf("Missing range: %d...%d\n", rangeStart, rangeEnd)
						++printed
					}
					rangeStart = i
					rangeEnd = i
				}
			}
		}
		if(rangeStart >= 0 && printed < 16) {
			printf("Missing range: %d...%d\n", rangeStart, rangeEnd)
		}
		printf("Missing frames: %d\n", missing)
		if(missing == 0) {
			print "Dataset status: complete"
		} else {
			print "Dataset status: incomplete"
		}
	}
' "$tmp_file"

print ""
print "BIWIN dataset size:"
du -sh "$dataset_root" 2>/dev/null || true
print "BIWIN free:"
df -h /Volumes/BIWIN | awk 'NR == 1 || NR == 2 {print}'

cache_root="${COLLIDE_VIS_CACHE:-/Volumes/BIWIN/collide-vis-001-cache}"
if [[ -d "$cache_root" ]]; then
	print ""
	print "Full-res cache root:"
	du -sh "$cache_root" 2>/dev/null || true
	latest_cache="$(find "$cache_root" -maxdepth 1 -type d -name 'dataset-*' -print | sort | tail -n 1)"
	if [[ -n "$latest_cache" ]]; then
		print "Latest render cache: $latest_cache"
		print "Cached frame bins: $(find "$latest_cache" -name 'frame-*.bin' -type f -print | wc -l | tr -d ' ')"
	fi
fi
