# Catalogue data

Edit `dataset.json` here. Package builds include this same file in ModelScout,
so the checkout and installed app use the same starting data.

The file has hardware specifications, canonical model records and a small set
of benchmark scores. Its `sources` entries link to publisher sites, and some
records include their own source URL. Those broad links are useful starting
points, but they don’t establish where every number came from.

The benchmark source adapters in `src/modelscout/benchmarks/sources/` currently
return fixed records. They are not live leaderboard scrapers. A record marked
`direct` describes the stored evidence category; it is not a verification made
by this app. The score dates and exact model matches still need checking against
published results before the catalogue can be called verified.

Use the hardware values and scores as a starting point. Check the original model
card for runtime support and terms before downloading weights. This repository
contains metadata, not model weights.

To check the file format and reload the local cache:

```bash
./start.sh database validate assets/dataset.json
./start.sh refresh
```

Validation checks the schema and basic bounds. It cannot establish that a model
exists or that a benchmark score is accurate.
