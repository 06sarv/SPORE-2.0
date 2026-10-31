# SPORE v3 Source Review

Run `collect_mgnify_catalog.py` first. The generated CSV is an inventory of candidate public soil studies, not an approved training input.

Approve a study only when all of these are recorded in its provenance entry:

- Public source URL/accession, release date, and explicit licence compatible with the intended use.
- Geographic coordinates at sample level, coordinate uncertainty, and soil habitat rather than a mixed or host-associated sample.
- Sequencing and taxonomy method, including classifier/database/version.
- A usable abundance or taxonomic table linked to each sample; genome catalogues alone are not sample labels.
- Whether it is an independent study/H3 block from current validation data.

MGnify is the first source because it offers API-accessible studies, samples, and analyses. Use the Soil genome catalogue only to help map taxonomy, never as a replacement for observed sample labels.

For Earth Microbiome Project data, select a Qiita/EMP study only after downloading its metadata and confirming public redistribution terms, coordinates, environmental descriptors, sequence method, and sample/taxon table linkage. Export it to the canonical schema in `V3_TRAINING_GUIDE.md`, preserving its source study ID and method in additional columns.

Do not combine records across sources until every taxon has been normalized to one frozen GTDB release and the source-specific taxonomy release is retained in provenance.
