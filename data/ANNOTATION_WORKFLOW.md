# MTSamples Attribution Annotation Workflow

This workflow creates source-grounded fact annotations for FaithfulMed. It is an
**attribution** task: record exactly what the selected report states. Annotators
must not correct the report, infer unstated information, or judge whether its
clinical claims are medically correct. The schema and validator check structure
and source attribution; they do not establish medical correctness.

## Selecting reports and passages

Select reports according to a written sampling plan before annotation begins.
Put the document you select into the google doc. Balance the four project content 
types, avoid choosing only unusually short or simple reports, and record the 
stable source identifier and URL.

Use `entire_report` when the report is reasonably sized, its sections form a
coherent unit, and all included material is relevant to the annotation task. Use
`self_contained_passage` when a report is long or contains separable material.
The passage must retain enough local context to understand its facts, including
negation and uncertainty. Record the retained section names in

Copy only the report text needed for the selected example. Exclude navigation,
menus, advertising, cookie notices, search controls, related-content links,
headers and footers, user comments, and other webpage chrome. Do not include
identifiers or content outside the de-identified dataset record.

Scratch work belongs directly in `data/annotations/scratch/`. Development and
test independent files belong in their respective `independent/` directories.

## Adjudication and freezing

After both independent files are complete, both annotators compares their
annotations and agree on a final version. This final version is then added
into the adjudicated directory.

The repository ignores filled annotation JSON by default so make sure to 
change it.

## Synthetic example

```json
{
  "example_id": "SYNTHETIC_DEMO_001",
  "source_url": "https://example.invalid/synthetic-demo",
  "primary_content_type": "instructions",
  "selection_mode": "entire_report",
  "source_text": "Instructions: Place the blue folder on Shelf 3 by Friday, June 5. Do not remove the label.",
  "facts": [
    {
      "fact_id": "F1",
      "fact_text": "The blue folder should be placed on Shelf 3 by Friday, June 5.",
      "fact_type": "instruction",
      "source_sentence": "Place the blue folder on Shelf 3 by Friday, June 5."
    },
    {
      "fact_id": "F2",
      "fact_text": "The label should not be removed.",
      "fact_type": "instruction",
      "source_sentence": "Do not remove the label."
    }
  ],
  "annotation_notes": "Synthetic formatting example only.",
}
```


## Validation

From the repository root, validate a split recursively:

```bash
python data/validate_annotations.py data/annotations/dev
```

Omitting the path validates all annotation directories. The validator checks
JSON syntax, required fields and types, allowed values, unique fact IDs,
whitespace-normalized source-sentence attribution, filenames and locations,
completed independent pairs, frozen test gold files, and parent-document overlap
between development and test. A successful structural validation is not a
medical review.
