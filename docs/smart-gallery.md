# Smart Gallery

<div dir="rtl">

گالری هوشمند: آمار کل گالری، گروه‌بندی خودکار افراد، و شش حالت جستجو در یک صفحه.

</div>

## Run it

The quickest path is the notebook: open `run_all.ipynb` in VS Code, set `GALLERY` to a
folder and press **Run All**. It indexes the folder and opens the gallery in the browser.

Once a folder is indexed, the gallery opens on its own:

```bash
python -m vie.gallery --config data/runs/photos-final/config.yaml
```

or double-click `smart_gallery.bat`. Everything a run writes, the index, named people,
cached crop vectors and reports, stays under `data/runs/<run>/`, which git ignores.

## What it shows

| Tab | What it is |
|:--|:--|
| Overview | Photos, distinct people, animals by group, dishes, subjects and settings |
| Search | One sub-tab per mode, so it is always clear what is being searched |
| People | Faces grouped into people; click one to see their photos and name them |
| Browse | The gallery filtered by any person, animal, dish, subject or setting |

Every search shows the photos with what matched boxed and labelled, a ranked table, and
an HTML report plus CSV to download.

## Search modes

| Mode | Input | How it answers |
|:--|:--|:--|
| Face | a photo of a person | `vie search face` |
| Person by name | a name given in the People tab, or a photo plus a name | face search with every face stored under that name |
| Animal | an English name | `vie search animal`, plus every stored animal of that group |
| Food | an English dish name | `vie search food`, minus photos that are plainly not food |
| Concept | an English description | `vie search text` |
| Similar photos | a photo | its whole-image vector against every stored one |

## What the gallery adds, and why

**People are counted from faces confirmed by a second model.** The face detector is
deliberately permissive, so it also fires on animals, food and objects. Before faces are
grouped into people, each crop is scored against "a human face" and rivals ("an animal",
"food", "an object", "a cartoon"). Only the overview uses this check. Face search still
sees every detection.

**Animals are grouped by ImageNet id range, not by name.** About 120 dog breeds occupy ids
151 to 268, and the search package's word list for "dog" does not name all of them
("vizsla" is missing). The overview groups by range, and an animal search for a group
name returns every stored animal of that group, so the search and the counts agree.
Animals the classifier was unsure of are named from their own crop by the
vision-language model. A box that reads as a person or an object is left out of the
counts, because the detector was wrong.

**Food results must look like food.** The food path scores a whole photo when the
permissive food gate admitted it without a food box. Its rival prompts are "a different
food" and "an empty plate", and neither says "not food at all", so a cat once scored 99%
as sushi. The gallery keeps a dish result only when the detector saw a dish or the
photo's main subject reads as food, and the result says how many it set aside.

**Subjects, settings and dishes cost no image inference.** They are zero-shot labels
scored against the whole-image vectors stored at index time. Each group is a softmax
over its own labels, as video labels are, and a label needs 40% of its group to stick
(30% for dishes).

**Crop vectors are cached.** The species and dish paths encode every candidate crop on
every query, which took two minutes per dish query on a CPU. The crops and the scoring
are unchanged; the vectors are kept in `crop_cache.db`, so a repeat costs nothing.
`Gallery.warm_up()` fills the cache before the page opens.

**Names belong to faces, not to cluster numbers.** A name is stored as face embeddings in
`people.json`, so it survives re-indexing and a new clustering. Nothing is ever named
automatically.
