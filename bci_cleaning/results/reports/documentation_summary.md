# Documentation Summary

## Sources consulted

- Zenodo dataset record: https://zenodo.org/records/8089820
- Scientific Data descriptor: https://www.nature.com/articles/s41597-023-02445-z
- Local English and French experiment instructions and checklists
- Local English and French questionnaire documents
- Local Mental Rotation questionnaire
- OpenViBE scenarios, scripts, channel list, and participant notes

## Documented structure

- 87 anonymized participants: A1-A60, B61-B81, and C82-C87.
- Each participant attended one session.
- A complete participant has two baseline GDF files, acquisition runs R1-R2, and online runs R3-R6.
- Each MI run contains 40 trials: 20 left-hand and 20 right-hand motor-imagery trials.
- Signals contain 27 EEG, 3 EOG, and 2 EMG channels sampled at 512 Hz.
- The workbook is the documented source for performance, demographic, questionnaire, personality, and cognitive-profile fields.

## Documented exceptions and constraints

- A1 acquisition runs were reconstructed from a concatenated recording and lack end-of-trial and end-of-run triggers.
- A59 did not complete R5 or R6; associated EEG and filters are absent.
- A9 and A11 used only R1 for frequency-band selection, although both acquisition recordings remain present.
- Thirteen Dataset B participants have documented questionnaire losses.
- C83 has documented ILS and 16PF5 losses.
- Noisy channels/trials and experimenter comments are intentionally published and must not be removed.
- Configuration-name variants and any undocumented missing assets are preserved and reported, never inferred.

## Cleaning authorization

Only line-ending/encoding changes that pass parsed-cell equivalence are authorized. GDF, XLSX, XML, configuration, questionnaire, instruction, timestamp, identifier, and scientific numeric content remain unchanged.
