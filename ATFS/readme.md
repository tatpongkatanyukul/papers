# ATFS

Nakjai and Katanyukul, Automatic Thai Finger Spelling Transcription.
Walailak Journal

Key contributions
  * (First) TFS transcription: taking a signing video as an input and output a word
  * found signing location can be effective cue (cf. signing time duration)
  * coupling image-classification and sequence modeling helps
  * smoothening is crucial
  
Limitations
  * tested on bi-gram words, although ATFS could be able to handle a word of any length (but it has never been tested)
  * covers only 42 consonants
  * requires a plain background
  * requires one word per video clip
