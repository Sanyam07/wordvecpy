# MassiveWordVec

**wordvecpy** is a library for processing text data, tokenizing it, and building word vector dictionaries and whole word vector embeddings from the corpus text.

**TextProcessor** takes a corpus of unprocessed text and processes it for use with word vectors.  Punctuation, stopwords, substitutions, contractions, and lemmatization can all be customized.

**VectorDictionary** loads pretrained word embeddings from .txt files so they can be used with other classes.  Every class in this package that requires a vector dictionary can take a pymagnitude vector or a VectorDictionary object.

**Vectokenizer** and **FastVectokenizer** both convert processed text corpus into integer embeddings and create vector dictionaries for those associated integer embeddings.  Both classes do the exact same thing but FastVectokenizer requires Keras to create integer embeddings and Vectokenizer does not.

**VectorEmbeddedDoc** and **LoadVectorEmbeddedDoc** generate (and save, if needed) and load complete word vector embeddings.  As these can take up a huge amount of memory quickly, it is capable of splitting and saving in slices of data.  This is most useful for using word vector embeddings in raw form.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wordvecpy.

```bash
pip install wordvecpy
```

## Usage


## Future Plans
In the future I will build support for just generating a word vector lookup dictionary for all (or some specified number of) words in the corpus.  Additionally, plan to use this dictionary to generate matrix embeddings in the format Keras uses to load weights into an embedding layer.

## License
None brah
# wordvecpy
