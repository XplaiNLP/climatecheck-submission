# climatecheck-submission

This repo contains the code for the ClimateCheck 2025 Shared Task submission `EFC`, which is documented in the paper [Comparing LLMs and BERT-based Classifiers for Resource-Sensitive Claim Verification in Social Media](https://aclanthology.org/2025.sdp-1.26/).

# Downloads
Please download the dataset and the corpus from:
- Training/Testing Datasets: https://huggingface.co/datasets/rabuahmad/climatecheck
- Publications Corpus: https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus

and save the files in the root dir of this repo.

# Installation
- create a new environment with `python -m venv venv`
- `source venv/bin/activate`
- `python -m pip install -r requirements.txt`

# Running
- The project files are split into the ClimateCheck@SDP 2025 Shared Task subtasks 
- The data fine-tuning and data preparation scripts can be found in each directory, as well as the inference scripts

## Citation

If you find this repo useful, please cite it as:
```bibtex
@inproceedings{upravitelev-etal-2025-comparing,
    title = "Comparing {LLM}s and {BERT}-based Classifiers for Resource-Sensitive Claim Verification in Social Media",
    author = "Upravitelev, Max  and
      Duran-Silva, Nicolau  and
      Woerle, Christian  and
      Guarino, Giuseppe  and
      Mohtaj, Salar  and
      Yang, Jing  and
      Solopova, Veronika  and
      Schmitt, Vera",
    editor = "Ghosal, Tirthankar  and
      Mayr, Philipp  and
      Singh, Amanpreet  and
      Naik, Aakanksha  and
      Rehm, Georg  and
      Freitag, Dayne  and
      Li, Dan  and
      Schimmler, Sonja  and
      De Waard, Anita",
    booktitle = "Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.sdp-1.26/",
    doi = "10.18653/v1/2025.sdp-1.26",
    pages = "281--287",
    ISBN = "979-8-89176-265-7"
}
\```
