# Agents

## Genomic Chatbot CLI

This repository now includes a standalone CLI for completing the genomic chatbot workflow outside of Colab.

### What this project does

* Crawls a knowledge source (Markdown) with Firecrawl.
* Starts a SambaNova-backed chat agent that uses the saved knowledge.
* Provides helpers to list and download ENCODE public datasets.

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the required API keys (or copy `.env.example` to `.env` and export them):

```bash
export FIRECRAWL_API_KEY="your-firecrawl-key"
export SAMBA_API_KEY="your-sambanova-key"
```

### Repository layout

* `genomic_chatbot.py`: CLI entrypoint and implementation.
* `requirements.txt`: pinned dependencies.
* `.env.example`: environment variable template.

### Crawl knowledge

```bash
python genomic_chatbot.py crawl-knowledge \
  --url "https://sambanova.ai/blog/qwen-2.5-32b-coder-available-on-sambanova-cloud"
```

### Chat

```bash
python genomic_chatbot.py chat
```

### ENCODE dataset helpers

List ENCODE objects:

```bash
python genomic_chatbot.py encode-list --limit 10
```

Download an ENCODE object:

```bash
python genomic_chatbot.py encode-download \
  "2010/11/16/0c744ab2-e5de-4852-9a93-021fdc6d82f7/ENCFF001TCG.broadPeak.gz" \
  --destination local_data/ENCFF001TCG.broadPeak
```

Download metadata/sample from a `files.txt` list:

```bash
python genomic_chatbot.py encode-metadata files.txt --columns accession assay term_name
```
