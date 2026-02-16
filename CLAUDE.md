# CLAUDE_new_project.md

This file provides guidance to Claude Code (claude.ai/code) for building the IA Collage Pipeline project.

## Project Description

An LLM agent-based pipeline that creates artistic collages from Internet Archive material. Three-stage workflow:

1. **Search** — An LLM agent issues multiple searches against the Internet Archive via the `internet-archive-mcp` server, gathering source images based on a creative prompt.
2. **Analysis** — NER (Named Entity Recognition) is performed on the retrieved images. Relevant visual elements are identified and cropped.
3. **Composition** — Cropped elements are assembled into a final collage using image editing tools.

## Internet Archive MCP Server Reference

The pipeline's search stage uses the `internet-archive-mcp` MCP server (located at `/home/mattiapggl/mcp-servers/internet-archive-mcp`). It exposes three tools over stdio transport.

### Available Tools

#### `search_images_tool`

Search the Internet Archive for images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Keywords to search for |
| `max_results` | `int` | `10` | Maximum number of results |
| `collection` | `str \| None` | `None` | Collection identifier to filter by |

Returns JSON array of objects:

```json
{
  "identifier": "img001",
  "title": "Sunset",
  "creator": "Alice",
  "date": "2024-01-01",
  "description": "A sunset photo",
  "thumbnail_url": "https://archive.org/services/img/img001",
  "details_url": "https://archive.org/details/img001"
}
```

- `thumbnail_url` is a direct image URL (pattern: `https://archive.org/services/img/{identifier}`)
- `details_url` links to the item's page (pattern: `https://archive.org/details/{identifier}`)
- Missing fields default to `""`

#### `list_curated_collections_tool`

Returns a hardcoded list of high-quality image collections. No parameters.

Included collections: `nasa`, `flickrcommons`, `metropolitanmuseumofart`, `biodiversitylibrary`, `smithsonian`, `brooklynmuseum`, `library_of_congress`.

Returns JSON array of `{"identifier", "title", "description"}`.

#### `search_collections_tool`

Search for Internet Archive collections by keyword.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Keywords to search for |
| `max_results` | `int` | `10` | Maximum number of results |

Returns JSON array of `{"identifier", "title", "description", "details_url"}`.

### Connecting to the MCP Server

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "internet-archive": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/internet-archive-mcp", "internet-archive-mcp"]
    }
  }
}
```

Claude Code:

```bash
claude mcp add internet-archive -- uv run --directory /path/to/internet-archive-mcp internet-archive-mcp
```

### Downloading Full-Resolution Images

The MCP server returns `thumbnail_url` (low-res preview) and `details_url` (item page). To download actual image files for cropping/compositing, use the `internetarchive` Python library directly:

```python
import internetarchive
item = internetarchive.get_item("identifier")
# item.files lists all files; filter by format (e.g. "JPEG", "PNG")
```

Or download via URL: `https://archive.org/download/{identifier}/{filename}`.

## Pipeline Architecture Guidelines

### Stage 1: Search Agent

- The LLM agent receives a creative prompt and decides which searches to perform (queries, collections, result counts).
- It should issue multiple `search_images_tool` calls with varied queries to get diverse material.
- Use `list_curated_collections_tool` or `search_collections_tool` to discover good source collections or then pass collection identifiers to `search_images_tool` for focused searches.
- The agent should deduplicate results by `identifier` across multiple searches.
- Thumbnail URLs can be used for quick previews; full-resolution downloads are needed for the analysis/composition stages.

### Stage 2: Analysis (NER + Cropping)

- Perform visual NER on downloaded images to identify and locate relevant elements (people, objects, text, landmarks).
- Crop identified regions for use as collage elements.
- Maintain metadata linkage: each crop should track its source `identifier`, original position, and detected entity label.

### Stage 3: Composition

- Arrange cropped elements into a collage based on the creative prompt.
- Image editing tools handle layering, scaling, rotation, and blending of elements.
