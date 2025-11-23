# Data Schema Documentation

## Fotocasa Scraped Data

### Overview
Este documento describe la estructura de datos obtenidos del scraper de Fotocasa.

### File Format
- **Format**: JSON
- **Encoding**: UTF-8
- **Structure**: Two main sections: `properties` (array) and `metadata` (object)

---

## Properties Array

Cada elemento en el array `properties` representa una propiedad inmobiliaria con los siguientes campos:

### Core Fields (Always Present)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier from Fotocasa | `"187843062"` |
| `url` | string | Direct URL to the property listing | `"https://www.fotocasa.es/..."` |
| `title` | string | Property title (truncated from description) | `"Encantador piso señorial..."` |
| `price` | integer | Price in EUR (from `rawPrice`) | `449000` |
| `location` | string | Formatted location (neighborhood, district, municipality) | `"Sant Antoni, Eixample, Barcelona Capital"` |
| `source` | string | Data source identifier | `"fotocasa"` |
| `scraped_at` | string | ISO timestamp of when property was scraped | `"2025-11-23T15:49:13.770664"` |

### Optional Fields

| Field | Type | Description | Example | Note |
|-------|------|-------------|---------|------|
| `size_m2` | float or null | Property size in square meters | `103.0` | From `features.surface` |
| `rooms` | integer or null | Number of rooms | `3` | From `features.rooms` |
| `bathrooms` | integer or null | Number of bathrooms | `1` | From `features.bathrooms` |
| `description` | string | Full property description in Spanish | `"Encantador piso..."` | Can be empty |
| `phone` | string | Contact phone number | `"932517542"` | Can be empty |
| `images` | array[string] | URLs of property images | `["https://static.fotocasa.es/..."]` | Can be empty array |

---

## Metadata Object

Contains information about the scraping process and data quality metrics.

### Structure

```json
{
  "scraping": {
    "timestamp": "2025-11-23T15:49:13.770681",
    "total_found": 6,
    "scraping_time_seconds": 5.97,
    "scraper_version": "v0.1"
  },
  "filters": {
    "location": "barcelona-capital",
    "property_type": "vivienda",
    "min_rooms": 2,
    "max_price": 500000,
    "min_size_m2": 40
  },
  "data_quality": {
    "properties_with_images": 6,
    "properties_with_description": 6,
    "properties_with_phone": 5,
    "avg_images_per_property": 28.5,
    "properties_with_complete_location": 6
  },
  "statistics": {
    "avg_price": 459500.0,
    "min_price": 449000,
    "max_price": 470000,
    "avg_size_m2": 94.5
  }
}
```

### Metadata Fields Explanation

#### `scraping` section
- **timestamp**: When the scraping was executed (ISO 8601 format)
- **total_found**: Number of properties successfully scraped
- **scraping_time_seconds**: Total time taken for the scraping process
- **scraper_version**: Version of the scraper used (for reproducibility)

#### `filters` section
Records all filters applied during scraping. Useful for:
- Understanding what subset of data was collected
- Reproducing the scraping with same parameters
- Analyzing filter effectiveness

#### `data_quality` section
Metrics about data completeness:
- **properties_with_images**: Count of properties with at least one image
- **properties_with_description**: Count with non-empty description
- **properties_with_phone**: Count with contact phone
- **avg_images_per_property**: Average number of images per listing
- **properties_with_complete_location**: Count with full location data

#### `statistics` section
Descriptive statistics about the collected data:
- **avg_price**: Average price across all properties
- **min_price**: Minimum price found
- **max_price**: Maximum price found
- **avg_size_m2**: Average size (only counting properties with size data)

---

## Data Quality Notes

### Expected Completeness Rates

Based on Fotocasa data structure:

| Field | Expected Completeness | Notes |
|-------|----------------------|-------|
| Core fields (id, url, price, location) | 100% | Always present or scraping fails |
| images | ~95% | Very few listings without images |
| description | ~90% | Most have descriptions |
| size_m2 | ~85% | Sometimes not provided |
| rooms | ~85% | Sometimes not provided |
| bathrooms | ~80% | Sometimes not provided |
| phone | ~70% | Many agencies don't publish phones |

### Known Data Issues

1. **Location format**: Some properties may have incomplete location hierarchy
2. **Price outliers**: Properties may have placeholder prices (e.g., 1€ for "price on request")
3. **Size variations**: Commercial properties may use different size metrics
4. **Description language**: All descriptions are in Spanish
5. **Image URLs**: URLs are CDN links and may expire over time

---

## Usage Example

```python
import json

# Load data
with open('data/raw/fotocasa_20251123_154917.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Access properties
properties = data['properties']
metadata = data['metadata']

# Example: Filter properties by price
affordable = [
    p for p in properties 
    if p['price'] < 400000
]

# Example: Get properties with many images
well_documented = [
    p for p in properties 
    if len(p['images']) >= 10
]

# Example: Check data quality
quality = metadata['data_quality']
print(f"Completeness rate for images: {quality['properties_with_images'] / metadata['scraping']['total_found'] * 100}%")
```

---

## Version History

### v0.1 (2025-11-23)
- Initial schema definition
- Basic fields from Fotocasa JSON structure
- Added comprehensive metadata tracking
- Implemented data quality metrics

---

## Future Enhancements

Potential fields to add in future versions:
- [ ] Energy efficiency rating
- [ ] Year of construction
- [ ] Floor number
- [ ] Elevator availability
- [ ] Parking availability
- [ ] Extracted features from images (via CV model)
- [ ] Sentiment score from description (via NLP)
- [ ] Neighborhood statistics (crime, schools, etc.)