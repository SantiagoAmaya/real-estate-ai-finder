"""Tests for scraper module"""
import pytest
from src.data.scraper import IdealistaScraperSimple

def test_scraper_initialization():
    """Test scraper can be initialized"""
    scraper = IdealistaScraperSimple()
    assert scraper.location == "barcelona"
    assert scraper.property_type == "local"

def test_scraper_returns_data():
    """Test scraper returns property data"""
    scraper = IdealistaScraperSimple()
    properties = scraper.scrape_search_results(max_pages=1)
    assert len(properties) > 0
    assert 'id' in properties[0]
    assert 'title' in properties[0]
    assert 'price' in properties[0]
    assert 'description' in properties[0]

def test_property_has_required_fields():
    """Test each property has required fields"""
    scraper = IdealistaScraperSimple()
    properties = scraper.scrape_search_results(max_pages=1)
    
    required_fields = ['id', 'title', 'price', 'description', 
                      'location', 'size_m2', 'url', 'scraped_at']
    
    for prop in properties:
        for field in required_fields:
            assert field in prop, f"Missing field: {field}"
