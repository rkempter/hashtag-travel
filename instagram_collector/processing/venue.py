"""
Augment the clusters with information from foursquare
"""

import logging

from instagram_collector.config import FOURSQUARE_RADIUS
from instagram_collector.helper import to_unicode_or_bust


def map_venue(data):
    """
    Return only the required keys from the received foursquare json

    id, name, lat, lng, address, url, category_name,
    stat_checkin, stat_user, stat_tip, stat_likes,
    stat_rating, stat_rating_count, stat_photo_count,
    tags, stat_listed, price
    """
    def _get_category(categories):
        for category in categories:
            if category['primary']:
                return category['name']


    def _get_price_tier(attributes):
        for attribute in attributes['groups']:
            if attribute['type'] == 'price':
                for item in attribute['items']:
                    if 'priceTier' in item:
                        return item['priceTier']

        return 0

    def _get_photo(photos):
        if photos:
            groups = photos['groups']
            if not groups:
              return None
            for group in groups:
                if group['type'] != 'venue':
                    continue

                items = group['items']
                return "%s/width720/%s" % (items[0]['prefix'], items[0]['suffix'])

        return None

    venue = dict()
    venue['id'] = data['id']
    venue['name'] = to_unicode_or_bust(data['name'])
    venue['address'] = to_unicode_or_bust(",".join(data['location']['formattedAddress']))
    venue['coordinates'] = [data['location']['lng'], data['location']['lat']]
    venue['category_name'] = to_unicode_or_bust(_get_category(data['categories']))
    venue['url'] = data['canonicalUrl']
    venue['fq_tags'] = to_unicode_or_bust(",".join(data['tags']))
    venue['price'] = _get_price_tier(data['attributes'])

    if hasattr(data, 'photos'):
        photo_url = _get_photo(data['photos'])
        if photo_url:
            venue['photo'] = photo_url

    return venue


def retrieve_foursquare_data(foursquare_api, query, lat, lng):

    data = foursquare_api.venues.search(params={
        'query': query,
        'll': "%s,%s" % (lat, lng),
        'radius': FOURSQUARE_RADIUS}
    )

    if data and data['venues']:
        venue = foursquare_api.venues(data['venues'][0]['id'])
        filtered_data = map_venue(venue['venue'])

        return filtered_data

    return None
