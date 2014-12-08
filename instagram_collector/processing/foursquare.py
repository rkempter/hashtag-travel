"""
Augment the clusters with information from foursquare
"""

import foursquare as fq
import logging

from instagram_collector.config import (FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET,
                                        FOURSQUARE_OAUTH_TOKEN, FOURSQUARE_RADIUS,
                                        FOURSQUARE_REDIRECT_URL)
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
            for group in groups:
                if group['type'] != 'venue':
                    continue

                items = group['items']

                return "%s/width720/%s" % (items['prefix'], items['suffix'])

        return None

    venue = dict()
    venue['id'] = data['id']
    venue['name'] = to_unicode_or_bust(data['name'])
    venue['address'] = to_unicode_or_bust(",".join(data['location']['formattedAddress']))
    venue['coordinates'] = [data['location']['lat'], data['location']['lng']]
    venue['category_name'] = to_unicode_or_bust(_get_category(data['categories']))
    venue['url'] = data['canonicalUrl']
    venue['fq_tags'] = to_unicode_or_bust(",".join(data['tags']))
    venue['price'] = _get_price_tier(data['attributes'])
    photo_url = _get_photo(data['photos'])
    if photo_url:
        venue['photo'] = photo_url

    return venue


def retrieve_foursquare_data(query, lat, lng):

    foursquare_api = fq.Foursquare(client_id=FOURSQUARE_CLIENT_ID,
                                   client_secret=FOURSQUARE_CLIENT_SECRET)

    try:
        data = foursquare_api.venues.search(params={
            'query': query,
            'll': "%s,%s" % (lat, lng),
            'radius': FOURSQUARE_RADIUS}
        )
    except fq.RateLimitExceeded:
        logging.getLogger(__name__).error("Rate Limit Exceeded!")
        raise fq.RateLimiteExceeded
    except fq.InvalidAuth:
        logging.getLogger(__name__).error("Error while logging in.")
        raise fq.InvalidAuth

    if data['venues']:
        venue = foursquare_api.venues(data['venues'][0]['id'])
        filtered_data = map_venue(venue['venue'])

        return filtered_data

    return None