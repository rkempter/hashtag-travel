#!/usr/bin/env python
# -*- coding: utf-8 -*-
# For all locations retrieved from Instagram, get detailed venue information.

from collections import defaultdict

import foursquare as fq
import logging

from instagram_collector.config import FOURSQUARE_CLIENT_ID,\
                                       FOURSQUARE_CLIENT_SECRET, FOURSQUARE_RADIUS, FOURSQUARE_LIMIT
from instagram_collector.collector import connect_db

# Configure logging
logging.basicConfig(
    filename='foursquare_logs.log',level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

insert_venue_sql = """
    INSERT IGNORE INTO
        venues
            (`id`, `name`, `lat`, `lng`, `address`, `url`, `category_name`,
             `stat_checkin`, `stat_user`, `stat_tip`, `stat_likes`,
             `stat_rating`, `stat_rating_count`, `stat_photo_count`,
             `tags`, `stat_listed`, `price`, `stat_female_likes`, `stat_male_likes`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

insert_tags_sql = """INSERT IGNORE INTO tags (id, tag_name) VALUES (%s, %s);"""

select_venue_sql = """SELECT id FROM venues;"""

# Query for selecting locations that have no information yet
select_location_sql = """
    SELECT DISTINCT
        `location_name`, `location_lat`, `location_lng`
    FROM
        media_events
    WHERE
        venue_id IS NULL AND location_name != ""
    ORDER BY
        location_name, location_lat, location_lng DESC
    LIMIT %s;
""" % FOURSQUARE_LIMIT

insert_foreign_key = """
    UPDATE media_events
    SET `venue_id` = %s
    WHERE
        `location_name` = %s AND
        `location_lat` LIKE %s AND
        `location_lng` LIKE %s;
"""

def to_unicode_or_bust(obj, encoding='utf-8'):
    """ Decode to unicode as soon as possible """
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            obj = unicode(obj, encoding)

    return obj

def retrieve_venue_map(conn):
    """
    Create a map with already retrieved venue names
    """
    cursor = conn.cursor()
    cursor.execute(select_venue_sql)

    venues = dict()

    for venue in cursor.fetchall():
        venues[venue[0]] = True

    return venues

def fetch_unvisited_locations(conn):
    """
    Fetch locations from instagram that have not been augmented yet.
    """
    cursor = conn.cursor()
    cursor.execute(select_location_sql)

    result = cursor.fetchall()

    return result

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

    def _get_gender_likes(likes):
        male_like = 0
        female_like = 0
        for like in likes:
            if 'gender' not in like:
                continue
            if like['gender'] == 'female':
                female_like += 1
            else:
                male_like += 1

        return female_like, male_like

    def _get_price_tier(attributes):
        for attribute in attributes['groups']:
            if attribute['type'] == 'price':
                for item in attribute['items']:
                    if 'priceTier' in item:
                        return item['priceTier']

        return 0

    venue = dict()
    venue['id'] = data['id']
    venue['name'] = to_unicode_or_bust(data['name'])
    venue['address'] = to_unicode_or_bust(",".join(data['location']['formattedAddress']))
    venue['lat'] = data['location']['lat']
    venue['lng'] = data['location']['lng']
    venue['category_name'] = to_unicode_or_bust(_get_category(data['categories']))
    venue['url'] = data['canonicalUrl']
    venue['stat_checkin'] = data['stats']['checkinsCount'] if hasattr(data, 'stats') else '0'
    venue['stat_user'] = data['stats']['usersCount'] if hasattr(data, 'stats') else '0'
    venue['stat_tip'] = data['stats']['tipCount'] if hasattr(data, 'stats') else '0'
    venue['stat_likes'] = data['likes']['count'] if hasattr(data, 'likes') else '0'
    venue['stat_female_likes'], venue['stat_male_likes'] = \
        _get_gender_likes(data['likes']['groups'])
    venue['stat_rating'] = data['rating'] if hasattr(data, 'rating') else '0'
    venue['stat_rating_count'] = data['ratingSignals'] if hasattr(data, 'ratingSignals') else '0'
    venue['stat_photo_count'] = data['photos']['count']
    venue['stat_listed'] = data['listed']['count']
    venue['tags'] = to_unicode_or_bust(",".join(data['tags']))
    venue['price'] = _get_price_tier(data['attributes'])

    # id, name, lat, lng, address, url, category_name,
    # stat_checkin, stat_user, stat_tip, stat_likes,
    # stat_rating, stat_rating_count, stat_photo_count,
    # tags, stat_listed, price, stat_female_likes, stat_male_likes
    return (venue['id'], venue['name'], venue['lat'], venue['lng'], venue['address'], venue['url'],
            venue['category_name'], venue['stat_checkin'], venue['stat_user'], venue['stat_tip'],
            venue['stat_likes'], venue['stat_rating'], venue['stat_rating_count'],
            venue['stat_photo_count'], venue['tags'], venue['stat_listed'], venue['price'],
            venue['stat_female_likes'], venue['stat_male_likes'])

def get_foursquare_data(conn, venue_map, locations):
    """
    For each location, access foursquare and retrieve information about
    the venue.
    """
    foursquare_api = fq.Foursquare(client_id=FOURSQUARE_CLIENT_ID,
                                   client_secret=FOURSQUARE_CLIENT_SECRET)

    venues = []
    location_venue_map = []

    cursor = conn.cursor()

    for location in locations:
        try:
            data = foursquare_api.venues.search(params={
                'query': location[0].strip(),
                'll': "%s,%s" % (location[1],location[2]),
                'radius': FOURSQUARE_RADIUS}
            )
            venues_data = data['venues']
            if venues_data:
                if venues_data[0]['id'] not in venue_map:
                    venue = foursquare_api.venues(venues_data[0]['id'])
                    filtered_data = map_venue(venue['venue'])
                    cursor.execute(insert_venue_sql, filtered_data)
                    conn.commit()
                    #venues.append(filtered_data)

                #location_venue_map.append(
                cursor.execute(insert_foreign_key, (venues_data[0]['id'], location[0].strip(),
                                                    location[1], location[2]))
                conn.commit()
            else:
                logging.getLogger(__name__).debug(
                    "Place with name %s has not been found" % location[0].strip()
                )
                cursor.execute(insert_foreign_key, (0, location[0].strip(), location[1], location[2]))
                conn.commit()
                #location_venue_map.append((0, location[0].strip(), location[1], location[2]))
        except fq.RateLimitExceeded:
            logging.getLogger(__name__).error("Rate Limit Exceeded!")
            break;
        except fq.InvalidAuth:
            logging.getLogger(__name__).error("Error while logging in.")
            break;

    # Add all venues to the database
    #cursor.executemany(insert_venue_sql, venues)
    #conn.commit()

    # Update all foreign keys
    #print location_venue_map
    #cursor.executemany(insert_foreign_key, location_venue_map)
    #conn.commit()

def execute():
    # create connection
    conn = connect_db()

    # create map of locations
    venue_map = retrieve_venue_map(conn)

    # fetch 2500 unvisited locations
    locations = fetch_unvisited_locations(conn)

    # retrieve foursquare data for all locations
    get_foursquare_data(conn, venue_map, locations)

if __name__ == "__main__":
    execute()