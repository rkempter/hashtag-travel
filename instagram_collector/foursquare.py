"""
For all locations retrieved from Instagram, get detailed venue information.
"""
from collections import defaultdict

import foursquare

from instagram_collector.config import FOURSQUARE_OAUTH_TOKEN, FOURSQUARE_RADIUS
from instagram_collector.collector import connect_db

insert_venue_sql = """
    INSERT IGNORE INTO
        venues
            (id, name, lat, lng, address, url, category_name,
             stat_checkin, stat_user, stat_tip, stat_likes,
             stat_rating, stat_rating_count, stat_photo_count,
             tags, stat_listed, price, stat_female_likes, stat_male_likes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

insert_tags_sql = """INSERT IGNORE INTO tags (id, tag_name) VALUES (%s, %s);"""

select_venue_sql = """SELECT id FROM venues;"""

# Query for selecting locations that have no information yet
select_location_sql = """
    SELECT DISTINCT
        location_name, location_lat, location_lng
    FROM
        media_events
    WHERE
        venue_id NOT NULL
    ORDER BY
        location_name, location_lat, location_lng
    LIMIT 2500;
"""

insert_foreign_key = """
    UPDATE media_events
    SET venue_id = %s
    WHERE
        location_name = %s AND
        location_lat = %s AND
        location_lng = %s;
"""

def retrieve_venue_map(conn):
    """
    Create a map with already retrieved venue names
    """
    cursor = conn.cursor()
    cursor.execute(select_venue_sql)

    venues = dict()

    for venue in cursor.fetchall():
        venues[venue] = True

    return venues

def fetch_unvisited_locations(conn, offset):
    """
    Fetch locations from instagram that have not been augmented yet.
    """
    cursor = conn.cursor()
    cursor.execute(select_location_sql, (offset,))

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
    venue['name'] = data['name']
    venue['address'] = data['location']['formattedAddress']
    venue['lat'] = data['location']['lat']
    venue['lng'] = data['location']['lng']
    venue['category_name'] = _get_category(data['categories'])
    venue['url'] = data['canonicalUrl']
    venue['stat_checkin'] = data['stats']['checkinsCount'] if hasattr(data, 'stats') else 0
    venue['stat_user'] = data['stats']['usersCount'] if hasattr(data, 'stats') else 0
    venue['stat_tip'] = data['stats']['tipCount'] if hasattr(data, 'stats') else 0
    venue['stat_likes'] = data['likes']['count'] if hasattr(data, 'likes') else 0
    venue['stat_female_likes'], venue['stat_male_likes'] = _get_gender_likes(data['likes']['items'])
    venue['stat_rating'] = data['rating'] if hasattr(data, 'rating') else 0
    venue['stat_rating_count'] = data['ratingSignals'] if hasattr(data, 'ratingSignals') else 0
    venue['stat_photo_count'] = data['photos']['count']
    venue['stat_listed'] = data['listed']['count']
    venue['tags'] = ",".join(data['tags'])
    venue['price'] = _get_price_tier(data['attributes'])

    return venue

def get_foursquare_data(conn, venue_map, locations):
    """
    For each location, access foursquare and retrieve information about
    the venue.
    """
    foursquare_api = foursquare.Foursquare(access_token=FOURSQUARE_OAUTH_TOKEN)

    venues = []
    location_venue_map = []

    for location in locations:
        data = foursquare_api.venues.search(params={
            'query': location[0],
            'll': "%s,%s" % (location[1],location[2]),
            'radius': FOURSQUARE_RADIUS}
        )

        if data:
            if data[0]['id'] not in venue_map:
                venue = foursquare_api.venues(data[0]['id'])
                filtered_data = map_venue(venue)
                venues.append(filtered_data)

            location_venue_map.append((data[0]['id'], location[0], location[1], location[2]))
        else:
            location_venue_map.append((0, location[0], location[1], location[2]))

    cursor = conn.cursor()
    # Add all venues to the database
    cursor.executemany(insert_venue_sql, venues)
    conn.commit()

    # Update all foreign keys
    cursor.executemany(insert_foreign_key, location_venue_map)
    conn.commit()

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
