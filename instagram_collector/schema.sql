CREATE TABLE `media_events` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `user_name` varchar(255) NOT NULL DEFAULT '',
  `user_id` int(11) NOT NULL,
  `tags` text,
  `location_name` varchar(255) DEFAULT '',
  `location_lat` float DEFAULT NULL,
  `location_lng` float DEFAULT NULL,
  `filter` varchar(100) NOT NULL DEFAULT '',
  `created_time` int(20) NOT NULL,
  `image_url` varchar(255) NOT NULL DEFAULT '',
  `media_url` varchar(255) NOT NULL DEFAULT '',
  `text` text NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;