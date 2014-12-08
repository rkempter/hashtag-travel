/**
 * Created by rkempter on 11/25/14.
 */

$(function() {

    function ViewModel() {
        var self = this;
        self.cluster = new ko.observableDictionary({
            "name": "",
            "media": []
        });
        self.hasCluster = ko.observable(false);
        self.topic = ko.observable();
    }

    var viewModel = new ViewModel()

    var parisCenter = [48.8588589,2.3470599];
    var $map = $("#map");
    var map = new google.maps.Map($map[0], {
             zoom: 13,
             mapTypeId: google.maps.MapTypeId.ROADMAP,
             center: new google.maps.LatLng(parisCenter[0], parisCenter[1]), // Mozambique
             styles:[{"stylers": [{"saturation": -50},{"lightness": 25}]}]
         });

    $.getJSON("http://livelyplanet.mooo.com:9998/clusters").done(function(geoJson) {
        var overlay = new google.maps.OverlayView();
        overlay.onAdd = function () {

            var layer = d3.select(this.getPanes().overlayMouseTarget).append("div").attr("class", "SvgOverlay");
            var svg = layer.append("svg");
            var cluster_points = svg.append("g").attr("class", "cluster_points");

            overlay.draw = function () {
                var markerOverlay = this;
                var overlayProjection = markerOverlay.getProjection();
                // Turn the overlay projection into a d3 projection
                var googleMapProjection = function (coordinates) {
                    var googleCoordinates = new google.maps.LatLng(coordinates[1], coordinates[0]);
                    var pixelCoordinates = overlayProjection.fromLatLngToDivPixel(googleCoordinates);
                    return [pixelCoordinates.x + 8000, pixelCoordinates.y + 8000];
                }
                path = d3.geo.path().projection(googleMapProjection);
                cluster_points.selectAll("path")
                    .data(geoJson.features)
                    .enter().append("svg:path")
                    .attr("d", path)
                    .attr("class", "marker")
                    .attr("data-bind", "click: clickCounter")
                    .on("click", function(d) { displayCluster(viewModel, d)} );
            };
        };
        overlay.setMap(map);
    });

    function displayCluster(viewModel, d) {
        $.getJSON("http://livelyplanet.mooo.com:9998/cluster/" + d.properties.id).done(function(cluster) {
            viewModel.hasCluster = true;
            for (var key in cluster) {
                if(cluster.hasOwnProperty(key)) {
                    console.log("Set key "+key)
                    viewModel.cluster.set(key, cluster[key])
                }
            }
            console.log(cluster.distribution)
        });
    }

    ko.applyBindings(viewModel)
})
