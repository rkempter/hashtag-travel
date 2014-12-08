/**
 * Created by rkempter on 11/25/14.
 */

$(function() {

    $.getJSON("http://livelyplanet.mooo.com:9998/sets").done(function(sets) {
        ko.applyBindings({
            sets: sets
        });
    });
})