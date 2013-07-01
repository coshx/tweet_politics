//embed_tweet.js

var twitter_api = "https://api.twitter.com/1/statuses/oembed.json?"
    + "align=center"
    + "&maxwidth=500"
    + "&hide_media=false"
    + "&hide_thread=false"
    + "&id=";

$.ajax({
  url: twitter_api + tweet_id,
  dataType: "jsonp",
  jsonpCallback: "embedTweet"
});

function embedTweet(data) {
  $("#tweet").html(data["html"]);
}