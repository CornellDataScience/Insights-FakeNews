
var width = 750,
    height = 500;

var fill = d3.scale.category20();

var color = d3.scale.linear()
        .domain([0,5])
        .range(["white","blue"]);

var wordScale = d3.scale.linear().range([10,200]);

d3.csv('../disagree_wordcounts.csv', function(data) {
  var most_frequent = data
    .filter(function(d) {return +d.Frequency > 0; })
    .map(function(d) {return {text: d.Word, size: +d.Frequency};})
    .sort(function(a,b) {return d3.descending(a.size,b.size); })
    .slice(0,200);

    wordScale.domain([
      d3.min(most_frequent, function(d) {return d.size;}),
      d3.max(most_frequent, function(d) {return d.size})
    ]);

    d3.layout.cloud().size([800, 400])
      .words(most_frequent)
      .padding(0)
      .rotate(0)
      .fontSize(function(d) { return wordScale(d.size); })
      .on("end", draw)
      .start();
});


function draw(words) {
    d3.select("body").append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("text-anchor","middle")
            .append("g")
            // without the transform, words words would get cutoff to the left and top, they would
            // appear outside of the SVG area
              .attr("transform", "translate(" + (width / 2 )+ "," + (height / 2 ) +")")
            .selectAll("text")
              .data(words)
            .enter().append("text")
              .style("font-size", function(d) { return d.size - 1 + "px"; })
              .style("fill", function(d, i) { return fill(i); })
              .attr("transform", function(d) {
                  return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
              })
              .text(function(d) { return d.text; });
}
